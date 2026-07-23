"""IRParser - IR 解析器

从 LLM 输出解析 PlanIR，支持修复和重试。
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import ValidationError

from .plan_ir import ActionType, PlanIR
from .repair_strategies import DEFAULT_REPAIR_STRATEGIES

logger = logging.getLogger(__name__)


@dataclass
class IRRepair:
    """修复记录"""

    strategy: str
    success: bool
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    description: str = ""


class IRRepairStrategy(ABC):
    """修复策略基类

    注意：该接口与 repair_strategies.py 中的 IRRepairStrategy 必须保持一致。
    """

    name: str = ""

    @abstractmethod
    def can_repair(self, data: dict[str, Any], error: str) -> bool:
        ...

    @abstractmethod
    def repair(self, data: dict[str, Any]) -> dict[str, Any]:
        ...


DEFAULT_STRATEGIES: list[IRRepairStrategy] = []


class IRParseStatus(str, Enum):
    """解析状态"""

    SUCCESS = "success"
    REPAIRABLE = "repairable"
    UNPARSEABLE = "unparseable"


@dataclass
class ParseMetrics:
    """解析指标"""

    total: int = 0
    success: int = 0
    repairable: int = 0
    unparseable: int = 0
    fast_path_hits: int = 0
    strategies_applied: dict[str, int] = field(default_factory=dict)

    def record(self, status: IRParseStatus, fast_path: bool = False, strategy_name: str | None = None) -> None:
        self.total += 1
        if status == IRParseStatus.SUCCESS:
            self.success += 1
        elif status == IRParseStatus.REPAIRABLE:
            self.repairable += 1
        else:
            self.unparseable += 1
        if fast_path:
            self.fast_path_hits += 1
        if strategy_name:
            self.strategies_applied[strategy_name] = self.strategies_applied.get(strategy_name, 0) + 1

    def summary(self) -> str:
        if self.total == 0:
            return "ParseMetrics: no data"
        rate_s = self.success / self.total * 100
        rate_r = self.repairable / self.total * 100
        rate_f = self.fast_path_hits / self.total * 100
        return (
            f"ParseMetrics: total={self.total}, "
            f"success={rate_s:.1f}%, repairable={rate_r:.1f}%, "
            f"fast_path={rate_f:.1f}%"
        )


@dataclass
class IRParseResult:
    """解析结果"""

    status: IRParseStatus
    plan: PlanIR | None = None
    repairs: list[IRRepair] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw_text: str = ""
    retry_count: int = 0

    @property
    def success(self) -> bool:
        return self.status == IRParseStatus.SUCCESS and self.plan is not None


class IRParser:
    """IR 解析器

    执行三轮解析流程：
    1. 快速路径：Structured Outputs 模式下直接 json.loads + Pydantic 验证
    2. 标准路径：_extract_json 提取 + Pydantic 验证
    3. 修复路径：尝试修复策略并重新解析

    Attributes:
        max_retries: 最大重试次数
        strategies: 修复策略列表
        metrics: 解析指标
    """

    def __init__(
        self,
        max_retries: int = 3,
        strategies: list[IRRepairStrategy] | None = None,
    ) -> None:
        self.max_retries = max_retries
        # 如果未传入策略，从 DEFAULT_REPAIR_STRATEGIES 加载
        from .repair_strategies import DEFAULT_REPAIR_STRATEGIES as _rs

        self.strategies = strategies or _rs
        self.metrics = ParseMetrics()

    def parse(self, llm_output: str, structured_output: bool = False) -> IRParseResult:
        """解析 LLM 输出

        Args:
            llm_output: LLM 输出文本
            structured_output: 是否使用 Structured Outputs 模式（跳过 _extract_json）

        Returns:
            解析结果
        """
        raw_text = llm_output
        errors: list[str] = []
        data: Any = None

        if structured_output:
            try:
                data = json.loads(llm_output)
                plan, parse_errors = self._try_parse(data)
                if plan is not None:
                    self.metrics.record(IRParseStatus.SUCCESS, fast_path=True)
                    logger.info("IR parse (fast path): status=SUCCESS")
                    return IRParseResult(
                        status=IRParseStatus.SUCCESS,
                        plan=plan,
                        raw_text=raw_text,
                    )
                errors.extend(parse_errors)
            except json.JSONDecodeError as e:
                errors.append(f"JSON 解析错误（structured mode）: {e}")

        json_text = llm_output if structured_output else self._extract_json(llm_output)
        if not json_text:
            self.metrics.record(IRParseStatus.UNPARSEABLE)
            return IRParseResult(
                status=IRParseStatus.UNPARSEABLE,
                errors=["无法从输出中提取 JSON"],
                raw_text=raw_text,
            )

        if not structured_output:
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as e:
                # 尝试修复截断的 JSON（超时导致的不完整输出）
                repaired = self._try_repair_truncated_json(json_text)
                if repaired:
                    try:
                        data = json.loads(repaired)
                    except json.JSONDecodeError:
                        data = {"_raw_text": json_text}
                        errors.append(f"JSON 解析错误: {e}")
                else:
                    data = {"_raw_text": json_text}
                    errors.append(f"JSON 解析错误: {e}")
        elif data is None:
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as e:
                data = {"_raw_text": json_text}
                errors.append(f"JSON 解析错误（structured fallback）: {e}")

        plan, parse_errors = self._try_parse(data)
        if plan is not None:
            self.metrics.record(IRParseStatus.SUCCESS)
            logger.info("IR parse: status=SUCCESS, repairs=0")
            return IRParseResult(
                status=IRParseStatus.SUCCESS,
                plan=plan,
                raw_text=raw_text,
            )

        errors.extend(parse_errors)

        repaired_data, repairs = self._attempt_repairs(data, parse_errors)
        if repairs:
            plan, parse_errors = self._try_parse(repaired_data)
            if plan is not None:
                for r in repairs:
                    self.metrics.record(IRParseStatus.REPAIRABLE, strategy_name=r.strategy)
                return IRParseResult(
                    status=IRParseStatus.REPAIRABLE,
                    plan=plan,
                    repairs=repairs,
                    raw_text=raw_text,
                )
            errors.extend(parse_errors)

        self.metrics.record(IRParseStatus.UNPARSEABLE)
        logger.info(
            "IR parse: status=%s, repairs=%d, errors=%d, metrics=%s",
            "REPAIRABLE" if repairs else "UNPARSEABLE",
            len(repairs),
            len(errors),
            self.metrics.summary(),
        )

        return IRParseResult(
            status=IRParseStatus.UNPARSEABLE,
            repairs=repairs,
            errors=errors,
            raw_text=raw_text,
        )

    # 已知工具名 → 自动映射为 call_tool
    _KNOWN_TOOL_NAMES = frozenset({
        "read_skill_md", "read_file", "write_file", "execute_code",
        "execute_file", "delete_file", "list_files", "query_weather",
        "tavily_search", "web_search", "search", "get_current_time",
    })

    def _try_parse(self, data: dict[str, Any]) -> tuple[PlanIR | None, list[str]]:
        """尝试解析数据

        Args:
            data: 数据字典

        Returns:
            (解析结果, 错误列表)
        """
        errors: list[str] = []

        # 预处理：将未知的 action（如 read_skill_md）自动映射为 call_tool
        data = self._normalize_actions(data)

        try:
            plan = PlanIR.model_validate(data)
            return plan, []
        except ValidationError as e:
            errors.append(f"Schema 验证错误: {e}")
        except Exception as e:
            errors.append(f"解析错误: {e}")

        return None, errors

    def _normalize_actions(self, data: dict[str, Any]) -> dict[str, Any]:
        """将未知的 action 自动映射为 call_tool + tool_name

        LLM 经常将工具名（如 read_skill_md）直接用作 action，
        而不是按照规范使用 call_tool + tool_name。
        此方法自动修复这类问题。
        """
        import copy
        data = copy.deepcopy(data)

        steps = data.get("steps", [])
        if not isinstance(steps, list):
            return data

        valid_actions = {e.value for e in ActionType}

        for step in steps:
            if not isinstance(step, dict):
                continue
            action = step.get("action", "")
            if action in valid_actions:
                continue

            # 未知 action → 映射为 call_tool
            # 1. 已知工具名（在白名单中）
            # 2. 看起来像工具名的非空字符串（包含下划线或全小写字母）
            should_map = action in self._KNOWN_TOOL_NAMES
            if not should_map and action and ("_" in action or action.islower() or action.isalpha()):
                should_map = True
                logger.info("自动映射未知 action 为 call_tool: %s", action)

            if should_map:
                args = step.get("args", {})
                if not isinstance(args, dict):
                    args = {}
                # 将原 action 作为 tool_name
                if "tool_name" not in args:
                    args["tool_name"] = action
                # 将 step 的其他参数移入 parameters
                params = args.get("parameters", {})
                if not isinstance(params, dict):
                    params = {}
                for key in list(args.keys()):
                    if key not in ("tool_name", "parameters"):
                        params[key] = args.pop(key)
                args["parameters"] = params
                step["args"] = args
                step["action"] = "call_tool"
                logger.info("自动修复 action: %s → call_tool(tool_name=%s)", action, args.get("tool_name"))
            elif not action:
                # 空 action — 标记为跳过
                step["action"] = "skip"
                step["args"] = {"reason": "空 action，可能因 LLM 超时导致计划不完整"}
                logger.warning("检测到空 action，标记为 skip")

        return data

    def _try_repair_truncated_json(self, json_text: str) -> str | None:
        """尝试修复截断的 JSON（token 限制导致的不完整输出）

        处理三种截断场景：
        1. 字符串中间断开（最常见）→ 闭合字符串 + 补齐括号
        2. 对象/数组中间断开 → 回退到最后一个完整元素 + 补齐括号
        3. 顶层断开 → 补齐括号
        """
        if not json_text.strip().startswith("{"):
            return None

        # 策略1: 先尝试闭合未闭合的字符串，再补齐括号（LIFO 顺序）
        candidate = self._close_open_strings_and_brackets(json_text)
        if candidate:
            return candidate

        # 策略2: 回退到最后一个完整的闭合元素
        import re

        candidates: list[int] = []
        # 找完整的 } 后跟逗号、换行或 ] 的位置
        for m in re.finditer(r'\}(?:\s*,|\s*\]|\s*\})', json_text):
            candidates.append(m.end() - 1)
        for m in re.finditer(r'\](?:\s*,|\s*\})', json_text):
            candidates.append(m.end() - 1)
        for m in re.finditer(r'\}\s*$', json_text):
            candidates.append(m.end() - 1)

        if not candidates:
            return None

        last_end = max(candidates)
        truncated = json_text[: last_end + 1]

        # 确保截断点不是原文本末尾（否则文本本身已完整）
        if last_end >= len(json_text) - 1:
            return None

        open_braces = truncated.count("{") - truncated.count("}")
        open_brackets = truncated.count("[") - truncated.count("]")
        if open_braces < 0 or open_brackets < 0:
            return None
        if open_braces == 0 and open_brackets == 0:
            return None  # 已完整

        truncated += "]" * open_brackets + "}" * open_braces
        try:
            json.loads(truncated)
            return truncated
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _close_open_strings_and_brackets(text: str) -> str | None:
        """闭合未闭合的字符串并补齐缺失的括号。

        逐字符扫描，跟踪字符串内/外状态和括号的开闭顺序。
        到达文本末尾时：
        - 如果在字符串内，先补一个闭合引号
        - 然后按 LIFO 顺序补齐所有未闭合的括号
        """
        in_string = False
        escape = False
        # 用栈记录括号开闭顺序（LIFO），以正确顺序补齐
        bracket_stack: list[str] = []

        for c in text:
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c == "{":
                    bracket_stack.append("{")
                elif c == "}":
                    if bracket_stack and bracket_stack[-1] == "{":
                        bracket_stack.pop()
                elif c == "[":
                    bracket_stack.append("[")
                elif c == "]":
                    if bracket_stack and bracket_stack[-1] == "[":
                        bracket_stack.pop()

        # 无需修复
        if not in_string and not bracket_stack:
            return None

        repaired = text
        if in_string:
            repaired += '"'

        # 按 LIFO 顺序补齐闭合括号
        for b in reversed(bracket_stack):
            repaired += "]" if b == "[" else "}"

        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            return None

    def _extract_json(self, text: str) -> str | None:
        """从文本中提取 JSON

        支持以下格式：
        1. 纯 JSON
        2. Markdown 代码块包裹的 JSON
        3. 混合文本中的 JSON

        Args:
            text: 输入文本

        Returns:
            提取的 JSON 字符串，如果无法提取则返回 None
        """
        text = text.strip()

        # 1. 先尝试 Markdown 代码块
        code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        if matches:
            for match in matches:
                if "{" in match or "[" in match:
                    return match.strip()

        # 2. 如果整个文本以 { 或 [ 开头，直接返回（纯 JSON）
        if text.startswith("{") or text.startswith("["):
            return text

        # 3. 在混合文本中提取最外层 JSON 对象
        # 使用括号匹配来找到最外层的完整 JSON 对象
        first_brace = text.find("{")
        if first_brace == -1:
            first_brace = text.find("[")
        if first_brace != -1:
            depth = 0
            in_string = False
            escape = False
            for i in range(first_brace, len(text)):
                c = text[i]
                if escape:
                    escape = False
                    continue
                if c == "\\":
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == "{" or c == "[":
                    depth += 1
                elif c == "}" or c == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = text[first_brace : i + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            continue

            # 深度未归零 → JSON 被截断，尝试修复
            truncated = text[first_brace:]
            try:
                repaired = self._try_repair_truncated_json(truncated)
                if repaired:
                    return repaired
            except Exception:
                pass

        return None

    def _attempt_repairs(
        self,
        data: dict[str, Any],
        errors: list[str],
    ) -> tuple[dict[str, Any], list[IRRepair]]:
        """尝试修复数据

        Args:
            data: 原始数据
            errors: 错误列表

        Returns:
            (修复后的数据, 修复记录列表)
        """
        repairs: list[IRRepair] = []

        for strategy in self.strategies:
            strategy_name = getattr(strategy, "name", strategy.__class__.__name__)
            for error_msg in errors:
                try:
                    if strategy.can_repair(data, error_msg):
                        before = dict(data)
                        data = strategy.repair(data)
                        repairs.append(IRRepair(
                            strategy=strategy_name,
                            success=True,
                            before=before,
                            after=dict(data),
                            description=f"应用修复策略 {strategy_name}: {error_msg[:80]}",
                        ))
                        break  # 只匹配第一个适用的错误
                except Exception as e:
                    logger.debug("修复策略 %s 执行异常: %s", strategy_name, e)
                    continue

        return data, repairs

    def build_retry_prompt(self, original: str, errors: list[str]) -> str:
        """构建重试 Prompt

        Args:
            original: 原始输出
            errors: 错误列表

        Returns:
            重试 Prompt
        """
        from .plan_ir import ActionType

        error_text = "\n".join(f"- {e}" for e in errors)
        action_types = ", ".join(e.value for e in ActionType)

        dep_guidance = ""
        if any("依赖关系" in e or "fallback" in e for e in errors):
            dep_guidance = (
                "\n\n⚠️ 依赖关系问题专属指引：\n"
                "- depends_on 只能引用已经存在的 step_id\n"
                "- fallback_step 只能引用已经存在的 step_id\n"
                "- 如果只有 N 个步骤(step_1 到 step_N)，你不能引用 step_{N+1}\n"
                "- 如果不需要回退步骤，请设置 fallback_step 为 null"
            )

        # 检测是否输出了非计划内容（如直接回答了问题）
        non_plan_hint = ""
        if any("plan_id" in e or "goal" in e for e in errors):
            non_plan_hint = (
                "\n\n⚠️ 你的输出不像是执行计划，而是直接回答了用户的问题。\n"
                "请生成一个结构化执行计划（JSON），包含 plan_id、goal 和 steps 三个必需字段，"
                "而不是直接回答问题。\n"
                "例如：\n"
                '```json\n'
                '{\n'
                '  "plan_id": "plan_weather_compare",\n'
                '  "goal": "对比杭州和苏州未来一周的天气",\n'
                '  "steps": [\n'
                '    {"step_id": "step_1", "action": "call_tool", "args": {"tool_name": "query_weather", "parameters": {"city": "杭州"}}}\n'
                '  ]\n'
                '}\n'
                '```\n'
            )

        return f"""之前的输出存在以下问题：

{error_text}
{dep_guidance}
{non_plan_hint}

原始输出：
```
{original}
```

请修正以上问题，输出有效的 JSON 格式。确保：
1. JSON 语法正确
2. action 类型必须是以下之一：{action_types}
3. 每个 step 必须有唯一的 step_id
4. depends_on 和 fallback_step 引用的 step_id 必须存在
5. 所有必需字段都已填写

请直接输出修正后的 JSON，不要包含其他解释。"""


def parse_ir(llm_output: str, max_retries: int = 3) -> IRParseResult:
    """便捷函数：解析 LLM 输出为 PlanIR

    Args:
        llm_output: LLM 输出文本
        max_retries: 最大重试次数

    Returns:
        解析结果
    """
    parser = IRParser(max_retries=max_retries)
    return parser.parse(llm_output)
