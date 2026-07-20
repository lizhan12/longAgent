"""CriticAgent — 纯评估角色（Evaluator Only）

规则优先 + 快速 LLM 双重审计。
对 Worker 产出进行非宽容性质量审查。

Critic 职责边界：
- ✅ 评估输出质量（PASS / FAIL）
- ✅ 分类失败类型（execution / data_validity / semantic）
- ✅ 列出具体问题（CritiqueIssue）
- ❌ 不决定 RETRY / REPLAN / HITL — 这是 EscalationController 的职责
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .escalation import FailureType

logger = logging.getLogger(__name__)


class CriticVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    # 保留旧枚举值用于向后兼容
    RETRY = "retry"
    REJECT_HITL = "reject_hitl"

    @property
    def is_pass(self) -> bool:
        return self == CriticVerdict.PASS


@dataclass
class CritiqueIssue:
    type: str
    description: str
    severity: str = "medium"
    failure_type: FailureType | None = None
    """映射到的失败类型（供 EscalationController 路由）"""


@dataclass
class CriticReport:
    """纯评估报告 — 不含路由决策

    Attributes:
        verdict: PASS 或 FAIL
        issues: 发现的问题列表
        failure_type: 主导失败类型（用于升级路由）
        summary: 人类可读的评估摘要
    """

    verdict: CriticVerdict
    issues: list[CritiqueIssue] = field(default_factory=list)
    dominant_failure_type: FailureType | None = None
    summary: str = ""

    @property
    def is_pass(self) -> bool:
        return self.verdict.is_pass

    @classmethod
    def passed(cls, summary: str = "评估通过") -> "CriticReport":
        return cls(verdict=CriticVerdict.PASS, summary=summary)

    @classmethod
    def failed(
        cls,
        issues: list[CritiqueIssue],
        summary: str = "",
    ) -> "CriticReport":
        failure_type = cls._classify_dominant(issues)
        return cls(
            verdict=CriticVerdict.FAIL,
            issues=issues,
            dominant_failure_type=failure_type,
            summary=summary or f"{len(issues)} 个问题",
        )

    @staticmethod
    def _classify_dominant(issues: list[CritiqueIssue]) -> FailureType | None:
        if not issues:
            return None
        for issue in issues:
            if issue.failure_type is not None:
                return issue.failure_type
        semantic_types = {"semantic_mismatch", "goal_mismatch", "wrong_output"}
        exec_types = {"execution_error", "tool_failure", "timeout", "empty_output", "too_short"}
        data_types = {"zero_result", "truncated_output", "redundant_search", "high_failure_rate"}

        for issue in issues:
            if issue.type in semantic_types:
                return FailureType.SEMANTIC
        for issue in issues:
            if issue.type in exec_types:
                return FailureType.EXECUTION
        for issue in issues:
            if issue.type in data_types:
                return FailureType.DATA_VALIDITY
        return FailureType.EXECUTION


_ZERO_OUTPUT_PATTERNS = frozenset({
    "无结果", "无法完成", "执行失败", "未知错误",
    "no result", "failed", "cannot complete",
})

_TRUNCATED_PATTERNS = frozenset({
    "结果已截断", "truncated", "[...]",
})


class CriticAgent:
    """评估 Agent — 仅负责质量审查，不做路由决策

    对 Worker 输出进行非宽容性审计：

    1. 规则层（必然执行）：
       - 空输出 / 错误输出 → FailureType.EXECUTION
       - 输出被截断 → FailureType.DATA_VALIDITY
       - 输出过短（疑似未完成）→ FailureType.EXECUTION
       - Worker 用尽所有轮次但无结果 → FailureType.EXECUTION
       - 工具失败率过高 → FailureType.EXECUTION
       - 冗余搜索 → FailureType.DATA_VALIDITY

    2. LLM 层（可选，仅在规则层有中等问题时触发）：
       - 语义一致性检查 → FailureType.SEMANTIC

    3. 输出 CriticReport：
       - PASS → 直接合并到主 Agent
       - FAIL → 交给 EscalationController 决定 RETRY / REPLAN / HITL

    用法:
        critic = CriticAgent(llm_chat_fn=llm.chat)
        report = await critic.review(worker_result, {
            "instruction": "编写斐波那契函数",
            "sub_agent_name": "code_agent",
        })
        if not report.is_pass:
            signal = FailureSignal(
                failure_type=report.dominant_failure_type,
                source="critic",
                ...
            )
            decision = escalation_controller.decide(signal)
    """

    def __init__(
        self,
        llm_chat_fn: Any = None,
        strategy_critique: Any = None,
    ) -> None:
        self._llm_chat = llm_chat_fn
        self._strategy_critique = strategy_critique

    async def review(
        self,
        result: Any,
        task_context: dict[str, Any] | None = None,
    ) -> CriticReport:
        """审查 Worker 产出（纯评估，不含路由决策）

        Args:
            result: WorkerResult 实例
            task_context: 任务上下文
                {
                    "instruction": str,
                    "sub_agent_name": str,
                    ...
                }

        Returns:
            CriticReport 评估报告
        """
        task_context = task_context or {}

        issues = self._rule_check(result, task_context)

        if issues and self._llm_chat is not None:
            semantic_ok = await self._llm_semantic_check(
                getattr(result, "output", ""),
                task_context,
            )
            if not semantic_ok:
                issues.append(CritiqueIssue(
                    type="semantic_mismatch",
                    description="LLM 语义检查未通过：输出与指令不相关",
                    severity="high",
                    failure_type=FailureType.SEMANTIC,
                ))

        if not issues:
            return CriticReport.passed()

        return CriticReport.failed(issues)

    async def review_simple(self, result: Any, task_context: dict[str, Any] | None = None) -> CriticVerdict:
        """便捷接口：返回简单 verdict（向后兼容）"""
        report = await self.review(result, task_context)
        return report.verdict

    def _rule_check(
        self,
        result: Any,
        task_context: dict[str, Any],  # noqa: ARG002
    ) -> list[CritiqueIssue]:
        """规则层检查（零延迟）"""
        issues: list[CritiqueIssue] = []

        output = getattr(result, "output", "") or ""
        success = getattr(result, "success", False)
        error = getattr(result, "error", "") or ""
        rounds = getattr(result, "rounds", 0)
        tool_history = getattr(result, "tool_history", []) or []

        if error:
            issues.append(CritiqueIssue(
                type="execution_error",
                description=f"Worker 执行出错: {error[:150]}",
                severity="high",
                failure_type=FailureType.EXECUTION,
            ))

        if not output or not output.strip():
            if not success:
                issues.append(CritiqueIssue(
                    type="empty_output",
                    description="Worker 输出为空且标记为失败",
                    severity="critical" if not tool_history else "high",
                    failure_type=FailureType.EXECUTION,
                ))
            else:
                issues.append(CritiqueIssue(
                    type="empty_output",
                    description="Worker 输出为空但标记为成功",
                    severity="medium",
                    failure_type=FailureType.EXECUTION,
                ))

        for pattern in _ZERO_OUTPUT_PATTERNS:
            if pattern in output[:200]:
                issues.append(CritiqueIssue(
                    type="zero_result",
                    description=f"Worker 输出表明无结果: '{pattern}'",
                    severity="medium" if tool_history else "high",
                    failure_type=FailureType.DATA_VALIDITY,
                ))
                break

        for pattern in _TRUNCATED_PATTERNS:
            if pattern in output:
                issues.append(CritiqueIssue(
                    type="truncated_output",
                    description="Worker 输出被截断，可能不完整",
                    severity="medium",
                    failure_type=FailureType.DATA_VALIDITY,
                ))
                break

        if success and len(output) < 20 and not tool_history:
            issues.append(CritiqueIssue(
                type="too_short",
                description=f"Worker 输出过短 ({len(output)} 字符)，疑似未完成",
                severity="medium",
                failure_type=FailureType.EXECUTION,
            ))

        if rounds >= 5 and not success and not output.strip():
            issues.append(CritiqueIssue(
                type="max_rounds_no_result",
                description=f"Worker 用尽 {rounds} 轮但无产出",
                severity="high",
                failure_type=FailureType.EXECUTION,
            ))

        strategy_issues = self._check_tool_usage(tool_history)
        issues.extend(strategy_issues)

        return issues

    def _check_tool_usage(self, tool_history: list[dict[str, Any]]) -> list[CritiqueIssue]:
        """检查工具使用策略问题"""
        issues: list[CritiqueIssue] = []
        if not tool_history:
            return issues

        search_calls = [t for t in tool_history if t.get("name") == "tavily_search"]
        if len(search_calls) >= 3:
            has_other = any(
                t.get("name") not in ("tavily_search",)
                for t in tool_history
            )
            if not has_other:
                issues.append(CritiqueIssue(
                    type="redundant_search",
                    description=f"Worker 连续搜索 {len(search_calls)} 次但未使用其他工具",
                    severity="medium",
                    failure_type=FailureType.DATA_VALIDITY,
                ))

        all_errors = [t for t in tool_history if t.get("error") or (
            isinstance(t.get("result"), str) and (
                "失败" in t.get("result", "") or "错误" in t.get("result", "")
            )
        )]
        failure_rate = len(all_errors) / max(len(tool_history), 1)
        if failure_rate >= 0.5 and len(tool_history) >= 3:
            issues.append(CritiqueIssue(
                type="high_failure_rate",
                description=f"Worker 工具失败率过高 ({failure_rate:.0%})",
                severity="high",
                failure_type=FailureType.EXECUTION,
            ))

        return issues

    async def _llm_semantic_check(self, output: str, task_context: dict[str, Any]) -> bool:
        """LLM 语义一致性检查（EDGE tier 快速模型）

        检查 Worker 输出是否与任务指令相关。
        """
        if not output or not self._llm_chat:
            return True

        instruction = task_context.get("instruction", "")
        if not instruction:
            return True

        prompt = (
            f"任务指令: {instruction}\n\n"
            f"Worker 输出（前 500 字符）:\n{output[:500]}\n\n"
            f"请判断上述输出是否与任务指令相关。只回答 YES 或 NO。"
        )

        try:
            response = await self._llm_chat(
                [{"role": "user", "content": prompt}],
                purpose="critic",
                temperature=0.0,
                max_tokens=10,
            )
            content = getattr(response, "content", "") or ""
            return "YES" in content.upper() and "NO" not in content.upper()
        except Exception:
            return True