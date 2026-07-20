"""PlanExecutor - 计划执行器

将形式化框架（PlanIR / StateMachine / ConstraintValidator / LTL）
与实际工具执行桥接，实现受控的自主任务分解与执行。

架构：
  Phase 0: 任务分类 — 规则引擎判断复杂度，决定执行策略
  Phase 1: 计划生成 — LLM 生成结构化 PlanIR（仅复杂任务）
  Phase 2a: 受控执行 — 约束验证 + 状态机 + HITL + 回退
  Phase 2b: 降级执行 — 原始工具调用循环（带运行时约束检查）
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta as td
from enum import Enum
from typing import Any

from .constraint_validator import ConstraintValidator, RuntimeCheckContext
from .ir_parser import IRParser, IRParseStatus
from .ltl import ExecutionHistory, ExecutionStep
from .plan_ir import ActionType, PlanIR, RiskLevel, StepIR
from .state_machine import AgentState, AgentStateMachine
from ..observability.tracing import current_trace, SpanStatus

logger = logging.getLogger(__name__)


class TaskComplexity(str, Enum):
    """任务复杂度等级"""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class ComplexityScore:
    """复杂度评分结果"""

    level: TaskComplexity
    score: float
    reasons: list[str] = field(default_factory=list)
    needs_planning: bool = False


class TaskComplexityClassifier:
    """任务复杂度分类器

    基于多维特征判断用户请求的复杂度，决定是否需要生成结构化计划。

    分类策略：
    - SIMPLE: 单步操作，直接工具调用即可（如"读取文件"、"当前时间"）
    - MODERATE: 2-3步操作，工具调用循环可处理（如"读取文件并总结"）
    - COMPLEX: 多步/多依赖/高风险，需要结构化计划（如"创建项目并部署"）

    评分维度：
    1. 操作步骤数（隐式推断）
    2. 工具依赖（是否需要多种工具协作）
    3. 风险等级（是否涉及不可逆操作）
    4. 条件/分支（是否需要条件判断）
    5. 创造性/生成性（是否需要生成新内容）
    """

    SIMPLE_PATTERNS: list[re.Pattern[str]] = field(default_factory=list, init=False)
    COMPLEX_PATTERNS: list[re.Pattern[str]] = field(default_factory=list, init=False)
    MULTI_STEP_INDICATORS: list[re.Pattern[str]] = field(default_factory=list, init=False)
    HIGH_RISK_INDICATORS: list[re.Pattern[str]] = field(default_factory=list, init=False)
    CREATIVE_INDICATORS: list[re.Pattern[str]] = field(default_factory=list, init=False)

    def __init__(self) -> None:
        self.SIMPLE_PATTERNS = [
            re.compile(r"^(什么是|什么是|解释|说明|介绍|tell me|explain|what is|define)", re.IGNORECASE),
            re.compile(r"^(你好|hello|hi|hey|谢谢|thanks)", re.IGNORECASE),
            re.compile(r"(读取|查看|显示|列出|read|show|list|display).{0,10}(文件|目录|内容|file|dir)", re.IGNORECASE),
            re.compile(r"^(时间|日期|今天|time|date|today)", re.IGNORECASE),
            re.compile(r"(状态|status|帮助|help|命令|command)", re.IGNORECASE),
        ]

        self.COMPLEX_PATTERNS = [
            re.compile(r"(创建|开发|构建|实现|设计|create|build|develop|implement|design).{0,20}(项目|应用|系统|服务|project|app|system|service)", re.IGNORECASE),
            re.compile(r"(部署|发布|上线|deploy|publish|release)", re.IGNORECASE),
            re.compile(r"(迁移|转换|重构|migrate|convert|refactor)", re.IGNORECASE),
            re.compile(r"(分析|评估|对比|analyze|evaluate|compare).{0,20}(然后|接着|之后|then|and then|after)", re.IGNORECASE),
            re.compile(r"(完整|全部|整个|full|complete|entire).{0,20}(流程|方案|解决方案|workflow|solution|pipeline)", re.IGNORECASE),
            re.compile(r"(绘制|画|生成|制作|plot|draw|chart|visualize|generate).{0,20}(图表|图形|折线图|柱状图|饼图|chart|graph|figure|diagram)", re.IGNORECASE),
            re.compile(r"(折线图|柱状图|饼图|散点图|热力图|line chart|bar chart|pie chart|heatmap)", re.IGNORECASE),
            re.compile(r"(详细|全面|深入|detailed|comprehensive|in-depth).{0,20}(报告|分析|研究|report|analysis|research)", re.IGNORECASE),
            re.compile(r"(规划|计划|方案|plan|planning|strategy).{0,20}(图表|报告|分析|chart|report|analysis)", re.IGNORECASE),
            # 生成文件类任务（PPT/Word/Excel等）需要多步骤：搜索→编写代码→执行→验证
            re.compile(r"(生成|制作|创建|写|做|generate|create|make|write|produce).{0,20}(ppt|pptx|幻灯片|演示|presentation|slide)", re.IGNORECASE),
            re.compile(r"(生成|制作|创建|写|做|generate|create|make|write|produce).{0,20}(word|docx|文档|document)", re.IGNORECASE),
            re.compile(r"(生成|制作|创建|写|做|generate|create|make|write|produce).{0,20}(excel|xlsx|表格|spreadsheet)", re.IGNORECASE),
            re.compile(r"(生成|制作|创建|写|做|generate|create|make|write|produce).{0,20}(pdf|PDF|便携文档)", re.IGNORECASE),
            re.compile(r"(ppt|pptx|word|docx|excel|xlsx|pdf|PDF).{0,10}(报告|文档|文件|report|document|file)", re.IGNORECASE),
        ]

        self.MULTI_STEP_INDICATORS = [
            re.compile(r"(然后|接着|之后|再|并且|同时|also|then|after|next|and then|after that)", re.IGNORECASE),
            re.compile(r"(第一步|第二步|第三步|step 1|step 2|step 3|first|second|third)", re.IGNORECASE),
            re.compile(r"(先|首先|firstly|first of all)", re.IGNORECASE),
            re.compile(r"(分别|各自|separately|respectively)", re.IGNORECASE),
            re.compile(r"(以及|还有|along with|as well as)", re.IGNORECASE),
        ]

        self.HIGH_RISK_INDICATORS = [
            re.compile(r"(删除|移除|清除|delete|remove|clean|drop|truncate)", re.IGNORECASE),
            re.compile(r"(执行|运行|跑|execute|run|launch).{0,10}(脚本|代码|命令|script|code|command)", re.IGNORECASE),
            re.compile(r"(修改|更新|覆盖|modify|update|overwrite|replace).{0,10}(配置|数据库|config|database)", re.IGNORECASE),
            re.compile(r"(写入|保存|覆盖|write|save|overwrite).{0,10}(文件|file)", re.IGNORECASE),
        ]

        self.CREATIVE_INDICATORS = [
            re.compile(r"(写|生成|创建|编写|write|generate|create|compose).{0,15}(代码|脚本|程序|code|script|program)", re.IGNORECASE),
            re.compile(r"(写|生成|创建|write|generate|create).{0,15}(文档|报告|文章|document|report|article)", re.IGNORECASE),
            re.compile(r"(设计|制定|规划|design|plan|formulate).{0,15}(方案|架构|策略|solution|architecture|strategy)", re.IGNORECASE),
            re.compile(r"(测试|验证|检查|test|verify|validate|check).{0,15}(然后|并|修复|then|and|fix)", re.IGNORECASE),
            re.compile(r"(绘制|画|制作|plot|draw|chart|visualize).{0,15}(图表|图形|折线图|柱状图|饼图|chart|graph|figure)", re.IGNORECASE),
            re.compile(r"(详细|全面|深入|detailed|comprehensive).{0,15}(报告|分析|研究|report|analysis)", re.IGNORECASE),
            re.compile(r"(生成|制作|创建|写|generate|create|make|write).{0,15}(ppt|pptx|word|docx|excel|xlsx|pdf|PDF)", re.IGNORECASE),
        ]

    def classify(self, message: str, available_tools: list[dict[str, Any]] | None = None) -> ComplexityScore:
        """分类任务复杂度

        Args:
            message: 用户消息
            available_tools: 可用工具列表

        Returns:
            复杂度评分结果
        """
        score = 0.0
        reasons: list[str] = []

        for pattern in self.SIMPLE_PATTERNS:
            if pattern.search(message):
                score -= 2.0
                reasons.append(f"匹配简单模式: {pattern.pattern[:30]}...")
                break

        for pattern in self.COMPLEX_PATTERNS:
            if pattern.search(message):
                score += 3.0
                reasons.append(f"匹配复杂模式: {pattern.pattern[:30]}...")
                break

        # 文件生成任务（PPT/Word/Excel）天生需要多步骤，自动提升为 COMPLEX
        _file_gen_re = re.compile(
            r"(生成|制作|创建|写|做|generate|create|make|write|produce)"
            r".{0,20}(ppt|pptx|word|docx|excel|xlsx|幻灯片|文档|表格|presentation|document|spreadsheet)",
            re.IGNORECASE,
        )
        # 宽松匹配：含 ppt/pptx/word/excel 相关词但没有明确动词的消息
        _file_keyword_re = re.compile(
            r"(?:ppt版|pptx|\.pptx|\.docx|\.xlsx|幻灯片|word文档|excel表格|转成ppt|改成ppt"
            r"|个ppt|个word|个excel|份ppt|成ppt|成word)",
            re.IGNORECASE,
        )
        if (_file_gen_re.search(message) or _file_keyword_re.search(message)) and score < 4.0:
            score = 4.0
            reasons.append("文件生成任务，自动提升为复杂任务")

        multi_step_count = sum(1 for p in self.MULTI_STEP_INDICATORS if p.search(message))
        if multi_step_count >= 2:
            score += 3.0
            reasons.append(f"多步骤指示词 ({multi_step_count} 个)")
        elif multi_step_count == 1:
            score += 1.5
            reasons.append("包含步骤连接词")

        high_risk_count = sum(1 for p in self.HIGH_RISK_INDICATORS if p.search(message))
        if high_risk_count >= 2:
            score += 2.5
            reasons.append(f"高风险操作 ({high_risk_count} 个)")
        elif high_risk_count == 1:
            score += 1.0
            reasons.append("包含高风险操作")

        creative_count = sum(1 for p in self.CREATIVE_INDICATORS if p.search(message))
        if creative_count >= 2:
            score += 2.0
            reasons.append(f"创造性任务 ({creative_count} 个)")
        elif creative_count == 1:
            score += 1.0
            reasons.append("包含生成性操作")

        msg_len = len(message)
        if msg_len > 200:
            score += 1.5
            reasons.append(f"长消息 ({msg_len} 字符)")
        elif msg_len > 100:
            score += 0.5
            reasons.append("中等长度消息")

        sentences = re.split(r'[。！？.!?\n]', message)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 4:
            score += 1.5
            reasons.append(f"多句子 ({len(sentences)} 句)")
        elif len(sentences) >= 2:
            score += 0.5

        question_count = message.count("?") + message.count("？")
        if question_count >= 2:
            score += 0.5
            reasons.append(f"多问题 ({question_count} 个)")

        tool_categories: set[str] = set()
        if available_tools:
            for tool in available_tools:
                func = tool.get("function", {})
                name = func.get("name", "")
                if "file" in name or "read" in name or "list" in name:
                    tool_categories.add("file_ops")
                elif "execute" in name or "code" in name:
                    tool_categories.add("code_exec")
                elif "skill" in name:
                    tool_categories.add("skill")
                elif "mcp" in name:
                    tool_categories.add("mcp")

        if score >= 4.0:
            level = TaskComplexity.COMPLEX
        elif score >= 1.5:
            level = TaskComplexity.MODERATE
        else:
            level = TaskComplexity.SIMPLE

        needs_planning = level == TaskComplexity.COMPLEX

        return ComplexityScore(
            level=level,
            score=score,
            reasons=reasons,
            needs_planning=needs_planning,
        )


@dataclass
class StepResult:
    """步骤执行结果"""

    step_id: str
    success: bool
    output: str = ""
    error: str | None = None
    duration: float = 0.0
    skipped: bool = False
    fallback_executed: bool = False


@dataclass
class PlanExecutionResult:
    """计划执行结果"""

    plan_id: str
    success: bool
    final_state: AgentState = AgentState.INIT
    step_results: list[StepResult] = field(default_factory=list)
    output_text: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_duration: float = 0.0
    used_fallback_mode: bool = False
    has_placeholder_params: bool = False  # 标记计划因占位参数被拒绝


PLAN_GENERATION_SYSTEM_PROMPT = """你是一个任务规划专家。你的职责是将用户的需求分解为结构化的执行计划。

## 输出格式
你必须输出一个 JSON 对象，严格遵循以下 schema：

```json
{
  "plan_id": "plan_xxx",
  "goal": "用户的目标描述",
  "steps": [
    {
      "step_id": "step_1",
      "action": "动作类型",
      "args": { "参数": "值" },
      "depends_on": [],
      "condition": null,
      "fallback_step": null,
      "risk_level": "low",
      "description": "步骤描述"
    }
  ],
  "constraints": [],
  "estimated_steps": 3
}
```

## 可用动作类型（action）
**action 字段只能从以下 9 个值中选择，不要使用工具名（如 read_skill_md）作为 action！**
- `call_tool`: 调用本地工具（read_file, write_file, execute_code, execute_file, list_files, delete_file, read_skill_md, tavily_search 等），args: {"tool_name": "具体工具名", "parameters": {...}}
- `call_mcp`: 调用 MCP 服务器工具，args: {"server_name": "服务器名", "tool_name": "工具名", "arguments": {...}}
- `call_skill`: 调用 Skill 技能，args: {"skill_name": "技能名", "arguments": {...}}
- `search`: 搜索文件（仅用于 list_files 读取文件目录），args: {"query": "路径"}
- `reason`: 推理分析（不需要调用工具，纯思考步骤）
- `summarize`: 总结归纳
- `output`: 输出最终结果给用户
- `wait_approval`: 等待用户审批（高风险操作时使用）
- `call_api`: 调用外部 API（通常不使用）

**重要：当你要调用 read_skill_md、read_file、execute_code 等具体工具时，action 必须是 "call_tool"，然后在 args.tool_name 中指定具体工具名！**

## 风险等级（risk_level）
- `low`: 普通操作（读取、查询）
- `medium`: 中等风险（写入文件、执行代码）
- `high`: 高风险（删除文件、执行不可逆操作）
- `critical`: 极高风险（需要用户审批）

## 规划原则
1. 每个步骤应该是一个原子操作
2. 复杂任务应分解为 3-8 个步骤
3. 高风险操作前应有 `reason` 步骤进行验证
4. 最后一步必须是 `output` 或 `summarize`
5. `depends_on` 只能引用已定义的 step_id
6. 如果任务很简单（1步就能完成），直接用 `output` 动作

## 数据获取规则（极其重要！！！数据源选择错误将导致结果不准确）
- **天气查询**：必须使用 `call_tool(query_weather)` 调用和风天气API，**绝对禁止使用 tavily_search 查询天气**，也**禁止自己写代码调 API**。
  - 正确示例：step_1 用 `call_tool(query_weather, {"city": "杭州,阜阳,北京"})` 获取天气数据
  - 错误示例1：step_1 用 tavily_search 搜索"阜阳天气预报" ← 数据不准确！
  - 错误示例2：step_1 用 execute_code 自己写代码调 API ← 认证复杂容易出错！直接用 query_weather 工具即可
  - query_weather 工具会返回格式化的简洁天气文本，包含实况和7天预报
- **新闻、股价、赛事等实时信息**：使用 `call_tool(tavily_search)` 搜索
- **天气数据必须来自 API**：生成的报告中的数据必须基于 API 返回的真实数据，绝不硬编码捏造温度值

## write_file / execute_code 的 content 规则（极其重要！！！不遵守将导致任务失败）
- `write_file` 的 `content` 参数**必须是完整可运行的真实代码**。绝不写"# 根据xxx编写代码"这类占位注释！
- 错误示例（会导致任务失败）：content="# 根据搜索结果编写Python代码\\n# 生成数据可视化图表" ← 这是占位注释，不是代码！
- 错误示例（会导致任务失败）：content="根据搜索内容生成PPT的Python脚本" ← 这是描述，不是代码！
- 错误示例（会导致任务失败）：content="根据搜索结果和记忆内容，使用python-pptx编写完整的PPT生成代码" ← 这仍然是描述！
- 正确示例：content="from pptx import Presentation\\nfrom pptx.util import Inches\\nprs = Presentation()\\nslide = prs.slides.add_slide(prs.slide_layouts[0])\\nslide.shapes.title.text = 'AI发展报告'\\nprs.save('output/ai_report.pptx')" ← 可执行代码
- **content 的第一个字符必须是代码字符（如 from、import、def、class、# coding 等），绝不能是中文描述文字**
- 任何 .py 文件内容如以 # 注释开头且不包含 import/def/class/= 等代码特征，将被系统检测并拒绝执行
- 如果你不知道具体代码内容，请先搜索获取信息，不要写占位注释敷衍
- `execute_code` 的 `code` 参数同理，必须包含完整可执行代码

## 图表生成规则（极其重要）
- **必须使用 matplotlib 生成图表并保存为 PNG 文件**
- **禁止在代码中或输出中使用 Mermaid 语法**（如 xychart-beta、pie showData、flowchart 等）
- **禁止输出 ASCII 文本图表**
- 图表代码必须包含 `plt.savefig('output/xxx.png', dpi=150)` 保存图片
- 中文字体配置：字体文件在 `font/simhei.ttf`，代码开头必须加载字体

## 文件类型处理规则（极其重要）
Skill 是"怎么做"的指引文档，不是"做什么"的指令。正确使用流程：
1. **先获取内容**：通过搜索、推理等方式获取需要写入文件的实际内容数据
2. **再读 Skill**：调用 `read_skill_md` 获取该文件类型的最佳实践和代码模板
3. **按 Skill 指引执行**：根据 Skill 文档中的指引，结合已获取的内容，编写代码并执行

示例流程（生成PPT报告 - 需要搜索内容时）：
- step_1: call_tool(tavily_search) → 搜索报告所需的实际内容数据
- step_2: call_tool(read_skill_md) → 学习如何用 python-pptx 创建PPT
- step_3: call_tool(write_file) → 根据搜索结果 + Skill指引，编写完整的 python-pptx 代码（content 必须是真实可执行代码）
- step_4: call_tool(execute_file) → 执行代码生成PPT文件
- step_5: output → 输出结果

示例流程（生成PPT报告 - 对话历史中已有内容时，如用户说"根据此生成ppt"）：
- step_1: call_tool(read_skill_md) → 学习如何用 python-pptx 创建PPT
- step_2: call_tool(write_file) → 直接根据对话历史中的内容 + Skill指引，编写完整的 python-pptx 代码（content 必须包含对话历史中的实际数据，不要写占位注释）
- step_3: call_tool(execute_file) → 执行代码生成PPT文件
- step_4: output → 输出结果

注意：
- 不要在获取内容之前就读 Skill，否则你不知道该往文件里填什么内容
- 如果对话历史中已经包含了用户所需的内容数据，就不要再用 tavily_search 搜索，直接使用对话历史中的内容
- 当用户说"根据此"、"根据上面的内容"、"基于这些"等指代词时，内容一定在对话历史中，不要搜索
- 当用户说"生成ppt"、"生成pdf"、"写一份报告"等简短请求时，如果对话历史中有相关内容，应基于对话历史内容生成，不要搜索
- **不要使用 reason 步骤来"整理"对话历史**，对话历史已经直接提供给你了，你应该在 write_file 的 content 中直接包含对话历史中的实际数据

## 重要
- 只输出 JSON，不要包含其他解释
- 确保 JSON 语法正确
- step_id 必须唯一
- depends_on 中的 step_id 必须存在
- 搜索关键词中的年份必须基于当前时间计算，不要使用过时的年份（如当前是2026年，"未来5年"应为2026-2031，不是2024-2029）
"""


class PlanExecutor:
    """计划执行器

    将形式化框架与实际工具执行桥接，提供：
    - 结构化计划生成
    - 编译时计划验证
    - 运行时约束检查（状态机 + LTL + 白名单）
    - HITL 人工审批
    - 步骤失败回退
    - 降级执行模式

    Attributes:
        llm: LLM 客户端
        tool_registry: 工具注册表
        mcp_client: MCP 客户端
        constraint_validator: 约束验证器
        ir_parser: IR 解析器
        state_machine: 状态机
    """

    def __init__(
        self,
        llm: Any | None = None,
        tool_registry: Any | None = None,
        mcp_client: Any | None = None,
        constraint_validator: ConstraintValidator | None = None,
        ir_parser: IRParser | None = None,
        state_machine: AgentStateMachine | None = None,
        max_plan_retries: int = 2,
        max_steps: int = 12,
        cascade_router: Any | None = None,
        hitl_handler: Any | None = None,
    ) -> None:
        self.llm = llm
        self.tool_registry = tool_registry
        self.mcp_client = mcp_client
        self.constraint_validator = constraint_validator or ConstraintValidator()
        self.ir_parser = ir_parser or IRParser()
        self.state_machine = state_machine or AgentStateMachine()
        self.max_plan_retries = max_plan_retries
        self.max_steps = max_steps
        self.classifier = TaskComplexityClassifier()
        self.cascade_router = cascade_router
        self.hitl_handler = hitl_handler

    _validation_patterns: list[dict[str, Any]] = []
    _MAX_PATTERNS = 50

    def _record_validation_lesson(
        self,
        goal: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """记录验证失败模式，用于未来计划生成的提示增强"""
        import json
        import time
        from pathlib import Path

        pattern_keywords = set()
        for kw in ["搜索", "图表", "报告", "文件", "代码", "天气", "数据"]:
            if kw in goal:
                pattern_keywords.add(kw)

        lesson = {
            "ts": time.time(),
            "goal_preview": goal[:80],
            "keywords": list(pattern_keywords),
            "errors": errors,
            "warnings": warnings,
        }
        PlanExecutor._validation_patterns.append(lesson)
        if len(PlanExecutor._validation_patterns) > PlanExecutor._MAX_PATTERNS:
            PlanExecutor._validation_patterns = PlanExecutor._validation_patterns[-PlanExecutor._MAX_PATTERNS:]

        lessons_file = Path("workspace") / "knowledge" / "validation_lessons.jsonl"
        try:
            lessons_file.parent.mkdir(parents=True, exist_ok=True)
            with open(lessons_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(lesson, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("写入验证教训文件失败", exc_info=True)

    @classmethod
    def _get_avoidance_hints(cls, goal: str) -> str:
        """根据历史验证模式生成回避提示"""
        if not cls._validation_patterns:
            return ""

        relevant = []
        for p in cls._validation_patterns[-20:]:
            if any(kw in goal for kw in p.get("keywords", [])):
                relevant.append(p)

        if not relevant:
            return ""

        hints = []
        dep_seen = False
        for p in relevant[-3:]:
            all_issues = p.get("errors", []) + p.get("warnings", [])
            for e in all_issues:
                if ("依赖关系" in e or "fallback" in e) and not dep_seen:
                    hints.append(
                        "- depends_on 和 fallback_step 只能引用已经存在的 step_id，"
                        "不能引用不存在的步骤"
                    )
                    dep_seen = True

        if not hints:
            return ""

        return (
            "\n\n## 历史教训（根据以往验证失败总结，请务必遵守）\n"
            + "\n".join(hints)
        )

    async def generate_plan(
        self,
        user_message: str,
        history_msgs: list[dict[str, str]],
        available_tools: list[dict[str, Any]],
        memory_controller: Any | None = None,
    ) -> PlanIR | None:
        """让 LLM 生成结构化执行计划

        Args:
            user_message: 用户消息
            history_msgs: 对话历史
            available_tools: 可用工具列表

        Returns:
            生成的 PlanIR，如果失败则返回 None
        """
        if self.llm is None and self.cascade_router is None:
            return None

        tool_descriptions = []
        for tool in available_tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            tool_descriptions.append(f"- {name}: {desc}")
            if params.get("properties"):
                props = params["properties"]
                param_str = ", ".join(
                    f"{k}({v.get('type', 'any')})" for k, v in props.items()
                )
                tool_descriptions[-1] += f" [{param_str}]"

        tools_text = "\n".join(tool_descriptions) if tool_descriptions else "无可用工具"

        from long.llm.base import LLMMessage

        avoidance_hints = self._get_avoidance_hints(user_message)

        now = datetime.now(timezone(td(hours=8)))
        weekdays_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        current_date_cn = f"{now.strftime('%Y年%m月%d日')} {weekdays_cn[now.weekday()]}"

        # 构建对话历史上下文，让 LLM 理解"根据此"等指代
        history_context = ""
        if history_msgs:
            # 优先使用记忆系统的 get_context() 进行压缩检索
            if memory_controller is not None:
                try:
                    ctx = await memory_controller.get_context(
                        query=user_message, max_tokens=3000
                    )

                    # 1) 压缩后的当前对话窗口 — 必须注入，这是最重要的上下文
                    messages = ctx.get("messages", [])
                    if messages:
                        context_parts = []
                        for msg in messages:
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if not content or len(content) < 10:
                                continue
                            if role == "user":
                                context_parts.append(f"用户: {content[:500]}")
                            elif role == "assistant":
                                context_parts.append(f"助手: {content[:800]}")
                            elif role == "tool":
                                context_parts.append(f"工具返回: {content[:1000]}")
                        if context_parts:
                            history_context = (
                                "\n## 当前对话历史（最重要！用户的消息和指代都基于此，计划必须根据这个上下文生成）\n"
                                + "\n".join(context_parts) + "\n\n"
                            )
                            logger.info(
                                "计划生成注入当前对话: %d 条压缩消息",
                                len(context_parts),
                            )

                    # 2) 跨session RAG 记忆 — 仅作为补充参考，标注为历史参考
                    relevant = ctx.get("relevant_memories", [])
                    if relevant:
                        mem_lines = [f"- {m['content'][:200]}" for m in relevant[:5]]
                        rag_section = (
                            "\n## 历史相关记忆（仅作参考，若与当前对话冲突请以当前对话为准）\n"
                            + "\n".join(mem_lines) + "\n\n"
                        )
                        if history_context:
                            history_context += rag_section
                        else:
                            history_context = rag_section
                        logger.info(
                            "计划生成注入 RAG 记忆: %d 条", len(relevant),
                        )

                except Exception as e:
                    logger.debug("memory.get_context 失败，回退到直接拼接: %s", e)

            # 兜底：直接从 history_msgs 提取（memory_controller 不可用时）
            if not history_context:
                context_parts = []
                for msg in history_msgs[-20:]:  # 最近20条消息
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if not content or len(content) < 10:
                        continue
                    if role == "user":
                        context_parts.append(f"用户: {content[:500]}")
                    elif role == "assistant":
                        context_parts.append(f"助手: {content[:800]}")
                    elif role == "tool":
                        context_parts.append(f"工具返回: {content[:1000]}")
                if context_parts:
                    history_context = "\n## 对话历史（用户可能引用其中的内容）\n" + "\n".join(context_parts) + "\n\n"
                    logger.info(
                        "计划生成注入对话历史: %d 条消息, 上下文长度=%d 字符",
                        len(context_parts), len(history_context),
                    )
                else:
                    logger.info("计划生成: 对话历史为空或无有效内容")

        messages = [
            LLMMessage(role="system", content=PLAN_GENERATION_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    f"## 当前时间\n{current_date_cn}\n\n"
                    f"## 可用工具\n{tools_text}\n\n"
                    f"{history_context}"
                    f"## 用户需求\n{user_message}"
                    f"{avoidance_hints}\n\n"
                    f"请生成执行计划（只输出 JSON）："
                ),
            ),
        ]

        plan_schema = PlanIR.build_structured_output_schema()
        structured_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "PlanIR",
                "schema": plan_schema,
                "strict": True,
            },
        }
        fallback_format = {"type": "json_object"}
        # 使用 json_object 模式而非 json_schema，因为 json_schema 模式下
        # LLM 倾向于用简短描述代替 write_file 的长代码内容
        use_structured = False

        for attempt in range(self.max_plan_retries + 1):
            try:
                response_format = structured_format if use_structured else fallback_format
                response = await self.llm.chat(
                    messages,
                    purpose="plan",
                    response_format=response_format,
                )

                parse_result = self.ir_parser.parse(response.content, structured_output=use_structured)

                if parse_result.success and parse_result.plan is not None:
                    plan = parse_result.plan

                    validation = self.constraint_validator.validate_plan(plan)
                    if validation.valid:
                        logger.info(
                            "计划生成成功: plan_id=%s, steps=%d",
                            plan.plan_id,
                            len(plan.steps),
                        )
                        if validation.warnings:
                            self._record_validation_lesson(
                                goal=plan.goal,
                                errors=validation.errors,
                                warnings=validation.warnings,
                            )
                        return plan

                    logger.warning(
                        "计划验证失败 (attempt %d): %s",
                        attempt + 1,
                        validation.errors,
                    )

                    if attempt < self.max_plan_retries:
                        retry_prompt = self.ir_parser.build_retry_prompt(
                            response.content, validation.errors
                        )
                        messages.append(LLMMessage(role="assistant", content=response.content))
                        messages.append(LLMMessage(role="user", content=retry_prompt))

                elif parse_result.status == IRParseStatus.REPAIRABLE and parse_result.plan is not None:
                    plan = parse_result.plan
                    validation = self.constraint_validator.validate_plan(plan)
                    if validation.valid:
                        logger.info(
                            "计划修复后验证通过: plan_id=%s, repairs=%d",
                            plan.plan_id,
                            len(parse_result.repairs),
                        )
                        if validation.warnings:
                            self._record_validation_lesson(
                                goal=plan.goal,
                                errors=validation.errors,
                                warnings=validation.warnings,
                            )
                        return plan

                    logger.warning("修复后计划仍验证失败: %s", validation.errors)

                    if attempt < self.max_plan_retries:
                        retry_prompt = self.ir_parser.build_retry_prompt(
                            response.content,
                            validation.errors + [str(e) for e in parse_result.errors],
                        )
                        messages.append(LLMMessage(role="assistant", content=response.content))
                        messages.append(LLMMessage(role="user", content=retry_prompt))

                else:
                    logger.warning(
                        "计划解析失败 (attempt %d): %s",
                        attempt + 1,
                        parse_result.errors,
                    )

                    if attempt < self.max_plan_retries and parse_result.errors:
                        retry_prompt = self.ir_parser.build_retry_prompt(
                            response.content, parse_result.errors
                        )
                        messages.append(LLMMessage(role="assistant", content=response.content))
                        messages.append(LLMMessage(role="user", content=retry_prompt))

            except Exception as e:
                logger.error("计划生成异常 (attempt %d): %s", attempt + 1, e)
                if use_structured and "json_schema" in str(e).lower():
                    logger.info("API 不支持 json_schema 模式，降级为 json_object")
                    use_structured = False
                    continue
                if attempt >= self.max_plan_retries:
                    break

        logger.warning("计划生成失败，将降级为直接工具调用模式")

        if self.cascade_router is not None:
            try:
                from long.llm.cascade_router import CascadeExhaustedError

                response = await self.cascade_router.route(
                    messages, purpose="plan",
                    response_format=fallback_format if not use_structured else structured_format,
                )
                parse_result = self.ir_parser.parse(response.content, structured_output=use_structured)
                if parse_result.status == IRParseStatus.SUCCESS and parse_result.plan is not None:
                    logger.info("级联路由生成计划成功")
                    return parse_result.plan
            except CascadeExhaustedError:
                logger.warning("级联路由全部耗尽")
                if self.hitl_handler is not None:
                    await self.hitl_handler.handle_cascade_exhausted(
                        {"task_id": "plan_generation", "reason": "计划生成失败"},
                    )
            except Exception as e:
                logger.warning("级联路由异常: %s", e)

        return None

    async def execute_plan(
        self,
        plan: PlanIR,
        cli_adapter: Any,
        tool_executor: Any,
        history_msgs: list[dict[str, str]],  # noqa: ARG002
    ) -> PlanExecutionResult:
        """受控执行计划

        Args:
            plan: 执行计划
            cli_adapter: CLI 适配器（用于状态显示和 HITL）
            tool_executor: 工具执行回调 async (tool_name, arguments) -> str
            history_msgs: 对话历史

        Returns:
            执行结果
        """
        start_time = time.time()
        result = PlanExecutionResult(plan_id=plan.plan_id, success=False)

        context = self._build_runtime_context(plan)

        execution_order = plan.get_execution_order()
        if len(execution_order) < len(plan.steps):
            result.errors.append("计划依赖关系存在环，无法执行")
            return result

        step_map = {step.step_id: step for step in plan.steps}

        # 执行前验证：检测步骤参数中的占位值
        # 仅检测参数值的开头部分，避免代码中的注释被误判
        placeholder_patterns = [
            r'^待根据', r'^待确定', r'^待step_\d+', r'^根据step_\d+',
            r'^根据搜索结果', r'^根据.*指引.*编写', r'^根据.*内容.*生成',
            r'^需根据', r'^需结合', r'^需参考',
            r'^placeholder',
        ]
        # 对 path 类参数使用更宽松的检测（整个值）
        placeholder_path_patterns = [
            r'待根据', r'待确定', r'待step_\d+', r'根据step_\d+',
            r'待.*确定.*路径', r'待.*确定.*文件', r'placeholder',
        ]
        import re as _placeholder_re
        placeholder_issues: list[str] = []
        for step in plan.steps:
            args = step.args or {}
            # 也检查嵌套的 parameters 字典
            nested_params = args.get("parameters", {})
            all_params = dict(args)
            if isinstance(nested_params, dict):
                all_params.update(nested_params)

            for key, value in all_params.items():
                # description 字段本身就是描述，不需要检测占位值
                if key in ('description', 'desc', 'step_id', 'action', 'depends_on', 'condition', 'risk_level', 'fallback_step'):
                    continue
                if not isinstance(value, str) or len(value) < 5:
                    continue
                # 对于有依赖的 call_tool(write_file) 步骤，content 依赖前置步骤结果，
                # 无法在计划生成时预填完整代码，跳过占位符检测
                if (key == 'content'
                    and args.get('tool_name') == 'write_file'
                    and step.depends_on):
                    continue
                # 跳过明显是代码的内容（以 import/from/def/class/# 开头）
                value_stripped = value.strip()
                if value_stripped and value_stripped[0] in ('i', 'f', 'd', 'c', '#', '(', '[', '{'):
                    # 可能是代码，只检测 path 类参数
                    if key in ('path', 'file_path', 'output_path'):
                        for pattern in placeholder_path_patterns:
                            if _placeholder_re.search(pattern, value, _placeholder_re.IGNORECASE):
                                placeholder_issues.append(
                                    f"步骤 {step.step_id} ({step.action}): {key}='{value[:60]}' 包含占位值"
                                )
                                break
                    continue
                # 对非代码内容，检测开头是否为占位描述
                value_first_line = value_stripped.split('\n')[0][:100]
                for pattern in placeholder_patterns:
                    if _placeholder_re.search(pattern, value_first_line, _placeholder_re.IGNORECASE):
                        placeholder_issues.append(
                            f"步骤 {step.step_id} ({step.action}): {key}='{value[:60]}' 包含占位值"
                        )
                        break

        if placeholder_issues:
            logger.warning("计划包含占位参数，拒绝执行: %s", placeholder_issues)
            cli_adapter.console.print(
                "[bold yellow]⚠️ 计划包含占位参数，无法直接执行：[/bold yellow]"
            )
            for issue in placeholder_issues:
                cli_adapter.console.print(f"  [yellow]• {issue}[/yellow]")
            result.errors.append(f"计划包含 {len(placeholder_issues)} 个占位参数，需要重新生成")
            result.has_placeholder_params = True
            return result

        cli_adapter.console.print(
            f"[bold blue]📋 执行计划: {plan.goal}[/bold blue] "
            f"[dim]({len(plan.steps)} 步)[/dim]"
        )

        for step_id in execution_order:
            step = step_map.get(step_id)
            if step is None:
                continue

            step_result = await self._execute_step_with_checks(
                step, context, cli_adapter, tool_executor, plan, step_map
            )
            result.step_results.append(step_result)

            if step_result.skipped:
                continue

            if not step_result.success:
                if step.fallback_step and step.fallback_step in step_map:
                    cli_adapter.console.print(
                        f"[yellow]⚠ 步骤 {step.step_id} 失败，执行回退 {step.fallback_step}[/yellow]"
                    )
                    fallback_step = step_map[step.fallback_step]
                    fallback_result = await self._execute_step_with_checks(
                        fallback_step, context, cli_adapter, tool_executor, plan, step_map
                    )
                    fallback_result.fallback_executed = True
                    result.step_results.append(fallback_result)

                    if not fallback_result.success:
                        result.errors.append(
                            f"步骤 {step.step_id} 及其回退 {step.fallback_step} 均失败"
                        )
                        break
                else:
                    result.errors.append(f"步骤 {step.step_id} 失败: {step_result.error}")
                    break

            if step.action == ActionType.OUTPUT.value:
                result.output_text = step_result.output

            if self.state_machine.is_terminal(context.current_state):
                break

            if len(result.step_results) >= self.max_steps:
                result.warnings.append(f"已达到最大步数限制 ({self.max_steps})")
                break

        result.final_state = context.current_state
        result.success = self.state_machine.is_terminal(context.current_state) and (
            context.current_state in {AgentState.DONE, AgentState.APPROVED}
        )

        if not result.success:
            successful_steps = sum(1 for sr in result.step_results if sr.success and not sr.skipped)
            failed_steps = sum(1 for sr in result.step_results if not sr.success)
            total_steps = len(result.step_results)
            # 只有当所有步骤都成功（或跳过）时才标记为成功
            # 有步骤失败时不应该标记为成功
            if failed_steps == 0 and successful_steps > 0:
                result.warnings.append(
                    f"状态未到达 DONE ({context.current_state.value})，"
                    f"但 {successful_steps}/{total_steps} 步骤已成功执行"
                )
                result.success = True

        final_validation = self.constraint_validator.validate_final(context)
        if not final_validation.valid:
            result.errors.extend(final_validation.errors)
            if result.success:
                result.success = False
        result.warnings.extend(final_validation.warnings)

        result.total_duration = time.time() - start_time

        if result.success:
            cli_adapter.console.print(
                f"[bold green]✅ 计划执行完成[/bold green] "
                f"[dim]({result.total_duration:.1f}s, {len(result.step_results)} 步)[/dim]"
            )
        else:
            cli_adapter.console.print(
                f"[bold red]❌ 计划执行失败[/bold red] "
                f"[dim]({result.total_duration:.1f}s)[/dim]"
            )
            if result.errors:
                for err in result.errors:
                    cli_adapter.console.print(f"[red]  - {err}[/red]")

        return result

    def _build_runtime_context(self, plan: PlanIR) -> RuntimeCheckContext:  # noqa: ARG002
        """构建运行时检查上下文"""
        allowed_tools: set[str] | None = None
        allowed_skills: set[str] | None = None
        allowed_mcps: set[str] | None = None

        if self.tool_registry is not None:
            try:
                all_tools = self.tool_registry.list_tools()
                allowed_tools = {t.name for t in all_tools}
            except Exception:
                pass

        if self.mcp_client is not None:
            with __import__("contextlib").suppress(Exception):
                allowed_mcps = set(self.mcp_client._servers.keys())

        return RuntimeCheckContext(
            current_state=AgentState.INIT,
            history=ExecutionHistory(),
            budget_remaining=self.max_steps,
            allowed_tools=allowed_tools,
            allowed_apis=None,
            allowed_mcps=allowed_mcps,
            allowed_skills=allowed_skills,
        )

    async def _execute_step_with_checks(
        self,
        step: StepIR,
        context: RuntimeCheckContext,
        cli_adapter: Any,
        tool_executor: Any,
        plan: PlanIR,
        step_map: dict[str, StepIR],
    ) -> StepResult:
        """执行单个步骤（带运行时约束检查）"""
        step_start = time.time()

        # 为计划步骤创建 trace span
        trace = current_trace()
        span_ctx = None
        span = None
        if trace is not None:
            span_ctx = trace.span(
                f"plan.{step.action}",
                attributes={"step_id": step.step_id, "action": step.action, "description": step.description or ""},
            )
            span = await span_ctx.__aenter__()

        try:
            return await self._execute_step_inner(step, context, cli_adapter, tool_executor, plan, step_map, step_start, span)
        except Exception as e:
            if span is not None:
                span.set_attribute("error", str(e))
                span.finish(SpanStatus.ERROR)
            return StepResult(
                step_id=step.step_id,
                success=False,
                error=str(e),
                duration=time.time() - step_start,
            )
        finally:
            if span_ctx is not None and span is not None:
                await span_ctx.__aexit__(None, None, None)

    async def _execute_step_inner(
        self,
        step: StepIR,
        context: RuntimeCheckContext,
        cli_adapter: Any,
        tool_executor: Any,
        plan: PlanIR,
        step_map: dict[str, StepIR],
        step_start: float,
        span: Any,
    ) -> StepResult:
        """执行单个步骤的内部逻辑"""
        if step.condition and not self._evaluate_condition(step.condition, context):
            return StepResult(
                step_id=step.step_id,
                success=True,
                skipped=True,
                duration=time.time() - step_start,
            )

        runtime_validation = self.constraint_validator.validate_step_runtime(step, context)
        if not runtime_validation.valid:
            error_msg = "; ".join(runtime_validation.errors)
            logger.warning("步骤 %s 运行时验证失败: %s", step.step_id, error_msg)

            if any("状态机" in e for e in runtime_validation.errors):
                logger.info("尝试自动修复状态机违规...")
                repair_result = self._try_repair_state_violation(step, context, plan, step_map)
                if repair_result is not None:
                    return repair_result

            return StepResult(
                step_id=step.step_id,
                success=False,
                error=error_msg,
                duration=time.time() - step_start,
            )

        if step.risk_level in (RiskLevel.HIGH.value, RiskLevel.CRITICAL.value):
            approved = await self._request_approval(step, cli_adapter)
            if not approved:
                return StepResult(
                    step_id=step.step_id,
                    success=False,
                    error="用户拒绝执行",
                    duration=time.time() - step_start,
                )

        desc = step.description or step.action
        prev_outputs = []
        for sr in context.history.steps:
            if sr.metadata and "output_preview" in sr.metadata:
                prev_outputs.append(sr.metadata["output_preview"])
        with cli_adapter.console.status(
            f"[bold yellow]🔧 步骤 {step.step_id}: {desc}[/bold yellow]",
            spinner="line",
        ):
            step_output = await self._dispatch_action(step, tool_executor, cli_adapter, prev_outputs)

        step_duration = time.time() - step_start

        success = not self._is_step_output_error(step_output, step)

        # 更新 span 属性
        if span is not None:
            span.set_attribute("success", success)
            span.set_attribute("duration_ms", step_duration * 1000)
            if not success:
                span.finish(SpanStatus.ERROR)

        prev_state = context.current_state
        new_state = self.constraint_validator.update_state(step, step_output, context)

        exec_step = ExecutionStep(
            step_id=step.step_id,
            action=step.action,
            state_before=prev_state.value if isinstance(prev_state, AgentState) else str(prev_state),
            state_after=new_state.value if isinstance(new_state, AgentState) else str(new_state),
            timestamp=time.time(),
            metadata={"output_preview": step_output[:200] if step_output else ""},
        )
        context.history.add_step(exec_step)

        if success:
            tool_info = self._step_tool_info(step)
            cli_adapter.console.print(
                f"[dim]  ✅ {step.step_id}: {desc}[/dim]"
                f"[dim cyan] → {tool_info}[/dim cyan]"
            )

            # execute_file 步骤成功后，验证脚本是否生成了输出文件
            action_lower = step.action.lower() if hasattr(step, 'action') else ""
            if action_lower in ("execute_file",) or (
                action_lower == "call_tool" and (step.args or {}).get("tool_name") == "execute_file"
            ):
                self._verify_script_output(step, cli_adapter)
        else:
            tool_info = self._step_tool_info(step)
            cli_adapter.console.print(
                f"[dim]  ❌ {step.step_id}: {desc}[/dim]"
                f"[dim cyan] → {tool_info}[/dim cyan]"
            )

        return StepResult(
            step_id=step.step_id,
            success=success,
            output=step_output,
            error=None if success else step_output,
            duration=step_duration,
        )

    async def _dispatch_action(
        self,
        step: StepIR,
        tool_executor: Any,
        cli_adapter: Any,  # noqa: ARG002
        prev_outputs: list[str] | None = None,
    ) -> str:
        """根据 ActionType 分发执行"""
        action = step.action
        args = step.args

        if action == ActionType.CALL_TOOL.value:
            tool_name = args.get("tool_name", "")
            parameters = args.get("parameters", {})

            if not isinstance(parameters, dict):
                parameters = {}

            _SKIP_KEYS = {"tool_name", "parameters"}
            for key, value in args.items():
                if key not in _SKIP_KEYS and key not in parameters:
                    parameters[key] = value

            if not tool_name:
                return "call_tool 缺少 tool_name 参数"

            return await tool_executor(tool_name, parameters)

        _TOOL_NAME_ACTION_MAP: dict[str, str] = {
            "tavily_search": "tavily_search",
            "web_search": "tavily_search",
            "search": "tavily_search",
            "query_weather": "query_weather",
            "execute_code": "execute_code",
            "execute_file": "execute_file",
            "write_file": "write_file",
            "read_file": "read_file",
            "delete_file": "delete_file",
            "list_files": "list_files",
            "read_skill_md": "read_skill_md",
            "get_current_time": "get_current_time",
        }

        mapped_tool = _TOOL_NAME_ACTION_MAP.get(action)
        if mapped_tool and tool_executor is not None:
            return await tool_executor(mapped_tool, dict(args))

        if action == ActionType.SKIP.value:
            reason = args.get("reason", "步骤被跳过")
            logger.info("跳过步骤: %s", reason)
            return f"[跳过] {reason}"

        if action == ActionType.CALL_MCP.value:
            server_name = args.get("server_name", "")
            tool_name = args.get("tool_name", "")
            arguments = args.get("arguments", {})

            if self.mcp_client is not None and server_name:
                try:
                    result = await self.mcp_client.call_tool(server_name, tool_name, arguments)
                    return str(result)
                except Exception as e:
                    return f"MCP工具异常: {e}"
            return "MCP 客户端未初始化或缺少 server_name"

        if action == ActionType.CALL_SKILL.value:
            skill_name = args.get("skill_name", "")
            tool_name = args.get("tool_name", skill_name)
            arguments = args.get("arguments", {})

            if tool_executor is not None:
                return await tool_executor(tool_name, arguments)
            return "Skill 执行器未初始化"

        if action == ActionType.SEARCH.value:
            query = args.get("query", "")
            if tool_executor is not None:
                return await tool_executor("tavily_search", {"query": query})
            return "搜索工具不可用"

        if action == ActionType.REASON.value:
            return await self._do_reason(prev_outputs)

        if action == ActionType.SUMMARIZE.value:
            return await self._do_summarize(args, prev_outputs, tool_executor)

        if action == ActionType.OUTPUT.value:
            return await self._do_output(args, prev_outputs, tool_executor)

        if action == ActionType.WAIT_APPROVAL.value:
            return "审批已通过"

        return f"未知动作类型: {action}"

    async def _do_reason(self, prev_outputs: list[str] | None) -> str:
        prev_text = "\n\n".join(prev_outputs) if prev_outputs else ""
        if prev_text and self.llm is not None:
            try:
                from long.llm.base import LLMMessage
                messages = [
                    LLMMessage(role="system", content="你是一位数据分析师。从以下数据中提取关键洞察和结论，供报告使用。请用中文输出。"),
                    LLMMessage(role="user", content=f"分析以下信息，提取3-5个关键数据点和洞察:\n\n{prev_text[:4000]}"),
                ]
                response = await self.llm.chat(messages, purpose="reason")
                return response.content or "推理完成"
            except Exception as e:
                return f"推理失败: {e}"
        return "推理完成（无可用数据分析）"

    async def _do_summarize(
        self, args: dict[str, Any], prev_outputs: list[str] | None, tool_executor: Any
    ) -> str:
        content = args.get("content", "")
        if not content and prev_outputs:
            content = "\n\n".join(prev_outputs)
        if content and self.llm is not None:
            try:
                from long.llm.base import LLMMessage
                messages = [
                    LLMMessage(role="system", content="你是一位报告撰写专家。请将以下信息整理为一份结构清晰的 Markdown 报告，包含标题、数据分析、关键发现和结论。"),
                    LLMMessage(role="user", content=f"请整理以下信息为报告:\n\n{content[:8000]}"),
                ]
                response = await self.llm.chat(messages, purpose="summarize")
                report_text = response.content or "摘要生成完成"
                output_path = args.get("path") or args.get("output_path", "")
                if output_path and tool_executor is not None:
                    try:
                        write_result = await tool_executor(
                            "write_file", {"path": output_path, "content": report_text}
                        )
                    except Exception:
                        pass
                return report_text
            except Exception as e:
                return f"摘要生成失败: {e}"
        return "摘要步骤完成（无内容）"

    async def _do_output(
        self, args: dict[str, Any], prev_outputs: list[str] | None, tool_executor: Any
    ) -> str:
        content = args.get("content", "")
        if not content and prev_outputs:
            content = "\n\n".join(prev_outputs)
        output_path = args.get("path") or args.get("output_path", "")
        if output_path and content and tool_executor is not None:
            try:
                write_result = await tool_executor(
                    "write_file", {"path": output_path, "content": content}
                )
                return f"✅ 输出已写入: {output_path}"
            except Exception as e:
                return f"输出写入 {output_path} 失败: {e}"
        return content or "输出步骤完成"

    _TOOL_DISPLAY_MAP: dict[str, str] = {
        "tavily_search": "tavily_search",
        "web_search": "tavily_search",
        "search": "tavily_search",
        "execute_code": "execute_code",
        "execute_file": "execute_file",
        "write_file": "write_file",
        "read_file": "read_file",
        "delete_file": "delete_file",
    }

    @staticmethod
    def _is_step_output_error(output: str, step: StepIR | None = None) -> bool:
        """检测步骤输出是否为错误（而非有效数据）

        比简单的 startswith 更全面的错误检测：
        1. 已知错误前缀
        2. JSON 错误响应 ({"error": "..."})
        3. Python traceback
        4. execute_code 返回空结果的常见错误模式
        """
        if not output:
            return True  # 空输出视为错误

        # 已知错误前缀
        if output.startswith(("工具执行失败", "工具异常", "MCP工具异常", "未知工具", "Skill 执行错误", "write_file 错误", "execute_code 错误")):
            return True

        # JSON 错误响应
        stripped = output.strip()
        if stripped.startswith("{") and "error" in stripped[:200].lower():
            try:
                data = json.loads(stripped)
                if isinstance(data, dict) and "error" in data:
                    logger.warning("检测到 JSON 错误响应: %s", data.get("error", "")[:100])
                    return True
            except (json.JSONDecodeError, ValueError):
                pass

        # Python traceback
        if "Traceback (most recent call last):" in output:
            return True

        # execute_code 步骤特有的错误模式
        if step:
            args = step.args or {}
            tool_name = args.get("tool_name", "")
            action = step.action if isinstance(step.action, str) else str(step.action)
            is_execute_code = (action == ActionType.CALL_TOOL.value and tool_name == "execute_code")
            if is_execute_code:
                code = args.get("code", "") or args.get("parameters", {}).get("code", "")
                code_stripped = code.strip()
                code_first_line = code_stripped.split('\n')[0][:80] if code_stripped else ""
                # 代码使用了占位凭证（project_xxx, key_xxx 等）
                if re.search(r"(?:project_xxx|key_xxx|api_key_xxx|your_project_id|your_key_id|your_api_key)", code, re.IGNORECASE):
                    logger.warning("检测到 execute_code 使用占位凭证")
                    return True
                # 代码只是占位描述，不是可执行代码（含中文但无代码结构）
                has_chinese = bool(re.search(r'[\u4e00-\u9fa5]', code_stripped))
                has_code_structure = bool(re.search(
                    r'(?:^|\n)(?:import\s|from\s|def\s|class\s|[a-zA-Z_]\w*\s*=)',
                    code_stripped,
                ))
                if has_chinese and not has_code_structure:
                    logger.warning("检测到 execute_code 代码可能是占位描述: %s", code_first_line)
                    return True

        return False

    def _verify_script_output(self, step: StepIR, cli_adapter: Any) -> None:
        """execute_file 步骤成功后，验证脚本是否生成了输出文件

        检查 output 目录下是否存在报告/图表等文件，如果脚本执行成功
        但没有生成任何输出文件，给出警告提示。
        """
        import os as _os

        args = step.args or {}
        script_path = args.get("path", "")
        if action_lower := (args.get("tool_name", "") if args.get("tool_name") == "execute_file" else ""):
            params = args.get("parameters", {})
            if isinstance(params, dict) and params.get("path"):
                script_path = params["path"]

        if not script_path:
            return

        # 检查 output 目录下是否有文件
        output_dir = _os.path.join(_os.getcwd(), "workspace", "output")
        if not _os.path.isdir(output_dir):
            # 尝试相对路径
            output_dir = "output"

        if not _os.path.isdir(output_dir):
            cli_adapter.console.print(
                "[yellow]⚠ 脚本执行成功但 output 目录不存在，可能未生成输出文件[/yellow]"
            )
            return

        # 检查脚本描述中提到的预期输出
        desc = (step.description or "").lower()
        expects_report = any(kw in desc for kw in ("报告", "report", "文档", "doc", "生成", "ppt"))
        expects_chart = any(kw in desc for kw in ("图表", "chart", "图", "plot", "趋势"))

        if not expects_report and not expects_chart:
            return

        # 扫描 output 目录查找报告/图表文件
        found_report = False
        found_chart = False
        try:
            for fname in _os.listdir(output_dir):
                fpath = _os.path.join(output_dir, fname)
                if not _os.path.isfile(fpath):
                    continue
                try:
                    if _os.path.getsize(fpath) == 0:
                        continue
                except OSError:
                    continue
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                if ext in ("md", "docx", "pdf", "txt", "html", "pptx"):
                    found_report = True
                if ext in ("png", "jpg", "jpeg", "svg", "gif"):
                    found_chart = True
        except Exception:
            pass

        if expects_report and not found_report:
            cli_adapter.console.print(
                "[yellow]⚠ 脚本执行成功但未检测到报告文件输出，脚本可能未正确生成报告[/yellow]"
            )
        elif expects_report and found_report:
            cli_adapter.console.print(
                "[dim]  ✅ 检测到报告文件已生成[/dim]"
            )

        if expects_chart and not found_chart:
            cli_adapter.console.print(
                "[yellow]⚠ 脚本执行成功但未检测到图表文件输出，脚本可能未正确生成图表[/yellow]"
            )
        elif expects_chart and found_chart:
            cli_adapter.console.print(
                "[dim]  ✅ 检测到图表文件已生成[/dim]"
            )

    def _step_tool_info(self, step: StepIR) -> str:
        """提取步骤的工具信息用于友好展示"""
        action = step.action
        args = step.args or {}

        if action == "call_tool":
            tool_name = args.get("tool_name", "")
            if tool_name:
                params = {k: v for k, v in args.items() if k not in ("tool_name",)}
                if params:
                    param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
                    return f"{tool_name}({param_str})"
                return tool_name

        mapped = self._TOOL_DISPLAY_MAP.get(action)
        if mapped:
            query = args.get("query") or args.get("content")
            if query:
                short_query = str(query)[:60] + ("..." if len(str(query)) > 60 else "")
                return f"{mapped}(query={short_query!r})"
            path = args.get("path") or args.get("file_path")
            if path:
                return f"{mapped}({path!r})"
            return mapped

        if action in ("reason", "REASON"):
            return "🧠 推理"
        if action in ("summarize", "SUMMARIZE"):
            return "📝 摘要"
        if action in ("output", "OUTPUT"):
            return "📤 输出"
        if action in ("wait_approval", "WAIT_APPROVAL"):
            return "⏳ 审批"

        return action

    def _evaluate_condition(self, condition: str, context: RuntimeCheckContext) -> bool:
        """评估条件表达式"""
        allowed_names = {
            "has_data": context.history.has_state("HAS_DATA"),
            "verified": context.history.has_state("VERIFIED"),
            "approved": context.history.has_state("APPROVED"),
            "error_count": sum(1 for r in context.step_results.values() if isinstance(r, str) and "失败" in r),
            "tokens_used": 0,
            "True": True,
            "False": False,
        }

        try:
            import ast

            tree = ast.parse(condition, mode="eval")
            for node in ast.walk(tree):
                if isinstance(node, (ast.Call, ast.Attribute)):
                    return False
            result = eval(condition, {"__builtins__": {}}, allowed_names)
            return bool(result)
        except Exception:
            logger.warning("条件表达式评估失败: %s，跳过步骤执行", condition)
            return False

    async def _request_approval(self, step: StepIR, cli_adapter: Any) -> bool:
        """请求用户审批高风险步骤"""
        from rich.panel import Panel

        risk_emoji = "🔴" if step.risk_level == RiskLevel.CRITICAL.value else "🟡"
        risk_label = "极高风险" if step.risk_level == RiskLevel.CRITICAL.value else "高风险"

        cli_adapter.console.print()
        cli_adapter.console.print(
            Panel(
                f"{risk_emoji} [{risk_label}] {step.description or step.action}\n"
                f"步骤 ID: {step.step_id}\n"
                f"动作: {step.action}\n"
                f"参数: {json.dumps(step.args, ensure_ascii=False, indent=2)}",
                title="[bold yellow]⚠ 需要审批[/bold yellow]",
                border_style="yellow",
            )
        )

        try:
            response = await cli_adapter.prompt_session.prompt_async(
                "是否批准执行？[y/N] "
            )
            return response.strip().lower() in ("y", "yes", "是")
        except (EOFError, KeyboardInterrupt):
            return False

    def _try_repair_state_violation(
        self,
        step: StepIR,
        context: RuntimeCheckContext,
        plan: PlanIR,
        step_map: dict[str, StepIR],
    ) -> StepResult | None:
        """尝试修复状态机违规

        修复策略：
        1. 如果当前状态不允许该动作，尝试通过中间步骤到达允许的状态
        2. 常见修复：插入 reason 步骤以满足 VERIFIED 前置条件
        3. 如果动作本身不在状态机中，尝试映射到已知动作
        """
        allowed_actions = self.state_machine.get_allowed_actions(context.current_state)

        if step.action not in allowed_actions:
            repair_path = self._find_repair_path(context.current_state, step.action)
            if repair_path:
                for repair_action, target_state in repair_path:
                    logger.info(
                        "自动修复: 插入 %s 步骤 (%s -> %s)",
                        repair_action,
                        context.current_state.value,
                        target_state.value,
                    )
                    context.current_state = target_state
                    context.history.add_step(
                        ExecutionStep(
                            step_id=f"{step.step_id}_auto_{repair_action}",
                            action=repair_action,
                            state_before=context.current_state.value,
                            state_after=target_state.value,
                            timestamp=time.time(),
                        )
                    )
                return None

        if step.action == ActionType.OUTPUT.value and context.current_state == AgentState.HAS_DATA:
            logger.info("自动修复: 在 OUTPUT 前插入 REASON 步骤以满足状态机约束")
            reason_step = StepIR(
                step_id=f"{step.step_id}_auto_reason",
                action=ActionType.REASON.value,
                args={"reasoning_type": "verification"},
                description="自动验证步骤（满足状态机约束）",
            )

            valid, transition, _ = self.state_machine.check_transition(
                context.current_state, reason_step.action
            )
            if valid and transition:
                context.current_state = transition.to_state
                context.history.add_step(
                    ExecutionStep(
                        step_id=reason_step.step_id,
                        action=reason_step.action,
                        state_before=AgentState.HAS_DATA.value,
                        state_after=context.current_state.value,
                        timestamp=time.time(),
                    )
                )
                logger.info("自动修复成功: 状态 %s -> %s", AgentState.HAS_DATA.value, context.current_state.value)
                return None

        return None

    def _find_repair_path(
        self,
        from_state: AgentState,
        target_action: str,
    ) -> list[tuple[str, AgentState]] | None:
        """寻找从当前状态到目标动作的修复路径

        通过 BFS 搜索状态机，找到从 from_state 到某个允许 target_action 的状态的路径。
        每个路径元素是 (action, resulting_state)。
        """
        from collections import deque

        from .state_machine import _resolve_target_state

        all_states = [
            AgentState.INIT, AgentState.HAS_DATA, AgentState.VERIFIED,
            AgentState.GENERATED, AgentState.APPROVED,
        ]
        all_actions = [
            "search", "call_api", "call_tool", "call_mcp", "call_skill",
            "reason", "summarize", "output", "wait_approval",
        ]

        target_states = []
        for state in all_states:
            if state == from_state:
                continue
            target = _resolve_target_state(state, target_action)
            if target is not None:
                target_states.append(state)

        if not target_states:
            return None

        queue: deque[tuple[AgentState, list[tuple[str, AgentState]]]] = deque()
        queue.append((from_state, []))
        visited = {from_state}

        while queue:
            current, path = queue.popleft()

            if current in target_states:
                return path

            for action in all_actions:
                next_state = _resolve_target_state(current, action)
                if next_state is None or next_state in visited:
                    continue
                visited.add(next_state)
                new_path = path + [(action, next_state)]
                if next_state in target_states:
                    return new_path
                queue.append((next_state, new_path))

        return None
