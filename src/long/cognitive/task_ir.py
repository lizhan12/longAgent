"""Task IR — 结构化任务表示

替代裸的 user_message，让 Agent 真正"知道自己在干什么"。

核心思想：
  用户输入 → TaskIR → Planner → Executor
  而不是：用户输入 → LLM → LLM → LLM ...

TaskIR 是认知状态工程的基础，替代 prompt engineering。
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field


@dataclass
class SubtaskIR:
    """子任务中间表示"""

    id: str = ""
    description: str = ""
    status: str = "pending"
    depends_on: list[str] = field(default_factory=list)
    tool_hint: str | None = None
    result_summary: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            self.id = uuid.uuid4().hex[:8]


@dataclass
class TaskIR:
    """任务中间表示 — Agent 的结构化认知状态

    替代裸的 user_message + messages 列表。
    LLM 每轮基于 TaskIR 做决策，而不是重新理解世界。

    Attributes:
        goal: 任务目标描述
        constraints: 约束条件
        deliverables: 交付物列表
        subtasks: 子任务列表
        completed_subtasks: 已完成的子任务 ID
        key_facts: 从搜索/工具结果中提取的关键事实
        intermediate_conclusions: 中间结论
    """

    goal: str = ""
    constraints: list[str] = field(default_factory=list)
    deliverables: list[str] = field(default_factory=list)
    subtasks: list[SubtaskIR] = field(default_factory=list)
    completed_subtasks: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    intermediate_conclusions: list[str] = field(default_factory=list)

    def pending_subtasks(self) -> list[SubtaskIR]:
        return [s for s in self.subtasks if s.status == "pending"]

    def in_progress_subtasks(self) -> list[SubtaskIR]:
        return [s for s in self.subtasks if s.status == "in_progress"]

    def next_executable_subtask(self) -> SubtaskIR | None:
        for subtask in self.subtasks:
            if subtask.status != "pending":
                continue
            deps_met = all(d in self.completed_subtasks for d in subtask.depends_on)
            if deps_met:
                return subtask
        return None

    def mark_subtask_in_progress(self, subtask_id: str) -> None:
        for s in self.subtasks:
            if s.id == subtask_id:
                s.status = "in_progress"
                break

    def complete_subtask(self, subtask_id: str, result_summary: str | None = None) -> None:
        for s in self.subtasks:
            if s.id == subtask_id:
                s.status = "completed"
                if result_summary:
                    s.result_summary = result_summary
                if subtask_id not in self.completed_subtasks:
                    self.completed_subtasks.append(subtask_id)
                break

    def fail_subtask(self, subtask_id: str) -> None:
        for s in self.subtasks:
            if s.id == subtask_id:
                s.status = "failed"
                break

    def add_key_fact(self, fact: str) -> None:
        if fact and fact not in self.key_facts:
            self.key_facts.append(fact)

    def add_conclusion(self, conclusion: str) -> None:
        if conclusion and conclusion not in self.intermediate_conclusions:
            self.intermediate_conclusions.append(conclusion)

    def progress_ratio(self) -> float:
        if not self.subtasks:
            return 0.0
        return len(self.completed_subtasks) / len(self.subtasks)

    def is_all_complete(self) -> bool:
        return bool(self.subtasks) and all(
            s.status in ("completed", "failed") for s in self.subtasks
        )

    def to_prompt_text(self) -> str:
        completed = [s for s in self.subtasks if s.status == "completed"]
        pending = [s for s in self.subtasks if s.status in ("pending", "in_progress")]
        failed = [s for s in self.subtasks if s.status == "failed"]

        parts = [f"当前任务：{self.goal}"]

        if self.constraints:
            parts.append(f"约束：{', '.join(self.constraints)}")

        if self.deliverables:
            parts.append(f"交付物：{', '.join(self.deliverables)}")

        if completed:
            descs = [f"  ✅ {s.description}" + (f" → {s.result_summary}" if s.result_summary else "") for s in completed]
            parts.append("已完成子任务：\n" + "\n".join(descs))

        if pending:
            descs = [f"  ⏳ {s.description}" + (f" (建议: {s.tool_hint})" if s.tool_hint else "") for s in pending]
            parts.append("待完成子任务：\n" + "\n".join(descs))

        if failed:
            descs = [f"  ❌ {s.description}" for s in failed]
            parts.append("失败子任务：\n" + "\n".join(descs))

        if self.key_facts:
            parts.append("关键事实：\n" + "\n".join(f"  • {f}" for f in self.key_facts[-10:]))

        if self.intermediate_conclusions:
            parts.append("中间结论：\n" + "\n".join(f"  • {c}" for c in self.intermediate_conclusions[-5:]))

        return "\n\n".join(parts)


_NEEDS_CHART_PATTERNS = re.compile(r"图|chart|plot|折线|可视化|图形|绘图")
_NEEDS_REPORT_PATTERNS = re.compile(r"报告|report|文档|导出|PPT|ppt|演示")
_WEATHER_PATTERNS = re.compile(r"天气|气温|温度|下雨|晴天|阴天|多云|风力|预报|weather|forecast|几度|多少度")
_SEARCH_PATTERNS = re.compile(r"搜索|查询|查找|了解|调研|search|query|股价|汇率|新闻|最新|当前|现在|多少|排名|价格|数据|统计")
_CODE_PATTERNS = re.compile(r"代码|执行|运行|计算|脚本|code|run|execute|script")
_FILE_PATTERNS = re.compile(r"文件|保存|写入|导出|file|save|write|export")


def parse_task_ir_from_message(user_message: str) -> TaskIR:
    """从用户消息解析生成 TaskIR

    基于规则引擎，零延迟，不调用 LLM。
    生成初始 TaskIR 后，THINK 节点可以进一步用 LLM 细化。
    """
    task = TaskIR(goal=user_message)

    needs_chart = bool(_NEEDS_CHART_PATTERNS.search(user_message))
    needs_report = bool(_NEEDS_REPORT_PATTERNS.search(user_message))
    needs_search = bool(_SEARCH_PATTERNS.search(user_message))
    needs_code = bool(_CODE_PATTERNS.search(user_message))
    needs_file = bool(_FILE_PATTERNS.search(user_message))
    is_weather = bool(_WEATHER_PATTERNS.search(user_message))

    if needs_chart:
        task.deliverables.append("图表/可视化")
    if needs_report:
        task.deliverables.append("报告/文档")

    subtasks: list[SubtaskIR] = []

    if is_weather:
        subtasks.append(SubtaskIR(
            description="查询天气信息",
            tool_hint="execute_file",
        ))
    elif needs_search:
        subtasks.append(SubtaskIR(
            description="搜索获取所需信息",
            tool_hint="tavily_search",
        ))

    if needs_code:
        subtasks.append(SubtaskIR(
            description="编写和执行代码",
            tool_hint="execute_code",
            depends_on=[s.id for s in subtasks if s.tool_hint == "tavily_search"],
        ))

    if needs_chart:
        subtasks.append(SubtaskIR(
            description="生成图表/可视化",
            tool_hint="execute_code",
            depends_on=[s.id for s in subtasks if s.tool_hint in ("tavily_search", "execute_code")],
        ))

    if needs_file or needs_report:
        subtasks.append(SubtaskIR(
            description="生成并保存报告/文件",
            tool_hint="write_file",
            depends_on=[s.id for s in subtasks],
        ))

    if not subtasks:
        if any(kw in user_message for kw in ("什么", "如何", "怎么", "为什么", "which", "how", "why", "what")):
            subtasks.append(SubtaskIR(
                description="搜索回答用户问题",
                tool_hint="tavily_search",
            ))
        else:
            subtasks.append(SubtaskIR(
                description="完成用户请求",
            ))

    task.subtasks = subtasks
    return task


_TASK_IR_GENERATION_PROMPT = """你是一个任务分析专家。请分析用户的请求，生成结构化的任务表示。

用户请求：{user_message}

请输出以下 JSON 格式（不要输出其他内容）：
{{
  "goal": "任务目标的一句话描述",
  "constraints": ["约束1", "约束2"],
  "deliverables": ["交付物1", "交付物2"],
  "subtasks": [
    {{
      "description": "子任务描述",
      "tool_hint": "tavily_search / execute_code / write_file / null",
      "depends_on": []
    }}
  ]
}}

规则：
1. goal 要简洁明确，不要复述用户原话
2. constraints 只列真正的约束（如"必须用中文"、"不超过500字"）
3. deliverables 是用户期望看到的具体产出
4. subtasks 按执行顺序排列，depends_on 填前面子任务的索引号（0, 1, 2...）
5. tool_hint 可选值：tavily_search（搜索）、execute_code（代码执行）、write_file（文件写入）、null（不确定）
6. 子任务粒度要适中，不要拆太细（2-6个即可）"""


def get_task_ir_generation_prompt(user_message: str) -> str:
    return _TASK_IR_GENERATION_PROMPT.format(user_message=user_message)
