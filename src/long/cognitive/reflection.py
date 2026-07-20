"""Strategy Reflection — 策略反思系统

替代 Reflector 的 retry policy（成功/失败→重试），
进化为三层反思：
  Layer 1: Execution Validation（现有，保留）
  Layer 2: Strategy Critique（新增，规则优先）
  Layer 3: Plan Repair（新增，基于 TaskIR）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .task_ir import TaskIR

logger = logging.getLogger(__name__)


@dataclass
class StrategyIssue:
    """策略问题"""

    type: str
    description: str
    suggestion: str = ""
    severity: str = "medium"


@dataclass
class StrategyCritiqueResult:
    """策略批判结果"""

    issues: list[StrategyIssue] = field(default_factory=list)
    needs_plan_repair: bool = False
    overall_assessment: str = "ok"


class StrategyCritique:
    """策略批判引擎

    规则优先，覆盖 80% 常见策略问题：
    - 重复搜索
    - 工具选择不合理
    - 进度停滞
    - 搜索结果浪费
    """

    def critique(self, context: Any) -> StrategyCritiqueResult:
        issues: list[StrategyIssue] = []
        task_ir: TaskIR | None = getattr(context, "task_ir", None)
        tool_history: list[dict[str, Any]] = getattr(context, "tool_history", [])
        round_count: int = getattr(context, "round_count", 0)

        issues.extend(self._check_redundant_search(tool_history))
        issues.extend(self._check_tool_mismatch(tool_history, task_ir))
        issues.extend(self._check_stalled_progress(round_count, task_ir))
        issues.extend(self._check_search_waste(tool_history, context))

        needs_repair = any(i.severity in ("high", "critical") for i in issues)

        assessment = "ok"
        if issues:
            critical = [i for i in issues if i.severity == "critical"]
            high = [i for i in issues if i.severity == "high"]
            if critical:
                assessment = "critical"
            elif high:
                assessment = "warning"
            else:
                assessment = "minor"

        return StrategyCritiqueResult(
            issues=issues,
            needs_plan_repair=needs_repair,
            overall_assessment=assessment,
        )

    def _check_redundant_search(self, tool_history: list[dict[str, Any]]) -> list[StrategyIssue]:
        issues: list[StrategyIssue] = []
        search_queries = [
            t.get("arguments", {}).get("query", "")
            for t in tool_history
            if t.get("name") == "tavily_search" and "error" not in t
        ]

        if len(search_queries) >= 2:
            seen = set()
            for q in search_queries:
                normalized = q.strip().lower()
                if normalized in seen:
                    issues.append(StrategyIssue(
                        type="redundant_search",
                        description=f"重复搜索相同关键词: '{q[:30]}'",
                        suggestion="换用不同关键词或使用已有结果",
                        severity="medium",
                    ))
                seen.add(normalized)

        return issues

    def _check_tool_mismatch(self, tool_history: list[dict[str, Any]], task_ir: TaskIR | None) -> list[StrategyIssue]:
        issues: list[StrategyIssue] = []
        if not task_ir or not tool_history:
            return issues

        current = task_ir.next_executable_subtask()
        if not current or not current.tool_hint:
            return issues

        last_tool = tool_history[-1].get("name", "")
        hint = current.tool_hint

        mismatch_map = {
            "tavily_search": {"execute_code", "write_file", "execute_file"},
            "execute_code": {"tavily_search"},
            "write_file": {"tavily_search"},
        }

        if hint in mismatch_map and last_tool in mismatch_map[hint]:
            issues.append(StrategyIssue(
                type="tool_mismatch",
                description=f"子任务建议用 {hint}，实际用了 {last_tool}",
                suggestion=f"考虑使用 {hint} 类工具",
                severity="high",
            ))

        return issues

    def _check_stalled_progress(self, round_count: int, task_ir: TaskIR | None) -> list[StrategyIssue]:
        issues: list[StrategyIssue] = []
        if not task_ir:
            return issues

        if round_count >= 3 and len(task_ir.completed_subtasks) == 0:
            issues.append(StrategyIssue(
                type="stalled_progress",
                description=f"3轮后无子任务完成 (round={round_count})",
                suggestion="重新评估任务分解或换策略",
                severity="high",
            ))

        if round_count >= 5 and task_ir.progress_ratio() < 0.3:
            issues.append(StrategyIssue(
                type="slow_progress",
                description=f"5轮后进度仅 {task_ir.progress_ratio():.0%}",
                suggestion="考虑简化任务或调整执行顺序",
                severity="medium",
            ))

        return issues

    def _check_search_waste(self, tool_history: list[dict[str, Any]], _context: Any) -> list[StrategyIssue]:
        issues: list[StrategyIssue] = []
        search_results = [
            t for t in tool_history
            if t.get("name") == "tavily_search" and "error" not in t
        ]

        if not search_results:
            return issues

        short_results = [
            t for t in search_results
            if len(t.get("result", "")) < 100
        ]

        if len(short_results) == len(search_results) and len(search_results) >= 2:
            issues.append(StrategyIssue(
                type="search_waste",
                description="多次搜索结果都很短，可能关键词不合适",
                suggestion="尝试更具体的关键词或换搜索策略",
                severity="medium",
            ))

        return issues


class PlanRepair:
    """计划修复器

    当 Strategy Critique 检测到策略问题时，
    修复 TaskIR（添加/删除/修改 subtask）。
    """

    def repair(self, task_ir: TaskIR, critique_result: StrategyCritiqueResult) -> list[str]:
        """修复 TaskIR，返回修复操作列表"""
        repairs = []

        for issue in critique_result.issues:
            if issue.type == "redundant_search":
                repairs.append(self._handle_redundant_search(task_ir, issue))
            elif issue.type == "stalled_progress":
                repairs.append(self._handle_stalled_progress(task_ir, issue))
            elif issue.type == "tool_mismatch":
                repairs.append(self._handle_tool_mismatch(task_ir, issue))
            elif issue.type == "search_waste":
                repairs.append(self._handle_search_waste(task_ir, issue))

        return [r for r in repairs if r]

    def _handle_redundant_search(self, task_ir: TaskIR, _issue: StrategyIssue) -> str:
        for s in task_ir.subtasks:
            if s.status == "pending" and s.tool_hint == "tavily_search":
                s.tool_hint = None
                return f"已移除子任务 '{s.description}' 的搜索建议，改为使用已有结果"
        return ""

    def _handle_stalled_progress(self, task_ir: TaskIR, _issue: StrategyIssue) -> str:
        pending = task_ir.pending_subtasks()
        if not pending:
            return ""

        first_pending = pending[0]
        first_pending.depends_on = []
        return f"已简化子任务 '{first_pending.description}' 的依赖，允许直接执行"

    def _handle_tool_mismatch(self, task_ir: TaskIR, _issue: StrategyIssue) -> str:
        current = task_ir.next_executable_subtask()
        if current and current.tool_hint:
            return f"已标记子任务 '{current.description}' 优先使用 {current.tool_hint}"
        return ""

    def _handle_search_waste(self, task_ir: TaskIR, _issue: StrategyIssue) -> str:
        for s in task_ir.subtasks:
            if s.status == "pending" and s.tool_hint == "tavily_search":
                s.description = f"{s.description}（请使用更具体的关键词）"
                return "已更新子任务描述，提示使用更具体的关键词"
        return ""
