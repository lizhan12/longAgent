"""Task Planner — 基于 TaskIR 的结构化规划

替代 PLAN 节点的 continuation policy（继续/停止二选一），
进化为基于 TaskIR 的 DAG 规划器。

核心思想：
  旧 PLAN: should_continue / is_complete（二选一）
  新 PLAN: PlanResult（含 next_subtask + strategy_hint + progress_report）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .task_ir import SubtaskIR, TaskIR

logger = logging.getLogger(__name__)


@dataclass
class PlanResult:
    """规划结果 — 替代 should_continue / is_complete 二元决策"""

    should_continue: bool = False
    is_complete: bool = False
    next_subtask: SubtaskIR | None = None
    strategy_hint: str | None = None
    progress_report: str = ""
    needs_plan_repair: bool = False


class TaskPlanner:
    """基于 TaskIR 的结构化规划器

    两阶段决策：
      Phase A: Task Progress Assessment（纯规则，零延迟）
      Phase B: Next Step Planning（基于 TaskIR）
    """

    def plan(self, context: Any) -> PlanResult:
        """执行规划决策

        Args:
            context: CognitiveContext 实例
        """
        progress = self._assess_progress(context)
        next_step = self._plan_next_step(context, progress)
        return next_step

    def _assess_progress(self, context: Any) -> dict[str, Any]:
        """Phase A: 评估任务进度（纯规则）"""
        task_ir: TaskIR | None = getattr(context, "task_ir", None)

        progress: dict[str, Any] = {
            "has_task_ir": task_ir is not None,
            "all_subtasks_complete": False,
            "has_pending": False,
            "has_in_progress": False,
            "pending_count": 0,
            "completed_count": 0,
            "failed_count": 0,
            "progress_ratio": 0.0,
        }

        if task_ir is None:
            return progress

        progress["all_subtasks_complete"] = task_ir.is_all_complete()
        progress["has_pending"] = len(task_ir.pending_subtasks()) > 0
        progress["has_in_progress"] = len(task_ir.in_progress_subtasks()) > 0
        progress["pending_count"] = len(task_ir.pending_subtasks())
        progress["completed_count"] = len(task_ir.completed_subtasks)
        progress["failed_count"] = sum(1 for s in task_ir.subtasks if s.status == "failed")
        progress["progress_ratio"] = task_ir.progress_ratio()

        return progress

    def _plan_next_step(self, context: Any, progress: dict[str, Any]) -> PlanResult:
        """Phase B: 基于进度决定下一步"""
        task_ir: TaskIR | None = getattr(context, "task_ir", None)
        round_count: int = getattr(context, "round_count", 0)
        max_rounds: int = getattr(context, "max_rounds", 8)
        search_count: int = getattr(context, "search_count", 0)
        max_search_count: int = getattr(context, "max_search_count", 3)
        errors: list[str] = getattr(context, "errors", [])
        tool_history: list[dict[str, Any]] = getattr(context, "tool_history", [])

        search_exhausted = search_count >= max_search_count
        has_errors = len(errors) > 0
        has_tool_results = len(tool_history) > 0

        needs_chart = any(
            kw in getattr(context, "user_message", "")
            for kw in ("图", "chart", "plot", "折线", "可视化")
        )
        needs_report = any(
            kw in getattr(context, "user_message", "")
            for kw in ("报告", "report", "文档", "生成", "保存", "导出")
        )
        needs_code_output = needs_chart or needs_report

        has_code_tools = any(
            t.get("name") in ("write_file", "execute_code", "execute_file")
            for t in tool_history
            if "error" not in t
        )

        all_intercepted = (
            len(tool_history) > 0
            and all(
                t.get("result", "").startswith("[") and ("限制" in t.get("result", "") or "约束" in t.get("result", ""))
                for t in tool_history
            )
        )
        intercepted_count = sum(
            1 for t in tool_history
            if t.get("result", "").startswith("[") and ("限制" in t.get("result", "") or "约束" in t.get("result", ""))
        )

        if round_count >= max_rounds:
            return PlanResult(
                is_complete=True,
                should_continue=False,
                progress_report=f"轮次耗尽 ({round_count}/{max_rounds})",
            )

        if progress["has_task_ir"] and progress["all_subtasks_complete"]:
            return PlanResult(
                is_complete=True,
                should_continue=False,
                progress_report="所有子任务已完成",
                next_subtask=None,
            )

        if all_intercepted and search_exhausted and not needs_code_output:
            return PlanResult(
                is_complete=True,
                should_continue=False,
                progress_report="所有工具被拦截且搜索耗尽",
            )

        if all_intercepted and intercepted_count >= 2 and not needs_code_output:
            return PlanResult(
                is_complete=True,
                should_continue=False,
                progress_report="工具多次被拦截",
            )

        if has_errors and getattr(context, "retry_count", 0) < getattr(context, "max_retries", 3):
            return PlanResult(
                should_continue=True,
                is_complete=False,
                strategy_hint="修复错误并重试",
                progress_report=f"有错误，重试中 ({getattr(context, 'retry_count', 0)}/{getattr(context, 'max_retries', 3)})",
            )

        if intercepted_count > 0 and search_exhausted and not needs_code_output:
            return PlanResult(
                is_complete=True,
                should_continue=False,
                progress_report="工具被拦截且搜索耗尽",
            )

        if search_exhausted and has_code_tools:
            next_sub = task_ir.next_executable_subtask() if task_ir else None
            return PlanResult(
                should_continue=True,
                is_complete=False,
                next_subtask=next_sub,
                strategy_hint="搜索耗尽，继续执行代码/文件工具",
                progress_report="搜索耗尽，有代码工具待执行",
            )

        if search_exhausted and has_tool_results and needs_code_output and not has_code_tools:
            next_sub = task_ir.next_executable_subtask() if task_ir else None
            return PlanResult(
                should_continue=True,
                is_complete=False,
                next_subtask=next_sub,
                strategy_hint="搜索完成，需要生成图表/报告",
                progress_report="搜索完成，需要代码输出",
            )

        if search_exhausted and has_tool_results and not needs_code_output:
            next_sub = task_ir.next_executable_subtask() if task_ir else None
            return PlanResult(
                should_continue=True,
                is_complete=False,
                next_subtask=next_sub,
                strategy_hint="基于已有结果生成最终回答",
                progress_report="搜索完成，生成回答",
            )

        if not has_tool_results:
            next_sub = task_ir.next_executable_subtask() if task_ir else None
            return PlanResult(
                should_continue=True,
                is_complete=False,
                next_subtask=next_sub,
                progress_report="尚未执行任何工具",
            )

        next_sub = task_ir.next_executable_subtask() if task_ir else None
        hint = None
        if next_sub and next_sub.tool_hint:
            hint = f"建议使用 {next_sub.tool_hint}"

        return PlanResult(
            should_continue=True,
            is_complete=False,
            next_subtask=next_sub,
            strategy_hint=hint,
            progress_report=f"进度: {progress['progress_ratio']:.0%}",
        )
