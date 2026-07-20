"""任务上下文隔离 — 防止一处失火烧穿全局

Harness Engineering 原则：隔离防污染（Sub-agent Isolation）
从"所有任务共享同一对话历史"升级到"每个子任务独立上下文"：
- 子任务失败不污染主任务对话
- 不同性质的任务隔开处理
- 错误边界（Error Boundary）防止级联失败

设计约束：
- 零外部依赖
- 与现有 FallbackLoop 和 PlanExecutor 兼容
- 隔离是可选的，不破坏现有流程
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class IsolationLevel(str, Enum):
    """隔离级别"""
    NONE = "none"           # 不隔离（默认行为）
    SOFT = "soft"           # 软隔离：共享 system prompt，独立工具历史
    HARD = "hard"           # 硬隔离：完全独立的上下文


class TaskPhase(str, Enum):
    """任务阶段"""
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IsolatedContext:
    """隔离的任务上下文"""
    task_id: str = ""
    parent_task_id: str = ""
    isolation_level: IsolationLevel = IsolationLevel.SOFT
    phase: TaskPhase = TaskPhase.PLANNING
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        msg: dict[str, Any] = {"role": role, "content": content}
        msg.update(kwargs)
        self.messages.append(msg)

    def add_tool_result(self, tool_name: str, result: str, is_error: bool = False) -> None:
        self.tool_results.append({
            "tool_name": tool_name,
            "result": result,
            "is_error": is_error,
            "timestamp": time.time(),
        })
        if is_error:
            self.errors.append(f"{tool_name}: {result[:200]}")

    def mark_completed(self) -> None:
        self.phase = TaskPhase.COMPLETED
        self.completed_at = time.time()

    def mark_failed(self, reason: str = "") -> None:
        self.phase = TaskPhase.FAILED
        self.completed_at = time.time()
        if reason:
            self.errors.append(reason)

    @property
    def duration(self) -> float:
        end = self.completed_at or time.time()
        return end - self.created_at

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def error_summary(self) -> str:
        if not self.errors:
            return ""
        return "; ".join(self.errors[-3:])  # 最近3条错误


class ErrorBoundary:
    """错误边界 — 捕获子任务错误，防止级联传播

    用法：
        boundary = ErrorBoundary()
        with boundary.guard("task_123"):
            # 执行子任务
            ...
        if boundary.has_failed("task_123"):
            logger.warning("子任务失败: %s", boundary.get_error("task_123"))
    """

    def __init__(self) -> None:
        self._failures: dict[str, str] = {}
        self._contexts: dict[str, IsolatedContext] = {}

    def guard(self, task_id: str) -> "ErrorBoundaryGuard":
        """创建一个错误边界守卫"""
        return ErrorBoundaryGuard(task_id, self)

    def record_failure(self, task_id: str, error: str) -> None:
        self._failures[task_id] = error

    def record_context(self, task_id: str, context: IsolatedContext) -> None:
        self._contexts[task_id] = context

    def has_failed(self, task_id: str) -> bool:
        return task_id in self._failures

    def get_error(self, task_id: str) -> str:
        return self._failures.get(task_id, "")

    def get_context(self, task_id: str) -> IsolatedContext | None:
        return self._contexts.get(task_id)

    def get_all_failures(self) -> dict[str, str]:
        return dict(self._failures)

    def clear(self) -> None:
        self._failures.clear()
        self._contexts.clear()


class ErrorBoundaryGuard:
    """错误边界守卫 — context manager"""

    def __init__(self, task_id: str, boundary: ErrorBoundary) -> None:
        self._task_id = task_id
        self._boundary = boundary

    def __enter__(self) -> "ErrorBoundaryGuard":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is not None:
            # 关键异常不抑制，允许传播
            if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
                return False
            error_msg = f"{exc_type.__name__}: {exc_val}"
            self._boundary.record_failure(self._task_id, error_msg)
            logger.warning(
                "错误边界捕获 [%s]: %s",
                self._task_id, error_msg[:200],
            )
            return True  # 抑制异常传播
        return False


class TaskContextIsolator:
    """任务上下文隔离器

    管理多个隔离的任务上下文，支持软隔离和硬隔离。

    用法：
        isolator = TaskContextIsolator()
        ctx = isolator.create_context("task_1", isolation_level="soft")
        ctx.add_message("user", "执行数据分析")
        ctx.add_tool_result("read_file", "数据内容...", is_error=False)
        ctx.mark_completed()

        # 获取合并后的上下文（用于主任务）
        merged = isolator.get_merged_messages("task_1", system_prompt="...")
    """

    def __init__(self) -> None:
        self._contexts: dict[str, IsolatedContext] = {}
        self._error_boundary = ErrorBoundary()

    @property
    def error_boundary(self) -> ErrorBoundary:
        return self._error_boundary

    def create_context(
        self,
        task_id: str,
        parent_task_id: str = "",
        isolation_level: IsolationLevel | str = IsolationLevel.SOFT,
    ) -> IsolatedContext:
        """创建隔离的任务上下文"""
        if isinstance(isolation_level, str):
            isolation_level = IsolationLevel(isolation_level)

        ctx = IsolatedContext(
            task_id=task_id,
            parent_task_id=parent_task_id,
            isolation_level=isolation_level,
        )
        self._contexts[task_id] = ctx
        self._error_boundary.record_context(task_id, ctx)
        logger.info("创建隔离上下文: %s (级别=%s)", task_id, isolation_level.value)
        return ctx

    def get_context(self, task_id: str) -> IsolatedContext | None:
        return self._contexts.get(task_id)

    def get_merged_messages(
        self,
        task_id: str,
        system_prompt: str = "",
    ) -> list[dict[str, Any]]:
        """获取合并后的消息列表（用于 LLM 调用）

        软隔离：包含 system_prompt + 子任务的工具结果摘要
        硬隔离：仅包含 system_prompt + 子任务的完整消息
        无隔离：返回子任务的完整消息
        """
        ctx = self._contexts.get(task_id)
        if ctx is None:
            return []

        messages: list[dict[str, Any]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if ctx.isolation_level == IsolationLevel.NONE:
            messages.extend(ctx.messages)

        elif ctx.isolation_level == IsolationLevel.SOFT:
            # 软隔离：只包含工具结果摘要
            if ctx.tool_results:
                summary_parts = []
                for tr in ctx.tool_results:
                    if tr["is_error"]:
                        summary_parts.append(f"❌ {tr['tool_name']}: (错误) {tr['result'][:100]}")
                    else:
                        summary_parts.append(f"✅ {tr['tool_name']}: {tr['result'][:100]}")
                messages.append({
                    "role": "user",
                    "content": f"[子任务 {task_id} 结果摘要]\n" + "\n".join(summary_parts),
                })

        elif ctx.isolation_level == IsolationLevel.HARD:
            # 硬隔离：完整消息
            messages.extend(ctx.messages)

        return messages

    def get_failed_tasks(self) -> list[str]:
        """获取所有失败的子任务"""
        return [
            task_id for task_id, ctx in self._contexts.items()
            if ctx.phase == TaskPhase.FAILED
        ]

    def get_stats(self) -> dict[str, Any]:
        total = len(self._contexts)
        completed = sum(1 for c in self._contexts.values() if c.phase == TaskPhase.COMPLETED)
        failed = sum(1 for c in self._contexts.values() if c.phase == TaskPhase.FAILED)
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "active": total - completed - failed,
            "error_boundary_failures": len(self._error_boundary.get_all_failures()),
        }

    def clear(self) -> None:
        self._contexts.clear()
        self._error_boundary.clear()
