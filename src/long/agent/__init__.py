"""多智能体协同模块

提供 SubAgentSpec（子 Agent 声明规格）、SubAgentRegistry（注册表自动发现）、
TaskOrchestrator（异步任务编排器）、DelegateTask（委派任务）等子 Agent 基础设施。

新增 P-W-E 三栖拓扑组件：
- PlannerAgent：旗舰模型，不碰工具，专司规划
- WorkerAgent：快速模型 + 有限工具集，独立 Think-Act-Observe 循环
- SubAgentRunner：连接 Spec + Worker + Orchestrator + Critic
- CriticAgent：规则 + 快速 LLM 双重审计
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """异步任务状态"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class SubAgentSpec:
    """子 Agent 声明规格

    在 workspace/subagents/ 目录下用 YAML/JSON 声明。
    示例：
    ```yaml
    name: data_fetcher
    description: 获取数据
    tools: [tavily_search, read_file]
    prompt: "你是一个数据获取助手..."
    timeout: 120
    ```
    """

    name: str
    description: str = ""
    tools: list[str] = field(default_factory=list)
    prompt: str = ""
    model: str = ""
    timeout: float = 120.0
    max_retries: int = 1
    isolation_scope: str = "session"
    """隔离级别: session | user | agent | global"""

    @classmethod
    def from_file(cls, path: str | Path) -> "SubAgentSpec":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"子 Agent 规格文件不存在: {path}")

        content = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)

        return cls(**data)


@dataclass
class DelegateTask:
    """委派给子 Agent 的任务"""

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    sub_agent_name: str = ""
    instruction: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    """传递给子 Agent 的上下文数据"""
    timeout: float = 120.0
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    started_at: float | None = None
    completed_at: float | None = None
    result: Any = None
    error: str = ""
    parent_session_id: str = ""


class TaskOrchestrator:
    """任务编排器

    管理异步子任务的完整生命周期：提交 → 执行 → 结果回收 → 超时取消。
    支持替换为跨进程实现（对接外部任务队列如 Celery、MQ）。
    """

    def __init__(self, max_concurrent: int = 5) -> None:
        self._tasks: dict[str, DelegateTask] = {}
        self._running: dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._result_callbacks: dict[str, list[Callable[[DelegateTask], Coroutine[Any, Any, None]]]] = {}

    def submit(
        self,
        sub_agent_name: str,
        instruction: str,
        *,
        context: dict[str, Any] | None = None,
        timeout: float = 120.0,
        parent_session_id: str = "",
        on_complete: Callable[[DelegateTask], Coroutine[Any, Any, None]] | None = None,
    ) -> DelegateTask:
        """提交异步任务"""
        task = DelegateTask(
            sub_agent_name=sub_agent_name,
            instruction=instruction,
            context=context or {},
            timeout=timeout,
            parent_session_id=parent_session_id,
        )
        self._tasks[task.task_id] = task
        if on_complete is not None:
            if task.task_id not in self._result_callbacks:
                self._result_callbacks[task.task_id] = []
            self._result_callbacks[task.task_id].append(on_complete)
        logger.info("任务已提交: %s → %s", task.task_id, sub_agent_name)
        return task

    async def execute(
        self,
        task: DelegateTask,
        executor: Callable[[DelegateTask], Coroutine[Any, Any, Any]],
    ) -> None:
        """执行单个任务（内部使用）"""
        async with self._semaphore:
            task.status = TaskStatus.RUNNING
            task.started_at = asyncio.get_event_loop().time()

            try:
                result = await asyncio.wait_for(
                    executor(task),
                    timeout=task.timeout,
                )
                task.status = TaskStatus.COMPLETED
                task.result = result
                logger.info("任务完成: %s", task.task_id)
            except asyncio.TimeoutError:
                task.status = TaskStatus.TIMEOUT
                task.error = f"任务超时 ({task.timeout}s)"
                logger.warning("任务超时: %s", task.task_id)
            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                task.error = "任务被取消"
                logger.info("任务被取消: %s", task.task_id)
            except Exception as exc:
                task.status = TaskStatus.FAILED
                task.error = f"{type(exc).__name__}: {exc}"
                logger.error("任务失败: %s → %s", task.task_id, task.error)

            task.completed_at = asyncio.get_event_loop().time()

            callbacks = self._result_callbacks.pop(task.task_id, [])
            for cb in callbacks:
                try:
                    await cb(task)
                except Exception:
                    logger.warning("任务回调执行失败: %s", task.task_id, exc_info=True)

    def cancel(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self._running:
            self._running[task_id].cancel()
            return True
        task = self._tasks.get(task_id)
        if task is not None and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            task.error = "任务被取消"
            return True
        return False

    def get_task(self, task_id: str) -> DelegateTask | None:
        return self._tasks.get(task_id)

    def list_tasks(self, status: TaskStatus | None = None) -> list[DelegateTask]:
        tasks = list(self._tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        return tasks

    def get_pending_count(self) -> int:
        return len([t for t in self._tasks.values() if t.status == TaskStatus.PENDING])

    def get_running_count(self) -> int:
        return len([t for t in self._tasks.values() if t.status == TaskStatus.RUNNING])


@dataclass
class SubAgentRegistry:
    """子 Agent 注册表

    管理所有声明的子 Agent 规格。
    从 workspace/subagents/ 目录自动加载。
    """

    specs: dict[str, SubAgentSpec] = field(default_factory=dict)

    def register(self, spec: SubAgentSpec) -> None:
        self.specs[spec.name] = spec

    def load_from_dir(self, directory: str | Path) -> int:
        """从目录加载子 Agent 声明"""
        directory = Path(directory)
        if not directory.is_dir():
            return 0

        count = 0
        for file_path in directory.glob("*"):
            if file_path.suffix in (".yaml", ".yml", ".json"):
                try:
                    spec = SubAgentSpec.from_file(file_path)
                    self.register(spec)
                    count += 1
                    logger.info("加载子 Agent: %s ← %s", spec.name, file_path.name)
                except Exception:
                    logger.warning(
                        "加载子 Agent 规格失败: %s", file_path, exc_info=True,
                    )
        return count

    def get(self, name: str) -> SubAgentSpec | None:
        return self.specs.get(name)

    def list_names(self) -> list[str]:
        return sorted(self.specs.keys())


from .critic import CriticAgent, CriticVerdict, CritiqueIssue, CriticReport
from .escalation import (
    EscalationAction,
    EscalationController,
    EscalationDecision,
    FailureSignal,
    FailureType,
)
from .planner import PlannerAgent
from .runner import SubAgentRunner
from .worker import WorkerAgent, WorkerResult