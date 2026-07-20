"""沙箱管理器

管理沙箱实例的生命周期，提供统一的执行入口。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from .base import (
    ExecutionResult,
    ExecutionSpec,
    ExecutionStatus,
    IsolationLevel,
    ResourceLimits,
    Sandbox,
)
from .code_scanner import CodeScanner
from .monitor import ResourceMonitor
from .process_sandbox import ProcessSandbox

logger = logging.getLogger(__name__)


class SandboxManager:
    """沙箱管理器

    管理沙箱的创建、执行和清理。集成代码预扫描和资源监控。

    Attributes:
        workspace_dir: 工作区目录
        default_isolation: 默认隔离级别
        default_limits: 默认资源限制
    """

    def __init__(
        self,
        workspace_dir: str | Path | None = None,
        default_isolation: IsolationLevel = IsolationLevel.PROCESS,
        default_limits: ResourceLimits | None = None,
        enable_scanner: bool = True,
    ) -> None:
        self._workspace_dir = Path(workspace_dir) if workspace_dir else None
        self._default_isolation = default_isolation
        self._default_limits = default_limits or ResourceLimits()
        self._enable_scanner = enable_scanner
        self._scanner = CodeScanner() if enable_scanner else None
        self._instances: dict[str, Sandbox] = {}
        self._monitors: dict[str, ResourceMonitor] = {}

    def _get_sandbox(self, level: IsolationLevel) -> Sandbox:
        """获取对应隔离级别的沙箱实例"""
        if level == IsolationLevel.PROCESS:
            return ProcessSandbox(workspace_dir=self._workspace_dir)
        elif level == IsolationLevel.NONE:
            # 无隔离，仍使用进程沙箱但不设限制
            return ProcessSandbox(workspace_dir=self._workspace_dir)
        else:
            # Container/MicroVM 暂未实现，降级为 PROCESS
            logger.warning(
                "Isolation level %s not yet implemented, falling back to PROCESS",
                level.value,
            )
            return ProcessSandbox(workspace_dir=self._workspace_dir)

    async def execute(self, spec: ExecutionSpec) -> ExecutionResult:
        """执行入口

        流程: 代码预扫描 → 创建沙箱 → 执行 → 监控 → 清理

        Args:
            spec: 执行规格

        Returns:
            执行结果
        """
        # 1. 代码预扫描
        if self._scanner:
            scan_result = self._scanner.scan(spec.code)
            if not scan_result.safe:
                return ExecutionResult(
                    status=ExecutionStatus.SECURITY_VIOLATION,
                    error=f"Code scan detected dangerous patterns: "
                    + "; ".join(t["description"] for t in scan_result.threats),
                )

        # 2. 合并默认资源限制
        if spec.resource_limits == ResourceLimits():
            spec = spec.model_copy(update={"resource_limits": self._default_limits})

        # 3. 获取沙箱实例并执行
        sandbox = self._get_sandbox(self._default_isolation)

        monitor = ResourceMonitor(limits=spec.resource_limits)

        try:
            sandbox_id = await sandbox.create(spec)

            pid_info = sandbox._sandboxes.get(sandbox_id)
            pid = pid_info.get("pid") if pid_info else None

            monitor_task: asyncio.Task[MonitorResult] | None = None
            if pid:
                monitor_task = asyncio.create_task(
                    monitor.start(pid)
                )

            result = await sandbox.run(sandbox_id)

            monitor.stop()
            if monitor_task:
                try:
                    if monitor_task.done():
                        mon_res = monitor_task.result()
                    else:
                        mon_res = await asyncio.wait_for(monitor_task, timeout=1.0)
                    if isinstance(mon_res, MonitorResult):
                        result.peak_cpu = mon_res.peak_cpu
                        result.peak_memory = mon_res.peak_memory
                except (asyncio.TimeoutError, Exception):
                    pass

            return result

        finally:
            try:
                await sandbox.cleanup(sandbox_id)
            except Exception:
                pass

    async def execute_with_session(
        self,
        spec: ExecutionSpec,
        session_id: str,
        sandbox_lifecycle: Any = None,
    ) -> tuple[ExecutionResult, str]:
        """会话级执行 — 沙箱在会话结束后才销毁

        Harness Engineering: 沙箱状态在多轮对话间保持，不必每次从零开始。

        Returns:
            (执行结果, sandbox_id) — sandbox_id 用于后续复用或销毁
        """
        if sandbox_lifecycle is None:
            return await self.execute(spec), ""

        sandbox_id = getattr(sandbox_lifecycle, "sandbox_id", "") if sandbox_lifecycle else ""

        scanned = False
        if self._scanner:
            scan_result = self._scanner.scan(spec.code)
            if not scan_result.safe:
                return (
                    ExecutionResult(
                        status=ExecutionStatus.SECURITY_VIOLATION,
                        error=f"Code scan detected dangerous patterns: "
                        + "; ".join(t["description"] for t in scan_result.threats),
                    ),
                    sandbox_id,
                )
            scanned = True

        if spec.resource_limits == ResourceLimits():
            spec = spec.model_copy(update={"resource_limits": self._default_limits})

        sandbox = self._get_sandbox(self._default_isolation)

        if not sandbox_id or sandbox_id not in sandbox._sandboxes:
            sandbox_id = await sandbox.create(spec)
            if sandbox_lifecycle is not None:
                sandbox_lifecycle.temp_dir = sandbox._sandboxes[sandbox_id].get("temp_dir", "")

        sandbox._sandboxes[sandbox_id]["spec"] = spec

        monitor = ResourceMonitor(limits=spec.resource_limits)
        try:
            result = await sandbox.run(sandbox_id)

            monitor.stop()
            try:
                monitor_result = await asyncio.wait_for(
                    asyncio.create_task(monitor.start(
                        sandbox._sandboxes.get(sandbox_id, {}).get("pid", 0)
                    )),
                    timeout=2.0,
                )
                result.peak_cpu = monitor_result.peak_cpu
                result.peak_memory = monitor_result.peak_memory
            except (asyncio.TimeoutError, Exception):
                pass

            return result, sandbox_id
        except Exception:
            return (
                ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    error=f"执行异常",
                ),
                sandbox_id,
            )

    async def cleanup_session_sandbox(self, sandbox_id: str) -> None:
        """清理会话级沙箱"""
        if not sandbox_id:
            return
        sandbox = self._get_sandbox(self._default_isolation)
        try:
            await sandbox.cleanup(sandbox_id)
        except Exception:
            pass

    async def kill_all(self) -> int:
        """紧急终止所有沙箱

        Returns:
            终止的沙箱数量
        """
        count = 0
        for sandbox_id, sandbox in list(self._instances.items()):
            try:
                if await sandbox.kill(sandbox_id):
                    count += 1
            except Exception:
                pass

        # 同时停止所有监控
        for monitor in self._monitors.values():
            monitor.stop()

        self._instances.clear()
        self._monitors.clear()
        return count
