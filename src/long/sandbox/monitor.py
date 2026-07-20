"""资源监控

使用 psutil 监控沙箱进程的资源使用。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .base import ResourceLimits

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """资源快照"""

    timestamp: float
    cpu_percent: float
    memory_bytes: int
    num_threads: int
    num_fds: int


@dataclass
class MonitorResult:
    """监控结果"""

    peak_cpu: float = 0.0
    peak_memory: int = 0
    avg_cpu: float = 0.0
    avg_memory: int = 0
    snapshots: list[ResourceSnapshot] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    killed: bool = False
    kill_reason: str | None = None


class ResourceMonitor:
    """资源监控器

    定期轮询进程资源使用，超出阈值告警或终止。

    Attributes:
        limits: 资源限制
        warn_threshold: 告警阈值（占限制的比例，0-1）
        kill_threshold: 终止阈值（占限制的比例，0-1）
        poll_interval: 轮询间隔（秒）
    """

    def __init__(
        self,
        limits: ResourceLimits | None = None,
        warn_threshold: float = 0.8,
        kill_threshold: float = 1.0,
        poll_interval: float = 0.5,
    ) -> None:
        self.limits = limits or ResourceLimits()
        self.warn_threshold = warn_threshold
        self.kill_threshold = kill_threshold
        self.poll_interval = poll_interval
        self._monitoring = False
        self._snapshots: list[ResourceSnapshot] = []
        self._warnings: list[str] = []

    async def start(self, pid: int) -> MonitorResult:
        """开始监控进程

        Args:
            pid: 进程 ID

        Returns:
            监控结果
        """
        self._monitoring = True
        self._snapshots = []
        self._warnings = []

        try:
            import psutil

            try:
                proc = psutil.Process(pid)
            except psutil.NoSuchProcess:
                return MonitorResult()

            while self._monitoring:
                try:
                    if not proc.is_running():
                        break

                    cpu = proc.cpu_percent(interval=None)
                    mem_info = proc.memory_info()
                    memory_bytes = mem_info.rss

                    try:
                        num_threads = proc.num_threads()
                    except psutil.AccessDenied:
                        num_threads = 0

                    try:
                        num_fds = proc.num_fds() if hasattr(proc, "num_fds") else 0
                    except (psutil.AccessDenied, AttributeError):
                        num_fds = 0

                    snapshot = ResourceSnapshot(
                        timestamp=time.time(),
                        cpu_percent=cpu,
                        memory_bytes=memory_bytes,
                        num_threads=num_threads,
                        num_fds=num_fds,
                    )
                    self._snapshots.append(snapshot)

                    # 检查阈值
                    self._check_thresholds(snapshot)

                    await asyncio.sleep(self.poll_interval)

                except psutil.NoSuchProcess:
                    break
                except psutil.AccessDenied:
                    break

        except ImportError:
            logger.debug("psutil not installed, resource monitoring disabled")

        return self._build_result()

    def stop(self) -> None:
        """停止监控"""
        self._monitoring = False

    def _check_thresholds(self, snapshot: ResourceSnapshot) -> None:
        """检查资源阈值"""
        # 内存检查
        memory_ratio = snapshot.memory_bytes / max(self.limits.memory, 1)
        if memory_ratio >= self.kill_threshold:
            self._warnings.append(
                f"Memory usage {snapshot.memory_bytes} bytes exceeds kill threshold "
                f"({self.kill_threshold * 100:.0f}% of {self.limits.memory})"
            )
        elif memory_ratio >= self.warn_threshold:
            self._warnings.append(
                f"Memory usage {snapshot.memory_bytes} bytes exceeds warn threshold "
                f"({self.warn_threshold * 100:.0f}% of {self.limits.memory})"
            )

        # 进程数检查
        if snapshot.num_threads > self.limits.processes:
            self._warnings.append(
                f"Thread count {snapshot.num_threads} exceeds limit {self.limits.processes}"
            )

        # 文件描述符检查
        if snapshot.num_fds > self.limits.file_descriptors:
            self._warnings.append(
                f"File descriptors {snapshot.num_fds} exceeds limit {self.limits.file_descriptors}"
            )

    def _build_result(self) -> MonitorResult:
        """构建监控结果"""
        if not self._snapshots:
            return MonitorResult()

        peak_cpu = max(s.cpu_percent for s in self._snapshots)
        peak_memory = max(s.memory_bytes for s in self._snapshots)
        avg_cpu = sum(s.cpu_percent for s in self._snapshots) / len(self._snapshots)
        avg_memory = sum(s.memory_bytes for s in self._snapshots) / len(self._snapshots)

        return MonitorResult(
            peak_cpu=peak_cpu,
            peak_memory=peak_memory,
            avg_cpu=avg_cpu,
            avg_memory=int(avg_memory),
            snapshots=self._snapshots,
            warnings=self._warnings,
        )
