"""会话级沙箱 — 跨轮复用沙箱实例，保持状态可恢复

Harness Engineering 理念：
沙箱不是"用完即毁"的，多轮对话中同一沙箱实例可恢复。
避免每次执行都重新初始化环境（重复 import、重复安装依赖）。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SandboxLifecycle(str, Enum):
    """沙箱生命周期模式"""

    EPHEMERAL = "ephemeral"
    """每次执行后销毁（当前默认行为）"""
    SESSION = "session"
    """会话级：同一次对话中复用沙箱"""
    PERSISTENT = "persistent"
    """持久化：跨会话复用，需手动清理"""


@dataclass
class SessionSandbox:
    """会话级沙箱实例

    维护一个可跨轮复用的沙箱环境。
    LLM 可以在同一个 Python 进程中交互式执行代码，不必每次"写文件 → 执行"两步走。
    """

    sandbox_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    lifecycle: SandboxLifecycle = SandboxLifecycle.SESSION
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    last_used_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())

    process: asyncio.subprocess.Process | None = None
    """REPL 模式下的持久进程（用于交互式 Python）"""
    temp_dir: str = ""
    """沙箱临时目录"""
    installed_packages: set[str] = field(default_factory=set)
    """已安装的 Python 包"""
    round_count: int = 0
    """已执行的轮次数"""

    def touch(self) -> None:
        self.last_used_at = asyncio.get_event_loop().time()
        self.round_count += 1

    async def cleanup(self) -> None:
        if self.process is not None:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
            self.process = None

        if self.temp_dir:
            import shutil
            from pathlib import Path

            tmp = Path(self.temp_dir)
            if tmp.exists():
                try:
                    shutil.rmtree(str(tmp))
                except OSError:
                    logger.warning("清理沙箱目录失败: %s", self.temp_dir)


@dataclass
class SandboxSessionConfig:
    """沙箱会话配置"""

    lifecycle: SandboxLifecycle = SandboxLifecycle.SESSION
    max_rounds_per_session: int = 20
    """一个会话沙箱最多复用轮次"""
    idle_timeout: float = 300.0
    """空闲超时（秒），超过后自动销毁"""
    max_installed_packages: int = 50
    """最多记录多少个已安装包"""


class SandboxSessionManager:
    """沙箱会话管理器

    管理沙箱实例的生命周期：创建、复用、销毁。
    与 SandboxManager 配合使用，在 SandboxManager.execute() 外层添加会话级复用逻辑。
    """

    def __init__(self, config: SandboxSessionConfig | None = None) -> None:
        self._config = config or SandboxSessionConfig()
        self._active: dict[str, SessionSandbox] = {}
        self._cleanup_task: asyncio.Task | None = None

    async def get_or_create(self, session_id: str) -> SessionSandbox:
        """获取或创建会话级沙箱"""
        if session_id in self._active:
            sb = self._active[session_id]
            if self._should_expire(sb):
                await self._destroy(session_id)
            else:
                sb.touch()
                return sb

        sb = SessionSandbox(
            sandbox_id=f"sandbox_{session_id}_{uuid.uuid4().hex[:4]}",
            lifecycle=self._config.lifecycle,
        )
        self._active[session_id] = sb
        return sb

    def _should_expire(self, sb: SessionSandbox) -> bool:
        loop = asyncio.get_event_loop()
        idle_time = loop.time() - sb.last_used_at
        return (
            sb.round_count >= self._config.max_rounds_per_session
            or idle_time > self._config.idle_timeout
        )

    async def _destroy(self, session_id: str) -> None:
        sb = self._active.pop(session_id, None)
        if sb is not None:
            await sb.cleanup()
            logger.info("沙箱会话销毁: %s (执行 %d 轮)", sb.sandbox_id, sb.round_count)

    async def destroy_all(self) -> None:
        for session_id in list(self._active.keys()):
            await self._destroy(session_id)

    async def cleanup_expired(self) -> None:
        for session_id in list(self._active.keys()):
            sb = self._active.get(session_id)
            if sb is not None and self._should_expire(sb):
                await self._destroy(session_id)