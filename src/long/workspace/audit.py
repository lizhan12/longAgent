"""工作区审计钩子"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AuditConfig:
    """审计配置"""

    def __init__(self, allowed_paths: list[str] | None = None) -> None:
        self.allowed_paths = allowed_paths or []


class WorkspaceAuditHook:
    """工作区审计钩子 — 监控文件操作"""

    def __init__(self, config: AuditConfig) -> None:
        self._config = config
        self._installed = False

    def install(self) -> None:
        """安装审计钩子"""
        self._installed = True
        logger.info(
            "工作区审计已启用，允许路径: %s",
            ", ".join(self._config.allowed_paths),
        )

    def uninstall(self) -> None:
        """卸载审计钩子"""
        self._installed = False
        logger.info("工作区审计已禁用")