"""Skill Manager

管理 Skill 的生命周期: 加载、启用、禁用、卸载、热重载。
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .skill_loader import SkillLoader
from .unified_tool_registry import ToolDefinition, ToolSource, UnifiedToolRegistry

logger = logging.getLogger(__name__)


class SkillState(str, Enum):
    """Skill 状态"""

    LOADED = "loaded"
    REGISTERED = "registered"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"


class SkillManifest(BaseModel):
    """Skill 清单

    Attributes:
        name: Skill 名称
        version: 版本
        description: 描述
        permissions: 权限声明
        tools: 工具列表
        dependencies: 依赖的其他 Skill
        entry_point: 入口点模块路径
    """

    name: str
    version: str = "0.1.0"
    description: str = ""
    permissions: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    entry_point: str = ""


class SkillRecord(BaseModel):
    """Skill 记录"""

    manifest: SkillManifest
    state: SkillState = SkillState.LOADED
    path: str = ""
    error: str | None = None
    module: Any = None

    model_config = {"arbitrary_types_allowed": True}


class SkillManager:
    """Skill 管理器

    管理 Skill 的完整生命周期，与 UnifiedToolRegistry 集成。

    Attributes:
        _registry: 工具注册表
        _loader: Skill 加载器
        _skills: Skill 记录
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry: UnifiedToolRegistry | None = None,
        skill_dir: str | Path | None = None,
        allowed_permissions: set[str] | None = None,
    ) -> None:
        self._config = config or {}
        self._registry = registry or UnifiedToolRegistry()
        self._loader = SkillLoader()
        self._skills: dict[str, SkillRecord] = {}
        self._skills_dir = Path(skill_dir) if skill_dir else None
        self._allowed_permissions = allowed_permissions or {
            "filesystem.read",
            "filesystem.write",
            "network.http",
            "compute.cpu",
            "compute.memory",
        }

    @property
    def registry(self) -> UnifiedToolRegistry:
        return self._registry

    def load_skill(self, skill_path: str | Path) -> SkillRecord | None:
        """加载 Skill

        支持 SKILL.md (skill-creator 格式) 和 __init__.py 两种格式。

        Args:
            skill_path: Skill 目录或文件路径

        Returns:
            Skill 记录，失败返回 None
        """
        skill_path = Path(skill_path)

        try:
            # 扫描代码安全性
            scan_result = self._loader.scan_code(skill_path)
            if not scan_result.safe:
                logger.error("Skill at %s failed security scan: %s", skill_path, scan_result.threats)
                return SkillRecord(
                    manifest=SkillManifest(name="unknown"),
                    state=SkillState.ERROR,
                    path=str(skill_path),
                    error=f"Security scan failed: {scan_result.threats}",
                )

            # 加载模块
            result = self._loader.load(skill_path)
            if result is None:
                return SkillRecord(
                    manifest=SkillManifest(name="unknown"),
                    state=SkillState.ERROR,
                    path=str(skill_path),
                    error="Failed to load skill module",
                )

            module, manifest_data = result

            # 解析清单
            manifest = SkillManifest(
                name=manifest_data.get("name", skill_path.stem),
                version=manifest_data.get("version", "0.1.0"),
                description=manifest_data.get("description", ""),
                permissions=manifest_data.get("permissions", []),
                tools=manifest_data.get("tools", []),
                dependencies=manifest_data.get("dependencies", []),
                entry_point=manifest_data.get("entry_point", ""),
            )

            # 元数据纯声明式 skill（仅有 SKILL.md，无 __init__.py）
            if module is None:
                record = SkillRecord(
                    manifest=manifest,
                    state=SkillState.REGISTERED,
                    path=str(skill_path),
                )
                self._register_tools(record)
                self._skills[manifest.name] = record
                logger.info("Registered metadata-only skill: %s", manifest.name)
                return record

            # 检查权限
            required_permissions = manifest.permissions
            if not self._check_permissions(required_permissions):
                return SkillRecord(
                    manifest=manifest,
                    state=SkillState.ERROR,
                    path=str(skill_path),
                    error=f"Missing required permissions: {required_permissions}",
                )

            # 注册工具
            record = SkillRecord(
                manifest=manifest,
                state=SkillState.LOADED,
                path=str(skill_path),
                module=module,
            )

            self._register_tools(record)
            record.state = SkillState.REGISTERED
            self._skills[manifest.name] = record

            logger.info("Loaded skill: %s v%s", manifest.name, manifest.version)
            return record

        except Exception as e:
            logger.error("Error loading skill from %s: %s", skill_path, e)
            return SkillRecord(
                manifest=SkillManifest(name="unknown"),
                state=SkillState.ERROR,
                path=str(skill_path),
                error=str(e),
            )

    def enable_skill(self, skill_name: str) -> bool:
        """启用 Skill

        Args:
            skill_name: Skill 名称

        Returns:
            是否成功启用
        """
        record = self._skills.get(skill_name)
        if record is None:
            return False

        if record.state == SkillState.ERROR:
            return False

        record.state = SkillState.ENABLED
        return True

    def disable_skill(self, skill_name: str) -> bool:
        """禁用 Skill

        Args:
            skill_name: Skill 名称

        Returns:
            是否成功禁用
        """
        record = self._skills.get(skill_name)
        if record is None:
            return False

        record.state = SkillState.DISABLED
        return True

    def unload_skill(self, skill_name: str) -> bool:
        """卸载 Skill

        Args:
            skill_name: Skill 名称

        Returns:
            是否成功卸载
        """
        record = self._skills.get(skill_name)
        if record is None:
            return False

        # 反注册工具
        for tool_name in record.manifest.tools:
            self._registry.unregister(tool_name)

        self._skills.pop(skill_name)
        logger.info("Unloaded skill: %s", skill_name)
        return True

    def reload_skill(self, skill_name: str) -> SkillRecord | None:
        """热重载 Skill

        Args:
            skill_name: Skill 名称

        Returns:
            新的 Skill 记录
        """
        record = self._skills.get(skill_name)
        if record is None:
            return None

        path = record.path
        was_enabled = record.state == SkillState.ENABLED

        self.unload_skill(skill_name)
        new_record = self.load_skill(path)

        if new_record and was_enabled:
            self.enable_skill(skill_name)

        return new_record

    def list_skills(self) -> list[SkillRecord]:
        """列出所有 Skill"""
        return list(self._skills.values())

    def get_skill(self, skill_name: str) -> SkillRecord | None:
        """获取 Skill 记录"""
        return self._skills.get(skill_name)

    def auto_discover(self) -> list[SkillRecord]:
        """自动发现并加载 Skill

        从 skills_dir 目录中发现所有 Skill 模块。
        支持 SKILL.md (skill-creator 格式) 和 __init__.py 两种格式。

        Returns:
            加载的 Skill 记录列表
        """
        if self._skills_dir is None or not self._skills_dir.exists():
            return []

        records = []
        for path in self._skills_dir.iterdir():
            if path.is_dir() and self._loader.has_valid_skill_format(path):
                record = self.load_skill(path)
                if record:
                    records.append(record)
            elif path.suffix == ".py" and path.name != "__init__.py":
                record = self.load_skill(path)
                if record:
                    records.append(record)

        return records

    def _register_tools(self, record: SkillRecord) -> None:
        """注册 Skill 工具到注册表（不覆盖已有的 LOCAL 工具）"""
        for tool_name in record.manifest.tools:
            existing = self._registry.get_tool(tool_name)
            if existing is not None:
                logger.debug(
                    "Tool '%s' already registered (source=%s), skipping skill registration for '%s'",
                    tool_name, existing.source.value, record.manifest.name,
                )
                continue
            tool_def = ToolDefinition(
                name=tool_name,
                description=f"Tool from skill '{record.manifest.name}'",
                source=ToolSource.SKILL,
                source_name=record.manifest.name,
            )
            self._registry.register_skill(record.manifest.name, tool_def)

    def _check_permissions(self, permissions: list[str]) -> bool:
        """检查权限

        验证 Skill 声明的权限是否在允许列表中。
        高危权限（filesystem.delete, network.raw, process.create）需要额外审批。
        """
        if not permissions:
            return True

        dangerous_permissions = {
            "filesystem.delete",
            "network.raw",
            "process.create",
            "system.admin",
            "security.bypass",
        }

        for perm in permissions:
            if perm in dangerous_permissions:
                logger.warning("Skill requests dangerous permission: %s", perm)
                return False
            if perm not in self._allowed_permissions and not perm.startswith(self._allowed_permissions):
                base_perm = perm.rsplit(".", 1)[0] if "." in perm else perm
                if base_perm not in {p.rsplit(".", 1)[0] for p in self._allowed_permissions}:
                    logger.warning("Skill requests unallowed permission: %s", perm)
                    return False

        return True
