"""Skill Loader

安全加载 Skill 模块，受限 globals 防止恶意操作。
支持 SKILL.md (skill-creator 格式) 和 __init__.py 两种技能格式。
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from ..sandbox.code_scanner import CodeScanner, ScanResult

logger = logging.getLogger(__name__)

# 安全的内置函数子集
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "Exception": Exception,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


class SkillLoader:
    """Skill 加载器

    安全加载 Skill 模块，使用受限 globals。
    支持 SKILL.md (skill-creator 格式) 和 __init__.py 两种技能格式。

    Attributes:
        _scanner: 代码扫描器
    """

    def __init__(self, scanner: CodeScanner | None = None) -> None:
        self._scanner = scanner or CodeScanner()

    @staticmethod
    def parse_skill_md(skill_dir: Path) -> dict[str, Any] | None:
        """解析 SKILL.md 文件获取元数据

        按 skill-creator 规范解析 YAML frontmatter。

        Args:
            skill_dir: Skill 目录路径

        Returns:
            元数据字典，或 None
        """
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            return None

        try:
            content = skill_md.read_text(encoding="utf-8")
        except Exception:
            return None

        if not content.startswith("---"):
            return None

        parts = content.split("---", 2)
        if len(parts) < 3:
            return None

        try:
            frontmatter = yaml.safe_load(parts[1])
        except yaml.YAMLError:
            return None

        if not isinstance(frontmatter, dict):
            return None

        metadata = frontmatter.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        raw_tools = frontmatter.get("tools") or metadata.get("tools") or []
        tools = list(raw_tools) if isinstance(raw_tools, list) else []

        return {
            "name": str(frontmatter.get("name", skill_dir.name)),
            "version": str(metadata.get("version", "0.1.0")),
            "description": str(frontmatter.get("description", "")),
            "permissions": list(metadata.get("permissions", [])),
            "tools": tools,
            "dependencies": list(metadata.get("dependencies", [])),
            "entry_point": str(metadata.get("entry_point", "")),
        }

    @staticmethod
    def has_valid_skill_format(skill_dir: Path) -> bool:
        """检查目录是否包含有效的 skill 格式

        支持 SKILL.md 或 __init__.py 任一格式。

        Args:
            skill_dir: Skill 目录路径

        Returns:
            是否包含有效格式
        """
        if not skill_dir.is_dir():
            return False
        return (skill_dir / "SKILL.md").exists() or (skill_dir / "__init__.py").exists()

    def scan_code(self, skill_path: Path) -> ScanResult:
        """扫描 Skill 代码

        对于仅有 SKILL.md 的声明式 skill，跳过代码扫描。

        Args:
            skill_path: Skill 路径

        Returns:
            扫描结果
        """
        # 仅有 SKILL.md 的声明式 skill，无需代码扫描
        if skill_path.is_dir() and not (skill_path / "__init__.py").exists():
            if (skill_path / "SKILL.md").exists():
                return ScanResult(safe=True, threats=[])
            return ScanResult(safe=False, threats=[{
                "name": "no_valid_format",
                "description": "No __init__.py or SKILL.md found",
                "level": "dangerous",
                "match_count": 1,
            }])

        code = self._read_code(skill_path)
        if code is None:
            return ScanResult(safe=False, threats=[{
                "name": "read_error",
                "description": "Cannot read skill code",
                "level": "dangerous",
                "match_count": 1,
            }])
        return self._scanner.scan(code)

    def load(self, skill_path: Path) -> tuple[Any, dict[str, Any]] | None:
        """加载 Skill 模块

        支持 SKILL.md 元数据和 __init__.py 模块两种格式。
        优先使用 SKILL.md 中的元数据，__init__.py 中的属性作为兜底。

        Args:
            skill_path: Skill 目录或文件路径

        Returns:
            (module, manifest_data) 或 None
        """
        skill_path = Path(skill_path)

        # 尝试从 SKILL.md 获取元数据
        skill_md_data: dict[str, Any] = {}
        if skill_path.is_dir():
            skill_md_data = self.parse_skill_md(skill_path) or {}

        # 确定模块路径
        if skill_path.is_dir():
            init_file = skill_path / "__init__.py"
            if not init_file.exists():
                # 只有 SKILL.md 没有 __init__.py 的情况（纯声明式 skill）
                if skill_md_data:
                    logger.info("Skill %s has SKILL.md but no __init__.py, loading as metadata-only", skill_path.name)
                    return None, skill_md_data
                logger.debug("No __init__.py or SKILL.md in %s", skill_path)
                return None
            module_path = init_file
            module_name = f"long_skill_{skill_path.name}"
        elif skill_path.is_file() and skill_path.suffix == ".py":
            module_path = skill_path
            module_name = f"long_skill_{skill_path.stem}"
        else:
            logger.error("Invalid skill path: %s", skill_path)
            return None

        try:
            module = self._safe_import(module_name, module_path)
            if module is None:
                return None

            # 提取清单数据：优先 SKILL.md，兜底模块属性
            manifest_data = {
                "name": skill_md_data.get("name") or getattr(module, "SKILL_NAME", module_name),
                "version": skill_md_data.get("version") or getattr(module, "SKILL_VERSION", "0.1.0"),
                "description": skill_md_data.get("description") or getattr(module, "SKILL_DESCRIPTION", ""),
                "permissions": skill_md_data.get("permissions") or getattr(module, "SKILL_PERMISSIONS", []),
                "tools": skill_md_data.get("tools") or getattr(module, "SKILL_TOOLS", []),
                "dependencies": skill_md_data.get("dependencies") or getattr(module, "SKILL_DEPENDENCIES", []),
                "entry_point": skill_md_data.get("entry_point") or getattr(module, "SKILL_ENTRY_POINT", ""),
            }

            return module, manifest_data

        except Exception as e:
            logger.error("Error loading skill from %s: %s", skill_path, e)
            return None

    def _safe_import(self, module_name: str, module_path: Path) -> Any | None:
        """安全导入模块

        使用受限 globals 防止恶意操作。

        Args:
            module_name: 模块名称
            module_path: 模块文件路径

        Returns:
            模块对象或 None
        """
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(module_path))
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)

            # 使用受限 globals
            safe_globals = {
                "__builtins__": self._get_safe_builtins(),
                "__name__": module_name,
                "__file__": str(module_path),
            }
            module.__dict__.update(safe_globals)

            # 执行模块代码
            spec.loader.exec_module(module)

            return module

        except Exception as e:
            logger.error("Error in safe_import for %s: %s", module_name, e)
            return None

    @staticmethod
    def _get_safe_builtins() -> dict[str, Any]:
        """获取安全的内置函数子集"""
        return dict(SAFE_BUILTINS)

    def _read_code(self, skill_path: Path) -> str | None:
        """读取 Skill 代码"""
        try:
            # 显式 UTF-8：默认按 locale 解码，Windows(GBK) 下含中文注释的 skill
            # 代码会 UnicodeDecodeError，被当成 "Cannot read skill code" 判定为不安全，
            # 整个 skill 直接加载失败。
            if skill_path.is_dir():
                init_file = skill_path / "__init__.py"
                if init_file.exists():
                    return init_file.read_text(encoding="utf-8", errors="replace")
            elif skill_path.is_file():
                return skill_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
        return None
