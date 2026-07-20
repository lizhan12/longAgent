"""能力边界清单 — 统一权限声明与分发

Harness Engineering 原则：能力边界先于自由（Tool Permissions）
从"边界分散在三处"升级到"一处声明，全局执行"：
- PERMISSIONS.md 作为唯一事实来源
- 启动时加载并分发到各检查点
- 修改一处，全局生效

设计约束：
- 人类可读的 Markdown 格式（PERMISSIONS.md）
- 机器可解析的 JSON 格式（permissions.json）
- 与 security.yaml、SafetyBoundary、TypeChecker 三处联动
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolPermission:
    """单个工具的权限定义"""
    name: str = ""
    allowed: bool = True
    forbidden_in: list[str] = field(default_factory=list)  # 在哪些模式下禁止
    requires_confirmation: bool = False  # 是否需要 HITL 确认
    risk_level: str = "low"  # low / medium / high / critical
    description: str = ""


@dataclass
class PermissionManifest:
    """能力边界清单"""
    tools: list[ToolPermission] = field(default_factory=list)
    global_forbidden: list[str] = field(default_factory=list)  # 全局禁止的操作
    global_rules: list[str] = field(default_factory=list)  # 不变式规则

    def get_tool(self, name: str) -> ToolPermission | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def is_allowed(self, tool_name: str, mode: str = "development") -> bool:
        tool = self.get_tool(tool_name)
        if tool is None:
            return False  # 未声明的工具默认拒绝（fail-closed）
        if not tool.allowed:
            return False
        if mode in tool.forbidden_in:
            return False
        return True

    def needs_confirmation(self, tool_name: str) -> bool:
        tool = self.get_tool(tool_name)
        if tool is None:
            return False
        return tool.requires_confirmation

    def get_risk_level(self, tool_name: str) -> str:
        tool = self.get_tool(tool_name)
        if tool is None:
            return "low"
        return tool.risk_level

    def get_forbidden_tools(self, mode: str = "development") -> list[str]:
        result = list(self.global_forbidden)
        for t in self.tools:
            if not t.allowed or mode in t.forbidden_in:
                result.append(t.name)
        return result

    def get_high_risk_tools(self) -> list[str]:
        return [t.name for t in self.tools if t.risk_level in ("high", "critical")]


# 默认权限清单 — 与现有 security.yaml 和 SafetyBoundary 对齐
DEFAULT_TOOLS = [
    ToolPermission(name="read_file", allowed=True, risk_level="low", description="读取文件"),
    ToolPermission(name="list_files", allowed=True, risk_level="low", description="列出目录"),
    ToolPermission(name="write_file", allowed=True, risk_level="medium", description="写入文件"),
    ToolPermission(name="delete_file", allowed=True, forbidden_in=["service"], requires_confirmation=True, risk_level="critical", description="删除文件"),
    ToolPermission(name="execute_code", allowed=True, forbidden_in=["service"], requires_confirmation=True, risk_level="high", description="执行代码"),
    ToolPermission(name="execute_file", allowed=True, forbidden_in=["service"], requires_confirmation=True, risk_level="high", description="执行文件"),
    ToolPermission(name="tavily_search", allowed=True, risk_level="low", description="网络搜索"),
    ToolPermission(name="get_current_time", allowed=True, risk_level="low", description="获取当前时间"),
    ToolPermission(name="read_skill_md", allowed=True, risk_level="low", description="读取技能文档"),
]

DEFAULT_GLOBAL_FORBIDDEN = [
    "访问 /etc/ 目录",
    "访问 /root/ 目录",
    "访问 /proc/ 目录",
    "执行 fork 炸弹",
    "建立 reverse shell",
    "修改系统环境变量",
]

DEFAULT_GLOBAL_RULES = [
    "delete_file 始终需要确认",
    "execute_code 始终需要 AST 扫描",
    "path 始终不允许 .. 或 /etc/",
    "token 预算始终有上限",
    "总轮次始终有上限",
]


class PermissionManifestLoader:
    """权限清单加载器

    用法：
        loader = PermissionManifestLoader(workspace_dir)
        manifest = loader.load()
        if manifest.is_allowed("delete_file", mode="service"):
            ...
    """

    def __init__(self, workspace_dir: str | Path) -> None:
        self._dir = Path(workspace_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _json_path(self) -> Path:
        return self._dir / "permissions.json"

    def _markdown_path(self) -> Path:
        return self._dir.parent / "PERMISSIONS.md"

    def load(self) -> PermissionManifest:
        """加载权限清单（优先 JSON，回退到默认）"""
        json_path = self._json_path()
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                return self._parse_json(data)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("权限清单 JSON 解析失败，使用默认: %s", e)

        return self._default_manifest()

    def save(self, manifest: PermissionManifest) -> None:
        """保存权限清单（JSON + Markdown 双写）"""
        self._save_json(manifest)
        self._save_markdown(manifest)

    def _parse_json(self, data: dict[str, Any]) -> PermissionManifest:
        tools = []
        for t in data.get("tools", []):
            tools.append(ToolPermission(
                name=t.get("name", ""),
                allowed=t.get("allowed", True),
                forbidden_in=t.get("forbidden_in", []),
                requires_confirmation=t.get("requires_confirmation", False),
                risk_level=t.get("risk_level", "low"),
                description=t.get("description", ""),
            ))
        return PermissionManifest(
            tools=tools,
            global_forbidden=data.get("global_forbidden", []),
            global_rules=data.get("global_rules", []),
        )

    def _save_json(self, manifest: PermissionManifest) -> None:
        data = {
            "tools": [
                {
                    "name": t.name,
                    "allowed": t.allowed,
                    "forbidden_in": t.forbidden_in,
                    "requires_confirmation": t.requires_confirmation,
                    "risk_level": t.risk_level,
                    "description": t.description,
                }
                for t in manifest.tools
            ],
            "global_forbidden": manifest.global_forbidden,
            "global_rules": manifest.global_rules,
        }
        self._json_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save_markdown(self, manifest: PermissionManifest) -> None:
        """同步到 PERMISSIONS.md（人类可读版本）"""
        lines = [
            "# 能力边界清单 (Permission Manifest)",
            "",
            "> Harness 原则：能力边界先于自由。先规定「不能做什么」，再谈「能做什么」。",
            "> 此文件是唯一事实来源，修改此处全局生效。",
            "",
            "## 全局禁止",
            "",
        ]
        for rule in manifest.global_forbidden:
            lines.append(f"- ❌ {rule}")
        lines.append("")
        lines.append("## 不变式规则")
        lines.append("")
        for rule in manifest.global_rules:
            lines.append(f"- 🔒 {rule}")
        lines.append("")
        lines.append("## 工具权限")
        lines.append("")
        lines.append("| 工具 | 允许 | 风险等级 | 需确认 | 禁止模式 | 说明 |")
        lines.append("|------|------|---------|--------|---------|------|")
        for t in manifest.tools:
            allowed_icon = "✅" if t.allowed else "❌"
            confirm_icon = "是" if t.requires_confirmation else "-"
            forbidden = ", ".join(t.forbidden_in) if t.forbidden_in else "-"
            lines.append(f"| {t.name} | {allowed_icon} | {t.risk_level} | {confirm_icon} | {forbidden} | {t.description} |")
        lines.append("")

        self._markdown_path().write_text("\n".join(lines), encoding="utf-8")

    def _default_manifest(self) -> PermissionManifest:
        return PermissionManifest(
            tools=list(DEFAULT_TOOLS),
            global_forbidden=list(DEFAULT_GLOBAL_FORBIDDEN),
            global_rules=list(DEFAULT_GLOBAL_RULES),
        )
