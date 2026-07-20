"""Capabilities 模块 - MCP 与 Skill 系统

提供 MCP Server 连接、Skill 管理和统一工具注册。
"""

from .mcp_client import MCPClient, MCPServerConfig, MCPToolInfo
from .mcp_server import SystemMCPServer
from .skill_loader import SkillLoader
from .skill_manager import SkillManager, SkillManifest, SkillRecord, SkillState
from .unified_tool_registry import (
    ToolCallResult,
    ToolDefinition,
    ToolSource,
    UnifiedToolRegistry,
)

__all__ = [
    "MCPClient",
    "MCPServerConfig",
    "MCPToolInfo",
    "SkillLoader",
    "SkillManager",
    "SkillManifest",
    "SkillRecord",
    "SkillState",
    "SystemMCPServer",
    "ToolCallResult",
    "ToolDefinition",
    "ToolSource",
    "UnifiedToolRegistry",
]
