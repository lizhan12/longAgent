"""统一工具注册表

统一管理本地工具、MCP 工具和 Skill 工具的注册与调用。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .tool_capability import ToolCapability

logger = logging.getLogger(__name__)


class ToolSource(str, Enum):
    """工具来源"""

    LOCAL = "local"
    MCP = "mcp"
    SKILL = "skill"


class ToolDefinition(BaseModel):
    """工具定义

    Attributes:
        name: 工具名称（全局唯一）
        description: 工具描述
        source: 工具来源
        source_name: 来源名称（MCP server 名称或 Skill 名称）
        parameters: 参数 Schema（JSON Schema 格式）
        capability: 工具能力模型
    """

    name: str
    description: str = ""
    source: ToolSource = ToolSource.LOCAL
    source_name: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    capability: ToolCapability = Field(default_factory=ToolCapability)
    model_config = {"arbitrary_types_allowed": True}


class ToolCallResult(BaseModel):
    """工具调用结果"""

    success: bool = True
    output: Any = None
    error: str | None = None


class UnifiedToolRegistry:
    """统一工具注册表

    将本地工具、MCP 工具和 Skill 工具统一注册到一个注册表中，
    提供统一的查询和调用接口。

    Attributes:
        _tools: 工具注册表
        _handlers: 本地工具处理函数
        _mcp_caller: MCP 工具调用函数
        _skill_caller: Skill 工具调用函数
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable[..., Awaitable[Any]]] = {}
        self._mcp_caller: Callable[..., Awaitable[Any]] | None = None
        self._skill_caller: Callable[..., Awaitable[Any]] | None = None

    def set_mcp_caller(self, caller: Callable[..., Awaitable[Any]]) -> None:
        """设置 MCP 工具调用函数"""
        self._mcp_caller = caller

    def set_skill_caller(self, caller: Callable[..., Awaitable[Any]]) -> None:
        """设置 Skill 工具调用函数"""
        self._skill_caller = caller

    def register_local(
        self,
        tool_def: ToolDefinition,
        handler: Callable[..., Awaitable[Any]],
    ) -> None:
        """注册本地工具

        Args:
            tool_def: 工具定义
            handler: 处理函数
        """
        tool_def.source = ToolSource.LOCAL
        self._tools[tool_def.name] = tool_def
        self._handlers[tool_def.name] = handler
        logger.debug("Registered local tool: %s", tool_def.name)

    def register_mcp(
        self,
        server_name: str,
        tool_def: ToolDefinition,
    ) -> None:
        """注册 MCP 工具

        Args:
            server_name: MCP 服务器名称
            tool_def: 工具定义
        """
        tool_def.source = ToolSource.MCP
        tool_def.source_name = server_name
        self._tools[tool_def.name] = tool_def
        logger.debug("Registered MCP tool: %s (from %s)", tool_def.name, server_name)

    def register_skill(
        self,
        skill_name: str,
        tool_def: ToolDefinition,
    ) -> None:
        """注册 Skill 工具

        Args:
            skill_name: Skill 名称
            tool_def: 工具定义
        """
        tool_def.source = ToolSource.SKILL
        tool_def.source_name = skill_name
        self._tools[tool_def.name] = tool_def
        logger.debug("Registered Skill tool: %s (from %s)", tool_def.name, skill_name)

    def get_tool(self, tool_name: str) -> ToolDefinition | None:
        """获取工具定义"""
        return self._tools.get(tool_name)

    def get_handlers(self) -> dict[str, Callable[..., Awaitable[Any]]]:
        """获取所有本地工具处理器

        Returns:
            工具名称 → 处理器函数的映射
        """
        return dict(self._handlers)

    def list_tools(
        self,
        source: ToolSource | None = None,
    ) -> list[ToolDefinition]:
        """列出工具

        Args:
            source: 可选来源过滤

        Returns:
            工具列表
        """
        if source is None:
            return list(self._tools.values())
        return [t for t in self._tools.values() if t.source == source]

    def unregister(self, tool_name: str) -> bool:
        """反注册工具

        Args:
            tool_name: 工具名称

        Returns:
            是否成功反注册
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            self._handlers.pop(tool_name, None)
            return True
        return False

    async def call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        _context: dict[str, Any] | None = None,
    ) -> ToolCallResult:
        """统一调用入口

        根据工具来源路由到不同的调用器。

        Args:
            tool_name: 工具名称
            arguments: 调用参数
            context: 调用上下文

        Returns:
            调用结果
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolCallResult(
                success=False,
                error=f"Tool '{tool_name}' not found",
            )

        args = arguments or {}
        try:
            if tool.source == ToolSource.LOCAL:
                handler = self._handlers.get(tool_name)
                if handler is None:
                    return ToolCallResult(
                        success=False,
                        error=f"No handler for local tool '{tool_name}'",
                    )
                result = await handler(**args)
                return ToolCallResult(success=True, output=result)

            elif tool.source == ToolSource.MCP:
                if self._mcp_caller is None:
                    return ToolCallResult(
                        success=False,
                        error="MCP caller not configured",
                    )
                result = await self._mcp_caller(
                    server_name=tool.source_name,
                    tool_name=tool_name,
                    arguments=args,
                )
                return ToolCallResult(success=True, output=result)

            elif tool.source == ToolSource.SKILL:
                if self._skill_caller is None:
                    return ToolCallResult(
                        success=False,
                        error="Skill caller not configured",
                    )
                result = await self._skill_caller(
                    skill_name=tool.source_name,
                    tool_name=tool_name,
                    arguments=args,
                )
                return ToolCallResult(success=True, output=result)

            else:
                return ToolCallResult(
                    success=False,
                    error=f"Unknown tool source: {tool.source}",
                )

        except Exception as e:
            return ToolCallResult(
                success=False,
                error=str(e),
            )
