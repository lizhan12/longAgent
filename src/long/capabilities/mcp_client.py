"""MCP Client

实现与 MCP Server 的连接、工具发现和调用。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    """MCP 服务器配置

    Attributes:
        name: 服务器名称
        command: 启动命令
        args: 命令参数
        env: 环境变量
        transport: 传输方式 (stdio, sse)
    """

    name: str
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    transport: str = "stdio"


class MCPToolInfo(BaseModel):
    """MCP 工具信息"""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPResourceInfo(BaseModel):
    """MCP 资源信息"""

    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = ""


class MCPPromptInfo(BaseModel):
    """MCP Prompt 信息"""

    name: str
    description: str = ""
    arguments: list[dict[str, Any]] = Field(default_factory=list)


class MCPClient:
    """MCP Client

    管理与多个 MCP Server 的连接，提供工具发现和调用。

    Attributes:
        _servers: 已连接的服务器配置
        _processes: 服务器子进程
        _tools: 已发现的工具缓存
        _resources: 已发现的资源缓存
        _prompts: 已发现的 Prompt 缓存
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConfig] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._tools: dict[str, list[MCPToolInfo]] = {}
        self._resources: dict[str, list[MCPResourceInfo]] = {}
        self._prompts: dict[str, list[MCPPromptInfo]] = {}
        self._request_id: int = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def connect(self, config: MCPServerConfig) -> bool:
        """连接 MCP Server

        启动子进程并与服务器建立连接。

        Args:
            config: 服务器配置

        Returns:
            是否连接成功
        """
        if config.name in self._servers:
            logger.warning("MCP server '%s' already connected", config.name)
            return True

        self._servers[config.name] = config

        if config.transport == "stdio" and config.command:
            try:
                env = dict(__import__("os").environ)
                env.update(config.env)

                process = await asyncio.create_subprocess_exec(
                    config.command,
                    *config.args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
                self._processes[config.name] = process

                # 发送初始化请求
                init_request = {
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "long", "version": "0.1.0"},
                    },
                }

                await self._send_request(config.name, init_request)

                # 发送 initialized 通知
                initialized_notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                }
                await self._send_notification(config.name, initialized_notification)

                logger.info("Connected to MCP server: %s", config.name)
                return True

            except Exception as e:
                logger.error("Failed to connect to MCP server '%s': %s", config.name, e)
                self._servers.pop(config.name, None)
                return False

        # 没有命令的配置，标记为已连接（可能是模拟服务器）
        return True

    async def discover_tools(self, server_name: str) -> list[MCPToolInfo]:
        """发现 MCP Server 提供的工具

        Args:
            server_name: 服务器名称

        Returns:
            工具列表
        """
        if server_name not in self._servers:
            logger.warning("MCP server '%s' not connected", server_name)
            return []

        if server_name in self._processes:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/list",
                "params": {},
            }
            response = await self._send_request(server_name, request)

            if response and "result" in response and "tools" in response["result"]:
                tools = []
                for tool_data in response["result"]["tools"]:
                    tools.append(MCPToolInfo(
                        name=tool_data.get("name", ""),
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("inputSchema", {}),
                    ))
                self._tools[server_name] = tools
                return tools

        return self._tools.get(server_name, [])

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """调用 MCP Server 的工具

        Args:
            server_name: 服务器名称
            tool_name: 工具名称
            arguments: 调用参数

        Returns:
            调用结果
        """
        if server_name not in self._servers:
            raise ValueError(f"MCP server '{server_name}' not connected")

        if server_name in self._processes:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments or {},
                },
            }
            response = await self._send_request(server_name, request)

            if response and "result" in response:
                return response["result"]

        raise RuntimeError(f"Failed to call tool '{tool_name}' on server '{server_name}'")

    async def discover_resources(self, server_name: str) -> list[MCPResourceInfo]:
        """发现 MCP Server 提供的资源

        Args:
            server_name: 服务器名称

        Returns:
            资源列表
        """
        if server_name not in self._servers:
            return []

        if server_name in self._processes:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "resources/list",
                "params": {},
            }
            response = await self._send_request(server_name, request)

            if response and "result" in response and "resources" in response["result"]:
                resources = []
                for res_data in response["result"]["resources"]:
                    resources.append(MCPResourceInfo(
                        uri=res_data.get("uri", ""),
                        name=res_data.get("name", ""),
                        description=res_data.get("description", ""),
                        mime_type=res_data.get("mimeType", ""),
                    ))
                self._resources[server_name] = resources
                return resources

        return self._resources.get(server_name, [])

    async def read_resource(self, server_name: str, uri: str) -> Any:
        """读取 MCP Server 的资源

        Args:
            server_name: 服务器名称
            uri: 资源 URI

        Returns:
            资源内容
        """
        if server_name not in self._servers:
            raise ValueError(f"MCP server '{server_name}' not connected")

        if server_name in self._processes:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "resources/read",
                "params": {"uri": uri},
            }
            response = await self._send_request(server_name, request)

            if response and "result" in response:
                return response["result"]

        raise RuntimeError(f"Failed to read resource '{uri}' from server '{server_name}'")

    async def discover_prompts(self, server_name: str) -> list[MCPPromptInfo]:
        """发现 MCP Server 提供的 Prompt

        Args:
            server_name: 服务器名称

        Returns:
            Prompt 列表
        """
        if server_name not in self._servers:
            return []

        if server_name in self._processes:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "prompts/list",
                "params": {},
            }
            response = await self._send_request(server_name, request)

            if response and "result" in response and "prompts" in response["result"]:
                prompts = []
                for p_data in response["result"]["prompts"]:
                    prompts.append(MCPPromptInfo(
                        name=p_data.get("name", ""),
                        description=p_data.get("description", ""),
                        arguments=p_data.get("arguments", []),
                    ))
                self._prompts[server_name] = prompts
                return prompts

        return self._prompts.get(server_name, [])

    async def disconnect(self, server_name: str) -> bool:
        """断开与 MCP Server 的连接

        Args:
            server_name: 服务器名称

        Returns:
            是否成功断开
        """
        process = self._processes.pop(server_name, None)
        if process:
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
            except Exception:
                pass

        self._servers.pop(server_name, None)
        self._tools.pop(server_name, None)
        self._resources.pop(server_name, None)
        self._prompts.pop(server_name, None)

        logger.info("Disconnected from MCP server: %s", server_name)
        return True

    async def _send_request(
        self, server_name: str, request: dict[str, Any]
    ) -> dict[str, Any] | None:
        """发送 JSON-RPC 请求

        Args:
            server_name: 服务器名称
            request: 请求体

        Returns:
            响应体
        """
        process = self._processes.get(server_name)
        if process is None or process.stdin is None or process.stdout is None:
            return None

        try:
            message = json.dumps(request) + "\n"
            process.stdin.write(message.encode())
            await process.stdin.drain()

            response_line = await asyncio.wait_for(
                process.stdout.readline(), timeout=30.0
            )
            if response_line:
                return json.loads(response_line.decode())
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for response from MCP server '%s'", server_name)
        except Exception as e:
            logger.error("Error communicating with MCP server '%s': %s", server_name, e)

        return None

    async def _send_notification(
        self, server_name: str, notification: dict[str, Any]
    ) -> None:
        """发送 JSON-RPC 通知（不期待响应）

        Args:
            server_name: 服务器名称
            notification: 通知体
        """
        process = self._processes.get(server_name)
        if process is None or process.stdin is None:
            return

        try:
            message = json.dumps(notification) + "\n"
            process.stdin.write(message.encode())
            await process.stdin.drain()
        except Exception as e:
            logger.error("Error sending notification to MCP server '%s': %s", server_name, e)
