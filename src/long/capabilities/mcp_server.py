"""MCP Server

暴露系统内部状态作为 MCP 工具/资源/Prompt。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SystemMCPServer:
    """系统 MCP Server

    将系统内部状态暴露为 MCP 工具、资源和 Prompt 模板。

    Attributes:
        _tools: 注册的工具
        _resources: 注册的资源
        _prompts: 注册的 Prompt 模板
        _state: 系统状态
        _memory_controller: 记忆控制器（可选，用于 query_memory）
    """

    def __init__(
        self,
        memory_controller: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._tools: dict[str, dict[str, Any]] = {}
        self._resources: dict[str, dict[str, Any]] = {}
        self._prompts: dict[str, dict[str, Any]] = {}
        self._memory_controller = memory_controller
        self._config = config or {}
        self._state: dict[str, Any] = {
            "version": "0.1.0",
            "started_at": time.time(),
        }

        self._register_default_tools()
        self._register_default_resources()
        self._register_default_prompts()

    def _register_default_tools(self) -> None:
        """注册默认系统工具"""
        self._tools["system_info"] = {
            "name": "system_info",
            "description": "获取系统信息（版本、运行时间等）",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        }
        self._tools["query_memory"] = {
            "name": "query_memory",
            "description": "查询记忆系统",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查询内容"},
                    "memory_type": {"type": "string", "description": "记忆类型"},
                    "limit": {"type": "integer", "description": "返回数量限制"},
                },
                "required": ["query"],
            },
        }
        self._tools["get_trace"] = {
            "name": "get_trace",
            "description": "获取执行轨迹",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "plan_id": {"type": "string", "description": "Plan ID"},
                },
                "required": ["plan_id"],
            },
        }
        self._tools["submit_task"] = {
            "name": "submit_task",
            "description": "提交任务到系统",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "任务目标"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["goal"],
            },
        }

    def _register_default_resources(self) -> None:
        """注册默认系统资源"""
        self._resources["long://system/status"] = {
            "uri": "long://system/status",
            "name": "系统状态",
            "description": "当前系统运行状态",
            "mimeType": "application/json",
        }
        self._resources["long://system/config"] = {
            "uri": "long://system/config",
            "name": "系统配置",
            "description": "当前系统配置信息",
            "mimeType": "application/json",
        }

    def _register_default_prompts(self) -> None:
        """注册默认 Prompt 模板"""
        self._prompts["plan_task"] = {
            "name": "plan_task",
            "description": "生成任务执行计划",
            "arguments": [
                {"name": "goal", "description": "任务目标", "required": True},
                {"name": "constraints", "description": "约束条件", "required": False},
            ],
        }
        self._prompts["analyze_error"] = {
            "name": "analyze_error",
            "description": "分析执行错误并提供修复建议",
            "arguments": [
                {"name": "error", "description": "错误信息", "required": True},
                {"name": "context", "description": "执行上下文", "required": False},
            ],
        }

    def list_tools(self) -> list[dict[str, Any]]:
        """列出所有工具"""
        return list(self._tools.values())

    def list_resources(self) -> list[dict[str, Any]]:
        """列出所有资源"""
        return list(self._resources.values())

    def list_prompts(self) -> list[dict[str, Any]]:
        """列出所有 Prompt"""
        return list(self._prompts.values())

    async def handle_request(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """处理 MCP 请求

        Args:
            method: 方法名
            params: 参数

        Returns:
            响应
        """
        if method == "initialize":
            return {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"subscribe": True, "listChanged": True},
                    "prompts": {"listChanged": True},
                },
                "serverInfo": {"name": "long-system", "version": "0.1.0"},
            }

        elif method == "tools/list":
            return {"tools": self.list_tools()}

        elif method == "tools/call":
            return await self._call_tool(params)

        elif method == "resources/list":
            return {"resources": self.list_resources()}

        elif method == "resources/read":
            return self._read_resource(params)

        elif method == "prompts/list":
            return {"prompts": self.list_prompts()}

        elif method == "prompts/get":
            return self._get_prompt(params)

        else:
            return {"error": {"code": -32601, "message": f"Method not found: {method}"}}

    async def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        """调用工具"""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name == "system_info":
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "version": self._state["version"],
                            "uptime_seconds": time.time() - self._state["started_at"],
                            "tools_count": len(self._tools),
                            "resources_count": len(self._resources),
                        }),
                    }
                ]
            }

        elif tool_name == "query_memory":
            if self._memory_controller is not None:
                try:
                    query = arguments.get("query", "")
                    memory_type_str = arguments.get("memory_type")
                    limit = arguments.get("limit", 10)
                    from long.memory.base import MemoryType

                    memory_type = None
                    if memory_type_str:
                        try:
                            memory_type = MemoryType(memory_type_str)
                        except ValueError:
                            pass
                    results = await self._memory_controller.search(
                        query=query, memory_type=memory_type, limit=limit
                    )
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({
                                    "query": query,
                                    "results": [
                                        {"id": r.id, "content": r.content, "importance": r.importance}
                                        for r in results
                                    ],
                                    "count": len(results),
                                }),
                            }
                        ]
                    }
                except Exception as e:
                    return {
                        "isError": True,
                        "content": [{"type": "text", "text": f"Memory query error: {e}"}],
                    }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "query": arguments.get("query", ""),
                            "results": [],
                            "message": "Memory controller not configured",
                        }),
                    }
                ]
            }

        elif tool_name == "get_trace":
            plan_id = arguments.get("plan_id", "")
            trace_dir = self._config.get("traces_dir")
            if trace_dir:
                from pathlib import Path

                trace_path = Path(trace_dir) / f"{plan_id}.json"
                if trace_path.exists():
                    with open(trace_path, "r", encoding="utf-8") as f:
                        trace_data = json.load(f)
                    return {
                        "content": [
                            {"type": "text", "text": json.dumps({"plan_id": plan_id, "trace": trace_data})}
                        ]
                    }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "plan_id": plan_id,
                            "trace": [],
                            "message": "No trace data available",
                        }),
                    }
                ]
            }

        elif tool_name == "submit_task":
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "status": "accepted",
                            "goal": arguments.get("goal", ""),
                        }),
                    }
                ]
            }

        else:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
            }

    def _read_resource(self, params: dict[str, Any]) -> dict[str, Any]:
        """读取资源"""
        uri = params.get("uri", "")

        if uri == "long://system/status":
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps({
                            "status": "running",
                            "version": self._state["version"],
                            "uptime": time.time() - self._state["started_at"],
                        }),
                    }
                ]
            }
        elif uri == "long://system/config":
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps({"config": "pending_integration"}),
                    }
                ]
            }

        return {"contents": []}

    def _get_prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        """获取 Prompt"""
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        if name == "plan_task":
            goal = arguments.get("goal", "")
            constraints = arguments.get("constraints", "")
            return {
                "description": "生成任务执行计划",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"请为以下目标生成执行计划:\n目标: {goal}\n"
                            + (f"约束: {constraints}" if constraints else ""),
                        },
                    }
                ],
            }
        elif name == "analyze_error":
            error = arguments.get("error", "")
            return {
                "description": "分析执行错误",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"请分析以下执行错误并提供修复建议:\n{error}",
                        },
                    }
                ],
            }

        return {"messages": []}
