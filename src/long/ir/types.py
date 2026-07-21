"""IR 类型系统

定义 Agent 行为参数类型，作为 ActionType 的参数约束。
每种 ActionType 对应一组参数类型，用于运行时 Schema 校验。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchArgs(BaseModel):
    """搜索参数"""

    query: str = Field(..., min_length=1, description="搜索查询")
    top_k: int = Field(default=5, ge=1, le=100, description="返回结果数量")


class CallToolArgs(BaseModel):
    """工具调用参数"""

    tool_name: str = Field(..., min_length=1, description="工具名称")
    parameters: dict[str, Any] = Field(default_factory=dict, description="工具参数")


class CallAPIArgs(BaseModel):
    """API 调用参数"""

    endpoint: str = Field(..., min_length=1, description="API 端点")
    method: str = Field(default="GET", description="HTTP 方法")
    body: dict[str, Any] | None = Field(default=None, description="请求体")


class CallMCPArgs(BaseModel):
    """MCP 工具调用参数"""

    server_name: str = Field(..., min_length=1, description="MCP 服务器名称")
    tool_name: str = Field(..., min_length=1, description="远程工具名称")
    arguments: dict[str, Any] = Field(default_factory=dict, description="工具参数")


class CallSkillArgs(BaseModel):
    """Skill 调用参数"""

    skill_name: str = Field(..., min_length=1, description="Skill 名称")
    tool_name: str = Field(default="", description="Skill 内工具名称")
    arguments: dict[str, Any] = Field(default_factory=dict, description="工具参数")


class WriteFileArgs(BaseModel):
    """写文件参数"""

    path: str = Field(..., min_length=1, description="文件路径")
    content: str = Field(..., min_length=1, description="文件内容")


class ReadFileArgs(BaseModel):
    """读文件参数"""

    path: str = Field(..., min_length=1, description="文件路径")


class ExecuteCodeArgs(BaseModel):
    """执行代码参数"""

    code: str = Field(..., min_length=1, description="代码内容")
    language: str = Field(default="python", description="编程语言")


# 参数类型映射：ActionType → 参数类型
ACTION_TYPE_ARGS_MAP: dict[str, type[BaseModel]] = {
    "search": SearchArgs,
    "call_tool": CallToolArgs,
    "call_api": CallAPIArgs,
    "call_mcp": CallMCPArgs,
    "call_skill": CallSkillArgs,
    "write_file": WriteFileArgs,
    "read_file": ReadFileArgs,
    "execute_code": ExecuteCodeArgs,
}


def get_args_type(action: str) -> type[BaseModel] | None:
    """获取 ActionType 对应的参数类型

    Args:
        action: 动作类型字符串

    Returns:
        参数类型，如果未找到则返回 None
    """
    return ACTION_TYPE_ARGS_MAP.get(action)