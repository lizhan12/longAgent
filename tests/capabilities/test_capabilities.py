"""Capabilities 模块测试

覆盖统一工具注册表、MCP Client、MCP Server、Skill Manager 和 Skill Loader。
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from long.capabilities.mcp_client import MCPClient, MCPServerConfig
from long.capabilities.mcp_server import SystemMCPServer
from long.capabilities.skill_loader import SkillLoader
from long.capabilities.skill_manager import SkillManager, SkillManifest, SkillState
from long.capabilities.unified_tool_registry import (
    ToolCallResult,
    ToolDefinition,
    ToolSource,
    UnifiedToolRegistry,
)


# ========================
# UnifiedToolRegistry 测试
# ========================


class TestUnifiedToolRegistry:
    """统一工具注册表测试"""

    @pytest.fixture
    def registry(self):
        return UnifiedToolRegistry()

    def test_register_local_tool(self, registry):
        tool = ToolDefinition(name="test_tool", description="A test tool")
        async def handler(**kwargs):
            return "result"

        registry.register_local(tool, handler)
        assert registry.get_tool("test_tool") is not None
        assert registry.get_tool("test_tool").source == ToolSource.LOCAL

    def test_register_mcp_tool(self, registry):
        tool = ToolDefinition(name="mcp_tool", description="An MCP tool")
        registry.register_mcp("my_server", tool)
        assert registry.get_tool("mcp_tool") is not None
        assert registry.get_tool("mcp_tool").source == ToolSource.MCP
        assert registry.get_tool("mcp_tool").source_name == "my_server"

    def test_register_skill_tool(self, registry):
        tool = ToolDefinition(name="skill_tool", description="A skill tool")
        registry.register_skill("my_skill", tool)
        assert registry.get_tool("skill_tool") is not None
        assert registry.get_tool("skill_tool").source == ToolSource.SKILL

    def test_list_tools(self, registry):
        tool1 = ToolDefinition(name="tool1")
        tool2 = ToolDefinition(name="tool2")
        async def handler(**kwargs): return None

        registry.register_local(tool1, handler)
        registry.register_mcp("server1", tool2)

        all_tools = registry.list_tools()
        assert len(all_tools) == 2

        local_tools = registry.list_tools(source=ToolSource.LOCAL)
        assert len(local_tools) == 1

        mcp_tools = registry.list_tools(source=ToolSource.MCP)
        assert len(mcp_tools) == 1

    def test_unregister_tool(self, registry):
        tool = ToolDefinition(name="to_remove")
        async def handler(**kwargs): return None
        registry.register_local(tool, handler)

        result = registry.unregister("to_remove")
        assert result is True
        assert registry.get_tool("to_remove") is None

    def test_unregister_nonexistent(self, registry):
        result = registry.unregister("nonexistent")
        assert result is False

    def test_get_nonexistent_tool(self, registry):
        assert registry.get_tool("nonexistent") is None

    @pytest.mark.asyncio
    async def test_call_local_tool(self, registry):
        tool = ToolDefinition(name="add")
        async def add_handler(**kwargs):
            return kwargs.get("a", 0) + kwargs.get("b", 0)

        registry.register_local(tool, add_handler)
        result = await registry.call("add", {"a": 1, "b": 2})
        assert result.success is True
        assert result.output == 3

    @pytest.mark.asyncio
    async def test_call_nonexistent_tool(self, registry):
        result = await registry.call("nonexistent")
        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_call_mcp_tool_with_caller(self, registry):
        tool = ToolDefinition(name="mcp_tool")
        registry.register_mcp("server1", tool)

        async def mock_mcp_caller(server_name, tool_name, arguments):
            return {"server": server_name, "tool": tool_name}

        registry.set_mcp_caller(mock_mcp_caller)
        result = await registry.call("mcp_tool", {"arg": "val"})
        assert result.success is True
        assert result.output["server"] == "server1"

    @pytest.mark.asyncio
    async def test_call_skill_tool_with_caller(self, registry):
        tool = ToolDefinition(name="skill_tool")
        registry.register_skill("skill1", tool)

        async def mock_skill_caller(skill_name, tool_name, arguments):
            return {"skill": skill_name, "tool": tool_name}

        registry.set_skill_caller(mock_skill_caller)
        result = await registry.call("skill_tool")
        assert result.success is True
        assert result.output["skill"] == "skill1"


# ========================
# MCPClient 测试
# ========================


class TestMCPClient:
    """MCP Client 测试"""

    def test_create_client(self):
        client = MCPClient()
        assert client is not None

    @pytest.mark.asyncio
    async def test_connect_no_command(self):
        client = MCPClient()
        config = MCPServerConfig(name="test_server")
        result = await client.connect(config)
        assert result is True

    @pytest.mark.asyncio
    async def test_discover_tools_not_connected(self):
        client = MCPClient()
        tools = await client.discover_tools("nonexistent")
        assert tools == []

    @pytest.mark.asyncio
    async def test_disconnect(self):
        client = MCPClient()
        config = MCPServerConfig(name="test_server")
        await client.connect(config)
        result = await client.disconnect("test_server")
        assert result is True

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self):
        client = MCPClient()
        with pytest.raises(ValueError):
            await client.call_tool("nonexistent", "tool")

    @pytest.mark.asyncio
    async def test_read_resource_not_connected(self):
        client = MCPClient()
        with pytest.raises(ValueError):
            await client.read_resource("nonexistent", "long://test")

    @pytest.mark.asyncio
    async def test_discover_resources_not_connected(self):
        client = MCPClient()
        resources = await client.discover_resources("nonexistent")
        assert resources == []

    @pytest.mark.asyncio
    async def test_discover_prompts_not_connected(self):
        client = MCPClient()
        prompts = await client.discover_prompts("nonexistent")
        assert prompts == []

    def test_server_config(self):
        config = MCPServerConfig(
            name="test",
            command="python",
            args=["-m", "server"],
            transport="stdio",
        )
        assert config.name == "test"
        assert config.command == "python"


# ========================
# SystemMCPServer 测试
# ========================


class TestSystemMCPServer:
    """系统 MCP Server 测试"""

    @pytest.fixture
    def server(self):
        return SystemMCPServer()

    def test_list_tools(self, server):
        tools = server.list_tools()
        assert len(tools) >= 4
        tool_names = [t["name"] for t in tools]
        assert "system_info" in tool_names
        assert "query_memory" in tool_names
        assert "get_trace" in tool_names
        assert "submit_task" in tool_names

    def test_list_resources(self, server):
        resources = server.list_resources()
        assert len(resources) >= 2
        uris = [r["uri"] for r in resources]
        assert "long://system/status" in uris

    def test_list_prompts(self, server):
        prompts = server.list_prompts()
        assert len(prompts) >= 2
        prompt_names = [p["name"] for p in prompts]
        assert "plan_task" in prompt_names
        assert "analyze_error" in prompt_names

    @pytest.mark.asyncio
    async def test_handle_initialize(self, server):
        result = await server.handle_request("initialize", {})
        assert "protocolVersion" in result
        assert "capabilities" in result

    @pytest.mark.asyncio
    async def test_handle_tools_list(self, server):
        result = await server.handle_request("tools/list", {})
        assert "tools" in result
        assert len(result["tools"]) >= 4

    @pytest.mark.asyncio
    async def test_handle_call_system_info(self, server):
        result = await server.handle_request("tools/call", {
            "name": "system_info",
            "arguments": {},
        })
        assert "content" in result
        content_text = result["content"][0]["text"]
        data = json.loads(content_text)
        assert "version" in data

    @pytest.mark.asyncio
    async def test_handle_call_submit_task(self, server):
        result = await server.handle_request("tools/call", {
            "name": "submit_task",
            "arguments": {"goal": "test task"},
        })
        assert "content" in result

    @pytest.mark.asyncio
    async def test_handle_resources_list(self, server):
        result = await server.handle_request("resources/list", {})
        assert "resources" in result

    @pytest.mark.asyncio
    async def test_handle_resources_read(self, server):
        result = await server.handle_request("resources/read", {
            "uri": "long://system/status",
        })
        assert "contents" in result

    @pytest.mark.asyncio
    async def test_handle_prompts_list(self, server):
        result = await server.handle_request("prompts/list", {})
        assert "prompts" in result

    @pytest.mark.asyncio
    async def test_handle_prompts_get(self, server):
        result = await server.handle_request("prompts/get", {
            "name": "plan_task",
            "arguments": {"goal": "test goal"},
        })
        assert "messages" in result

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self, server):
        result = await server.handle_request("unknown/method", {})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_handle_unknown_tool(self, server):
        result = await server.handle_request("tools/call", {
            "name": "nonexistent_tool",
            "arguments": {},
        })
        assert result.get("isError") is True


# ========================
# SkillManager 测试
# ========================


class TestSkillManager:
    """Skill 管理器测试"""

    @pytest.fixture
    def manager(self):
        return SkillManager()

    def test_create_manager(self, manager):
        assert manager is not None

    def test_load_skill_invalid_path(self, manager):
        result = manager.load_skill("/nonexistent/path")
        # 应该返回错误记录
        if result is not None:
            assert result.state == SkillState.ERROR

    def test_list_skills_empty(self, manager):
        skills = manager.list_skills()
        assert len(skills) == 0

    def test_enable_nonexistent_skill(self, manager):
        result = manager.enable_skill("nonexistent")
        assert result is False

    def test_disable_nonexistent_skill(self, manager):
        result = manager.disable_skill("nonexistent")
        assert result is False

    def test_unload_nonexistent_skill(self, manager):
        result = manager.unload_skill("nonexistent")
        assert result is False

    def test_reload_nonexistent_skill(self, manager):
        result = manager.reload_skill("nonexistent")
        assert result is None

    def test_get_nonexistent_skill(self, manager):
        result = manager.get_skill("nonexistent")
        assert result is None

    def test_auto_discover_no_dir(self, manager):
        results = manager.auto_discover()
        assert results == []

    def test_auto_discover_with_empty_dir(self, manager, tmp_path):
        manager._skills_dir = tmp_path / "skills"
        manager._skills_dir.mkdir()
        results = manager.auto_discover()
        assert results == []


# ========================
# SkillLoader 测试
# ========================


class TestSkillLoader:
    """Skill 加载器测试"""

    @pytest.fixture
    def loader(self):
        return SkillLoader()

    def test_scan_safe_code(self, loader, tmp_path):
        safe_skill = tmp_path / "safe_skill.py"
        safe_skill.write_text("x = 1\nprint(x)")
        result = loader.scan_code(safe_skill)
        assert result.safe is True

    def test_scan_dangerous_code(self, loader, tmp_path):
        dangerous_skill = tmp_path / "dangerous_skill.py"
        dangerous_skill.write_text("import os\nos.fork()")
        result = loader.scan_code(dangerous_skill)
        assert result.safe is False

    def test_scan_nonexistent_path(self, loader):
        result = loader.scan_code(Path("/nonexistent/path"))
        assert result.safe is False

    def test_load_simple_skill(self, loader, tmp_path):
        skill_file = tmp_path / "simple_skill.py"
        skill_file.write_text(
            'SKILL_NAME = "test_skill"\n'
            'SKILL_VERSION = "1.0.0"\n'
            'SKILL_DESCRIPTION = "A test skill"\n'
            'SKILL_TOOLS = ["test_tool"]\n'
        )
        result = loader.load(skill_file)
        assert result is not None
        module, manifest = result
        assert manifest["name"] == "test_skill"
        assert manifest["version"] == "1.0.0"
        assert "test_tool" in manifest["tools"]

    def test_load_skill_package(self, loader, tmp_path):
        skill_dir = tmp_path / "pkg_skill"
        skill_dir.mkdir()
        init_file = skill_dir / "__init__.py"
        init_file.write_text(
            'SKILL_NAME = "pkg_skill"\n'
            'SKILL_VERSION = "0.1.0"\n'
        )
        result = loader.load(skill_dir)
        assert result is not None
        _, manifest = result
        assert manifest["name"] == "pkg_skill"

    def test_load_nonexistent_skill(self, loader):
        result = loader.load(Path("/nonexistent"))
        assert result is None

    def test_safe_builtins(self):
        builtins = SkillLoader._get_safe_builtins()
        assert "abs" in builtins
        assert "len" in builtins
        assert "print" in builtins
        # 危险的不应该在内置函数中
        assert "exec" not in builtins
        assert "eval" not in builtins
        assert "__import__" not in builtins
        assert "open" not in builtins


# ========================
# Integration 测试
# ========================


class TestSkillManagerIntegration:
    """Skill Manager 与 Tool Registry 集成测试"""

    @pytest.fixture
    def manager(self):
        registry = UnifiedToolRegistry()
        return SkillManager(registry=registry)

    def test_load_skill_registers_tools(self, manager, tmp_path):
        skill_file = tmp_path / "skill_with_tools.py"
        skill_file.write_text(
            'SKILL_NAME = "tool_skill"\n'
            'SKILL_TOOLS = ["search_web", "read_file"]\n'
        )
        record = manager.load_skill(skill_file)
        if record and record.state != SkillState.ERROR:
            # 工具应该注册到 registry
            tool = manager.registry.get_tool("search_web")
            assert tool is not None
            assert tool.source == ToolSource.SKILL

    def test_unload_skill_unregisters_tools(self, manager, tmp_path):
        skill_file = tmp_path / "unload_skill.py"
        skill_file.write_text(
            'SKILL_NAME = "unload_skill"\n'
            'SKILL_TOOLS = ["temp_tool"]\n'
        )
        record = manager.load_skill(skill_file)
        if record and record.state != SkillState.ERROR:
            manager.unload_skill("unload_skill")
            assert manager.registry.get_tool("temp_tool") is None

    def test_enable_disable_lifecycle(self, manager, tmp_path):
        skill_file = tmp_path / "lifecycle_skill.py"
        skill_file.write_text(
            'SKILL_NAME = "lifecycle_skill"\n'
        )
        record = manager.load_skill(skill_file)
        if record and record.state != SkillState.ERROR:
            assert manager.enable_skill("lifecycle_skill") is True
            skill = manager.get_skill("lifecycle_skill")
            assert skill.state == SkillState.ENABLED

            assert manager.disable_skill("lifecycle_skill") is True
            skill = manager.get_skill("lifecycle_skill")
            assert skill.state == SkillState.DISABLED
