from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

from long.capabilities.mcp_client import MCPClient
from long.capabilities.skill_manager import SkillManager, SkillState
from long.capabilities.unified_tool_registry import UnifiedToolRegistry
from long.observability.tracing import Tracer, current_trace_id
from long.session.models import Session

logger = logging.getLogger(__name__)

_MAX_TOOL_RESULT_LEN = 8000


class TaskTimeline:
    """任务执行逻辑时钟

    跟踪任务在其生命周期内的逻辑时间线:
    - 创建时间 / 截止时间 / TTL
    - 重试计数 / 上次活动时间
    - 状态转换时间戳

    与物理时钟（System Time）不同，逻辑时钟是 Agent 内部的工作流时间轴。
    """
    __slots__ = (
        "created_at", "deadline", "ttl", "retry_count",
        "last_activity", "step_count", "status",
    )

    def __init__(self, ttl: float = 600.0) -> None:
        import time as _time
        now = _time.time()
        self.created_at: float = now
        self.deadline: float = now + ttl
        self.ttl: float = ttl
        self.retry_count: int = 0
        self.last_activity: float = now
        self.step_count: int = 0
        self.status: str = "created"  # created / running / retry / timeout / done

    def touch(self) -> None:
        """记录活动时间戳"""
        import time as _time
        self.last_activity = _time.time()

    def inc_retry(self) -> int:
        self.retry_count += 1
        return self.retry_count

    def inc_step(self) -> int:
        self.step_count += 1
        return self.step_count

    def is_expired(self) -> bool:
        import time as _time
        return _time.time() > self.deadline

    def remaining(self) -> float:
        import time as _time
        return max(0, self.deadline - _time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "deadline": self.deadline,
            "ttl": self.ttl,
            "retry_count": self.retry_count,
            "last_activity": self.last_activity,
            "step_count": self.step_count,
            "status": self.status,
        }


class ToolManager:
    """工具管理器 — 负责工具执行调度、skill 发现、MCP 连接等。

    注意：工具处理函数（_execute_code, _write_file 等）和工具注册逻辑
    统一在 cli.py 中实现，本类不再重复定义。
    """

    def __init__(
        self,
        *,
        workspace: Any,
        tool_registry: UnifiedToolRegistry | None,
        sandbox: Any,
        sandbox_session_manager: Any,
        skill_manager: SkillManager | None,
        mcp_client: MCPClient | None,
        subagent_registry: Any,
        task_orchestrator: Any,
        llm: Any,
        output_guard: Any,
        feature_flags: Any,
        alert_manager: Any,
        workspace_fs: Any,
        config_dir: Path,
        security_mode: str,
        denied_tools: set[str],
        active_session_getter: Callable[[], Session | None],
        tracer: Tracer,
        configs: dict[str, Any],
        on_llm_stats: Callable[[Any], None] | None = None,
        on_llm_timeout: Callable[[], None] | None = None,
        on_llm_fail: Callable[[], None] | None = None,
        escalation: Any = None,
        tool_cache: Any = None,
    ) -> None:
        self.workspace = workspace
        self.tool_registry = tool_registry
        self.sandbox = sandbox
        self.sandbox_session_manager = sandbox_session_manager
        self.skill_manager = skill_manager
        self.mcp_client = mcp_client
        self.subagent_registry = subagent_registry
        self.task_orchestrator = task_orchestrator
        self.llm = llm
        self.output_guard = output_guard
        self.feature_flags = feature_flags
        self.alert_manager = alert_manager
        self.workspace_fs = workspace_fs
        self.config_dir = config_dir
        self.security_mode = security_mode
        self.denied_tools = denied_tools
        self._active_session_getter = active_session_getter
        self.tracer = tracer
        self._configs = configs
        self.on_llm_stats = on_llm_stats
        self.on_llm_timeout = on_llm_timeout
        self.on_llm_fail = on_llm_fail
        self.escalation = escalation
        self.tool_cache = tool_cache
        self._timeline: TaskTimeline | None = None

    @property
    def active_session(self) -> Session | None:
        return self._active_session_getter()

    @property
    def timeline(self) -> TaskTimeline | None:
        return self._timeline

    def start_task(self, ttl: float = 600.0) -> TaskTimeline:
        """开始一个新任务，创建逻辑时钟"""
        self._timeline = TaskTimeline(ttl=ttl)
        return self._timeline

    def end_task(self) -> None:
        """结束任务，标记完成"""
        if self._timeline is not None:
            self._timeline.status = "done"
            self._timeline = None

    def auto_discover_skills(self) -> None:
        if self.skill_manager is None:
            return

        skills_config = self._configs.get("skills", {}).get("skills", self._configs.get("skills", {}))
        auto_discover = skills_config.get("auto_discover", True)
        if not auto_discover:
            return

        search_paths = skills_config.get("search_paths", [])
        for search_path in search_paths:
            search_dir = Path(search_path)
            if not search_dir.is_absolute():
                search_dir = self.config_dir.parent.resolve() / search_path
            if search_dir.exists() and search_dir.is_dir():
                for skill_path in search_dir.iterdir():
                    if skill_path.is_dir() and self.skill_manager._loader.has_valid_skill_format(skill_path):
                        record = self.skill_manager.load_skill(skill_path)
                        if record and record.state != SkillState.ERROR:
                            self.skill_manager.enable_skill(record.manifest.name)
                            logger.info("Skill 已加载: %s (来自 %s)", record.manifest.name, search_path)

        loaded = self.skill_manager.auto_discover()
        logger.info("从 workspace/skills 发现 %d 个 Skill（同名覆盖项目根目录）", len(loaded))

        total = len(self.skill_manager.list_skills())
        logger.info("总共加载 %d 个 Skill", total)

    async def connect_mcp_servers(self) -> None:
        if self.mcp_client is None:
            return

        mcp_config = self._configs.get("mcp", {}).get("mcp", self._configs.get("mcp", {}))
        servers = mcp_config.get("servers", [])
        if not servers:
            return

        from long.capabilities.mcp_client import MCPServerConfig

        for server_cfg in servers:
            name = server_cfg.get("name", "")
            if not name:
                continue
            config = MCPServerConfig(
                name=name,
                command=server_cfg.get("command", ""),
                args=server_cfg.get("args", []),
                env=server_cfg.get("env", {}),
                transport=server_cfg.get("transport", "stdio"),
            )
            try:
                success = await self.mcp_client.connect(config)
                if success:
                    logger.info("MCP Server 已连接: %s", name)
                else:
                    logger.warning("MCP Server 连接失败: %s", name)
            except Exception as e:
                logger.warning("MCP Server 连接异常: %s - %s", name, e)

    def gather_tools(self) -> list[dict[str, Any]]:
        """从 tool_registry 中收集已注册的工具，生成 OpenAI function calling 格式。

        注意：工具 schema 定义已移至 cli.py 的 _gather_tools()，
        本方法保留用于 SubAgentRunner 等内部组件的兼容性。
        """
        tools: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        if self.tool_registry is not None:
            for tool_def in self.tool_registry.list_tools():
                name = tool_def.name
                if name in seen_names:
                    continue
                tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool_def.description,
                        "parameters": {"type": "object", "properties": {}},
                    },
                    "_source": "local",
                })
                seen_names.add(name)

        if self.skill_manager is not None:
            for skill in self.skill_manager.list_skills():
                for tool_name in skill.manifest.tools:
                    if tool_name in seen_names:
                        continue
                    if not self.tool_registry or not self.tool_registry.get_tool(tool_name):
                        continue
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": f"Skill '{skill.manifest.name}': {skill.manifest.description}",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string", "description": "输入参数"}
                                },
                                "required": ["text"],
                            },
                        },
                        "_source": "skill",
                    })
                    seen_names.add(tool_name)

        if self.mcp_client is not None:
            for server_name in self.mcp_client._servers:
                mcp_tools = self.mcp_client._tools.get(server_name, [])
                for tool_info in mcp_tools:
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": f"{server_name}_{tool_info.name}",
                            "description": f"MCP '{server_name}': {tool_info.description}",
                            "parameters": tool_info.input_schema if tool_info.input_schema else {"type": "object"},
                        },
                        "_source": "mcp",
                        "_mcp_server": server_name,
                    })

        return tools

    def clean_tools_for_api(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned = []
        for tool in tools:
            t = {
                "type": tool.get("type", "function"),
                "function": dict(tool.get("function", {})),
            }
            func = t["function"]
            if "parameters" not in func:
                func["parameters"] = {"type": "object", "properties": {}}
            params = func["parameters"]
            if "type" not in params:
                params["type"] = "object"
            if "properties" not in params:
                params["properties"] = {}
            cleaned.append(t)
        return cleaned

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        # 检查任务是否超时
        if self._timeline is not None:
            self._timeline.touch()
            self._timeline.inc_step()
            if self._timeline.is_expired():
                return "任务执行已超时，请简化任务或重试"

        # 检查工具结果缓存（前缀稳定的工具可复用，减少外部 API 调用）
        if self.tool_cache is not None:
            cached = self.tool_cache.get(tool_name, arguments)
            if cached is not None:
                return cached.raw_result

        trace = self.tracer.get_trace(current_trace_id()) if current_trace_id() else None

        async def _do_execute() -> str:
            if self.tool_registry is not None:
                tool = self.tool_registry.get_tool(tool_name)
                if tool is not None:
                    try:
                        result = await self.tool_registry.call(tool_name, arguments)
                        if result.success:
                            return str(result.output)
                        return f"工具执行失败: {result.error}"
                    except Exception as e:
                        return f"工具异常: {e}"

            if self.mcp_client is not None:
                for srv_name in self.mcp_client._servers:
                    prefix = f"{srv_name}_"
                    if tool_name.startswith(prefix):
                        actual_name = tool_name[len(prefix):]
                        try:
                            result = await self.mcp_client.call_tool(srv_name, actual_name, arguments)
                            return str(result)
                        except Exception as e:
                            return f"MCP工具异常: {e}"

            return f"未知工具: {tool_name}"

        if trace is not None:
            async with trace.span(
                "tool.execute",
                attributes={"tool_name": tool_name},
            ) as span:
                result = await _do_execute()
                is_error = result.startswith(("未知工具:", "工具执行失败:", "工具异常:", "MCP工具异常:"))
                span.set_attribute("success", not is_error)
                if is_error:
                    span.set_attribute("error", result[:200])
        else:
            result = await _do_execute()

        if len(result) > _MAX_TOOL_RESULT_LEN:
            result = result[:_MAX_TOOL_RESULT_LEN] + "\n...[结果已截断]"

        # 逻辑时钟追踪: 错误增加重试计数
        if self._timeline is not None and result.startswith(("工具执行失败:", "工具异常:", "MCP工具异常:")):
            self._timeline.inc_retry()
            self._timeline.status = "retry"

        # 存入缓存（仅缓存成功的、可缓存的工具结果）
        if self.tool_cache is not None and not result.startswith(("未知工具:", "工具执行失败:", "工具异常:", "MCP工具异常:")):
            self.tool_cache.put(tool_name, arguments, result)

        return result

    def check_dangerous_tool(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        high_risk_tools = frozenset({"delete_file"})
        if tool_name in high_risk_tools:
            if self.security_mode == "service":
                return False
        return True

    def build_virtual_step_args(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_source_map: dict[str, str],
        mcp_server_map: dict[str, str],
    ) -> tuple[str, dict[str, Any]]:
        from long.ir.plan_ir import ActionType

        source = tool_source_map.get(tool_name, "local")

        if source == "skill":
            return ActionType.CALL_SKILL.value, {
                "skill_name": tool_name,
                "arguments": arguments,
            }
        elif source == "mcp":
            server_name = mcp_server_map.get(tool_name, "")
            parts = tool_name.split("_", 1)
            actual_tool_name = parts[1] if len(parts) == 2 else tool_name
            return ActionType.CALL_MCP.value, {
                "server_name": server_name,
                "tool_name": actual_tool_name,
                "arguments": arguments,
            }
        else:
            if tool_name in ("list_files", "read_file", "read_skill_md"):
                return ActionType.SEARCH.value, {
                    "tool_name": tool_name,
                    "parameters": arguments,
                }
            return ActionType.CALL_TOOL.value, {
                "tool_name": tool_name,
                "parameters": arguments,
            }

    @staticmethod
    def map_tool_to_action(tool_name: str) -> str:
        from long.ir.plan_ir import ActionType

        tool_action_map = {
            "list_files": ActionType.SEARCH.value,
            "read_file": ActionType.SEARCH.value,
            "read_skill_md": ActionType.SEARCH.value,
            "write_file": ActionType.CALL_TOOL.value,
            "delete_file": ActionType.CALL_TOOL.value,
            "execute_code": ActionType.CALL_TOOL.value,
            "execute_file": ActionType.CALL_TOOL.value,
            "query_weather": ActionType.CALL_TOOL.value,
        }

        if tool_name in tool_action_map:
            return tool_action_map[tool_name]

        if "_" in tool_name:
            parts = tool_name.split("_", 1)
            if len(parts) == 2:
                return ActionType.CALL_MCP.value

        return ActionType.CALL_TOOL.value
