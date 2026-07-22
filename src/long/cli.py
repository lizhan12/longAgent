"""CLI 入口模块 — 可控AI智能系统命令行入口"""

import argparse
import asyncio
import logging
import re
import os
import time
from pathlib import Path
from typing import Any
import json
import yaml

logger = logging.getLogger(__name__)


from long.capabilities.mcp_client import MCPClient
from long.capabilities.mcp_server import SystemMCPServer
from long.capabilities.skill_manager import SkillManager, SkillState
from long.capabilities.unified_tool_registry import UnifiedToolRegistry
from long.components.prompt_builder import PromptCache
from long.errors import LLMError, LLMBudgetExceededError, LLMTimeoutError
from long.eval.pipeline import EvalPipeline
from long.interaction.controller import InteractionController
from long.interaction.base import InteractionEvent, InteractionEventType
from long.ir.constraint_validator import ConstraintValidator
from long.ir.executor import PlanExecutor
from long.ir.ir_parser import IRParser
from long.ir.ltl import LTLValidator
from long.ir.state_machine import AgentStateMachine
from long.ir.type_checker import TypeChecker
from long.llm.client import LLMClient
from long.llm.base import LLMConfig
from long.memory.controller import MemoryController
from long.memory.base import MemoryType
from long.observability.tracing import Tracer, current_trace_id, current_span_id
from long.observability.structured_logging import get_logger, setup_structured_logging
from long.optimization.optimizer import AutoOptimizer
from long.optimization.collector import MetricsCollector
from long.sandbox.manager import SandboxManager
from long.session.models import Session
from long.session.store import SessionStore
from long.session.preference import PreferenceStore
from long.session.profile import UserProfile
from long.session.summary import DailySummaryStore
from long.workspace.audit import AuditConfig, WorkspaceAuditHook
from long.workspace.manager import WorkspaceManager


class LongSystem:
    """可控AI智能系统 — 完整系统集成"""

    def __init__(self, config_dir: str | Path = "configs", workspace_root: str | Path = "./workspace") -> None:
        self._config_dir = Path(config_dir)
        self._workspace_root = Path(workspace_root)
        self._configs: dict[str, Any] = {}
        self._initialized = False
        self._auto_eval_interval = 5
        self._conversation_turn_count = 0
        self._current_session_date: str | None = None

        self.workspace: WorkspaceManager | None = None
        self.audit_hook: WorkspaceAuditHook | None = None
        self.llm: LLMClient | None = None
        self.memory: MemoryController | None = None
        self.sandbox: SandboxManager | None = None
        self.state_machine: AgentStateMachine | None = None
        self.ltl_validator: LTLValidator | None = None
        self.type_checker: TypeChecker | None = None
        self.constraint_validator: ConstraintValidator | None = None
        self.ir_parser: IRParser | None = None
        self.tool_registry: UnifiedToolRegistry | None = None
        self.mcp_client: MCPClient | None = None
        self.mcp_server: SystemMCPServer | None = None
        self.skill_manager: SkillManager | None = None
        self.interaction: InteractionController | None = None
        self.eval_pipeline: EvalPipeline | None = None
        self.optimizer: AutoOptimizer | None = None
        self.plan_executor: PlanExecutor | None = None
        self.session_store: SessionStore | None = None
        self.preference_store: PreferenceStore | None = None
        self.summary_store: DailySummaryStore | None = None
        self.user_profile: UserProfile | None = None
        self.active_session: Session | None = None
        self._ws_sessions: dict[str, Session] = {}  # ws_session_id → Session 映射
        self.tracer: Tracer = Tracer()
        self._background_tasks: list[asyncio.Task] = []
        self._max_background_tasks: int = 3
        self._session_dirty: bool = False
        self._session_generated_files: set[str] = set()  # 当前会话生成的文件路径
        self._session_start_ts: float = 0.0  # 会话开始时间戳，用于过滤历史文件
        self._tools_cache: list[dict[str, Any]] | None = None
        self._prompt_cache: PromptCache | None = None

        self.dialog_compressor: Any = None
        self.sandbox_session_manager: Any = None
        self.task_orchestrator: Any = None
        self.subagent_registry: Any = None
        self._security_mode: str = "development"
        self._denied_tools: set[str] = set()
        self._workspace_fs: Any = None

        self.feature_flags: Any = None
        self.prompt_version: Any = None
        self.output_guard: Any = None
        self.alert_manager: Any = None
        self.feedback_loop: Any = None
        self._llm_call_total: int = 0
        self._llm_call_timeout: int = 0
        self._llm_call_fail: int = 0
        self._llm_total_tokens: int = 0
        self._llm_budget_tokens: int = 200000

    def load_configs(self) -> dict[str, Any]:
        """加载所有配置文件"""
        config_files = [
            "workspace", "llm", "ir", "memory", "sandbox", "interaction",
            "eval", "optimization", "mcp", "skills", "security",
        ]
        for name in config_files:
            config_path = self._config_dir / f"{name}.yaml"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    self._configs[name] = yaml.safe_load(f) or {}
            else:
                self._configs[name] = {}
        return self._configs

    def initialize(self) -> None:
        """初始化所有模块（按依赖顺序）"""
        if self._initialized:
            return

        self.load_configs()

        ws_config = self._configs.get("workspace", {})
        root = ws_config.get("root", str(self._workspace_root))
        self.workspace = WorkspaceManager(root=root)

        if ws_config.get("audit", {}).get("enabled", True):
            allowed_paths = [str(self.workspace.root)]
            audit_config = AuditConfig(allowed_paths=allowed_paths)
            self.audit_hook = WorkspaceAuditHook(audit_config)
            self.audit_hook.install()

        llm_config = self._configs.get("llm", {}).get("llm", self._configs.get("llm", {}))
        self.llm = LLMClient(llm_config)
        logger.info("LLM client model: %s", self.llm.config.model)

        self.state_machine = AgentStateMachine()
        self.ltl_validator = LTLValidator()
        self.type_checker = TypeChecker()
        self.constraint_validator = ConstraintValidator(
            state_machine=self.state_machine,
            ltl_validator=self.ltl_validator,
            type_checker=self.type_checker,
        )
        self.ir_parser = IRParser()

        self.tool_registry = UnifiedToolRegistry()
        self.mcp_client = MCPClient()

        self.memory = MemoryController(data_dir=str(self.workspace.data_dir))

        self.mcp_server = SystemMCPServer(
            memory_controller=self.memory,
            config={"traces_dir": str(self.workspace.traces_dir)},
        )
        self.skill_manager = SkillManager(
            config=self._configs.get("skills", {}),
            registry=self.tool_registry,
            skill_dir=str(self.workspace.skills_dir),
        )

        self.sandbox = SandboxManager(workspace_dir=str(self.workspace.sandbox_dir))

        sec_config = self._configs.get("security", {})
        self._security_mode = sec_config.get("mode", "development")
        self._denied_tools = set(
            sec_config.get("service_denied_tools", [])
        ) if self._security_mode == "service" else set()

        from long.sandbox.session_sandbox import (
            SandboxLifecycle,
            SandboxSessionConfig,
            SandboxSessionManager,
        )
        sandbox_lifecycle = sec_config.get("sandbox_lifecycle", "session")
        try:
            lifecycle_enum = SandboxLifecycle(sandbox_lifecycle)
        except ValueError:
            lifecycle_enum = SandboxLifecycle.SESSION
        self.sandbox_session_manager = SandboxSessionManager(
            SandboxSessionConfig(lifecycle=lifecycle_enum)
        )

        from long.context.compressor import CompressConfig, DialogCompressor
        comp_config = sec_config.get("compression", {})
        self.dialog_compressor = DialogCompressor(
            CompressConfig(
                max_prompt_chars=comp_config.get("max_prompt_chars", 12000),
                min_rounds_before_compress=comp_config.get("min_rounds_before_compress", 3),
                keep_recent_rounds=comp_config.get("keep_recent_rounds", 2),
            )
        )

        from long.agent import SubAgentRegistry, TaskOrchestrator
        self.subagent_registry = SubAgentRegistry()
        subagent_config = sec_config.get("subagent", {})
        if subagent_config.get("enabled", True):
            subagents_dir = self.workspace.root / "subagents"
            subagents_dir.mkdir(parents=True, exist_ok=True)
            loaded = self.subagent_registry.load_from_dir(str(subagents_dir))
            if loaded > 0:
                logger.info("加载了 %d 个子 Agent 声明", loaded)

        self.task_orchestrator = TaskOrchestrator(
            max_concurrent=subagent_config.get("max_concurrent", 5)
        )

        from long.agent import EscalationController
        self.escalation = EscalationController(
            max_retries=subagent_config.get("max_retries", 3),
            max_refine_attempts=subagent_config.get("max_refine_attempts", 2),
            hitl_enabled=subagent_config.get("hitl_enabled", True),
        )

        from long.workspace.filesystem import LocalFilesystem
        self._workspace_fs = LocalFilesystem(str(self.workspace.root))

        from long.harness.feature_flag import FeatureFlag, PromptVersion
        flag_config_path = self._config_dir / "feature_flags.yaml"
        self.feature_flags = FeatureFlag.from_yaml(str(flag_config_path))
        self.prompt_version = PromptVersion(str(self.workspace.root), self.feature_flags)

        from long.harness.output_guard import OutputGuard, OutputGuardConfig
        output_pii_enabled = self.feature_flags.is_enabled("output_pii_filter")
        self.output_guard = OutputGuard(OutputGuardConfig(enabled=output_pii_enabled))

        from long.harness.alert import AlertManager
        self.alert_manager = AlertManager(default_rules=True)
        self._llm_budget_tokens = self._configs.get("llm", {}).get("llm", {}).get("budget_tokens", 200000)

        from long.components.tool_cache import ToolResultCache
        cache_dir = self.workspace.root / "cache"
        self._tool_cache = ToolResultCache(
            max_size=200,
            persist_path=cache_dir / "tool_results.json",
            ttl_overrides=self._configs.get("tool_cache_ttl", None),
        )

        from long.harness.feedback_loop import FeedbackLoop
        self.feedback_loop = FeedbackLoop(str(self.workspace.root))

        self.interaction = InteractionController()

        self.eval_pipeline = EvalPipeline()

        # 注册本地文件操作工具
        self._register_local_tools()

        if self.llm is not None:
            from long.eval.process_eval import ProcessEvaluator

            judge_fn = self.llm.get_judge_fn()
            self.eval_pipeline._process_evaluator = ProcessEvaluator(judge_fn=judge_fn)

        self.optimizer = AutoOptimizer(
            collector=MetricsCollector(
                db_path=str(self.workspace.data_dir / "metrics.db")
            ),
        )

        self.plan_executor = PlanExecutor(
            llm=self.llm,
            tool_registry=self.tool_registry,
            mcp_client=self.mcp_client,
            constraint_validator=self.constraint_validator,
            ir_parser=self.ir_parser,
            state_machine=self.state_machine,
        )

        self._auto_discover_skills()

        self._prompt_cache = PromptCache()
        self._prompt_cache.set_static(self._build_static_prompt())

        self._init_session_system()

        async def _skill_caller(skill_name: str = "", tool_name: str = "", arguments: dict[str, Any] | None = None) -> str:
            """执行 Skill 工具函数"""
            if arguments is None:
                arguments = {}
            for skill in self.skill_manager.list_skills():
                if tool_name not in skill.manifest.tools:
                    continue
                if skill.module is None:
                    continue
                func = getattr(skill.module, tool_name, None) or getattr(
                    skill.module, skill.manifest.entry_point, None
                )
                if func is None:
                    continue
                try:
                    import asyncio as _asyncio
                    from inspect import signature

                    sig = signature(func)
                    if _asyncio.iscoroutinefunction(func):
                        if sig.parameters:
                            result = await func(**arguments) if arguments else await func()
                        else:
                            result = await func()
                    else:
                        if sig.parameters:
                            result = func(**arguments) if arguments else func()
                        else:
                            result = func()
                    return str(result)
                except Exception as e:
                    return f"Skill 执行错误: {e}"
            return f"Skill 工具 '{tool_name}' 未找到或未实现"

        self.tool_registry.set_skill_caller(_skill_caller)

        self._init_components()

        self._initialized = True

        if self.llm is not None:
            base_url = self.llm.config.resolve_base_url()
            logger.info(
                "LLM 就绪: model=%s, base_url=%s",
                self.llm.config.model,
                base_url or "(默认)",
            )

    def _init_components(self) -> None:
        """初始化拆分后的组件"""
        from long.components.tool_manager import ToolManager
        from long.components.prompt_builder import PromptBuilder
        from long.components.session_manager import SessionManager
        from long.components.execution_orchestrator import ExecutionOrchestrator
        from long.components.memory_bridge import MemoryBridge

        self._tool_manager = ToolManager(
            workspace=self.workspace,
            tool_registry=self.tool_registry,
            sandbox=self.sandbox,
            sandbox_session_manager=self.sandbox_session_manager,
            skill_manager=self.skill_manager,
            mcp_client=self.mcp_client,
            subagent_registry=self.subagent_registry,
            task_orchestrator=self.task_orchestrator,
            llm=self.llm,
            output_guard=self.output_guard,
            feature_flags=self.feature_flags,
            alert_manager=self.alert_manager,
            workspace_fs=self._workspace_fs,
            config_dir=self._config_dir,
            security_mode=self._security_mode,
            denied_tools=self._denied_tools,
            active_session_getter=lambda: self.active_session,
            tracer=self.tracer,
            configs=self._configs,
            on_llm_stats=lambda resp: self._record_llm_stats(resp),
            on_llm_timeout=lambda: self._record_llm_timeout(),
            on_llm_fail=lambda: self._record_llm_fail(),
            escalation=self.escalation,
            tool_cache=self._tool_cache,
        )

        self._prompt_builder = PromptBuilder(
            workspace=self.workspace,
            memory=self.memory,
            user_profile=self.user_profile,
            skill_manager=self.skill_manager,
            preference_store=self.preference_store,
            feature_flags=self.feature_flags,
            prompt_version=self.prompt_version,
            configs=self._configs,
            active_session=lambda: self.active_session,
            summary_store=self.summary_store,
        )

        self._session_manager = SessionManager(
            workspace=self.workspace,
            session_store=self.session_store,
            preference_store=self.preference_store,
            summary_store=self.summary_store,
            user_profile=self.user_profile,
            memory=self.memory,
            llm=self.llm,
        )

        self._memory_bridge = MemoryBridge(
            output_guard=self.output_guard,
            alert_manager=self.alert_manager,
            eval_pipeline=self.eval_pipeline,
            optimizer=self.optimizer,
            feedback_loop=self.feedback_loop,
            conversation_turn_getter=lambda: (
                self.active_session.recent_messages(limit=6)
                if self.active_session
                else []
            ),
            auto_eval_interval=self._auto_eval_interval,
            feature_flags=self.feature_flags,
            llm_budget_tokens=self._llm_budget_tokens,
        )

        self._execution_orchestrator = ExecutionOrchestrator(
            llm=self.llm,
            tool_manager=self._tool_manager,
            plan_executor=self.plan_executor,
            dialog_compressor=self.dialog_compressor,
            memory=self.memory,
            tracer=self.tracer,
            budget_tokens=self._llm_budget_tokens,
            constraint_validator=self.constraint_validator,
            state_machine=self.state_machine,
            ir_parser=self.ir_parser,
            type_checker=self.type_checker,
            ltl_validator=self.ltl_validator,
            active_session_getter=lambda: self.active_session,
            configs=self._configs,
            prompt_builder_getter=lambda: self._prompt_builder,
            session_manager_getter=lambda: self._session_manager,
            memory_bridge_getter=lambda: self._memory_bridge,
        )

    async def run_cli(self) -> None:
        """启动 CLI 交互模式"""
        if not self._initialized:
            self.initialize()

        assert self.interaction is not None

        from long.interaction.adapters.cli import CLIAdapter

        cli_adapter = CLIAdapter()
        self.interaction.protocol = cli_adapter

        self._register_cli_commands(cli_adapter)

        await self._connect_mcp_servers()

        session = self.interaction.create_session(metadata={"mode": "cli"})
        self.interaction.activate_session(session.session_id)

        try:
            while True:
                user_input = await cli_adapter.receive_input_async()
                if not user_input:
                    continue
                if user_input.strip().lower() in {"exit", "quit", "/exit"}:
                    break
                cmd, args = cli_adapter.parse_command(user_input)
                if cmd is not None:
                    result = cli_adapter.handle_command(cmd, args)
                    if result == "__exit__":
                        break
                else:
                    await self._handle_user_message(cli_adapter, user_input)
        except KeyboardInterrupt:
            pass
        finally:
            self.interaction.end_session(session.session_id)

    async def run_webui(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """启动 WebUI 模式"""
        if not self._initialized:
            self.initialize()

        assert self.interaction is not None

        from long.interaction.adapters.webui import WebUIAdapter

        webui_adapter = WebUIAdapter(host=host, port=port)
        webui_adapter.set_skill_manager(self.skill_manager)
        webui_adapter._tracer = self.tracer  # 注入 tracer 引用
        self.interaction.protocol = webui_adapter

        # 注册命令（和 CLI 模式相同）
        self._register_cli_commands(webui_adapter)

        await self._connect_mcp_servers()

        session = self.interaction.create_session(metadata={"mode": "webui"})
        self.interaction.activate_session(session.session_id)

        # 启动服务器（在后台任务中运行）
        server_task = asyncio.create_task(webui_adapter.start_server())

        # 等待服务器启动
        await asyncio.sleep(1)
        actual_port = webui_adapter.port
        webui_adapter.console.print(f"WebUI 已启动: http://{host}:{actual_port}")

        try:
            # 消息处理主循环（和 CLI 模式相同）
            while webui_adapter._active:
                ws_session_id, user_input = await webui_adapter.receive_input_async(timeout=1.0)
                if not user_input:
                    continue

                if user_input.strip().lower() in {"exit", "quit", "/exit"}:
                    break

                cmd, args = webui_adapter.parse_command(user_input)
                if cmd is not None:
                    result = webui_adapter.handle_command(cmd, args)
                    if result == "__exit__":
                        break
                else:
                    try:
                        await self._handle_user_message(webui_adapter, user_input, ws_session_id=ws_session_id)
                    except Exception as e:
                        logger.error("WebUI message handling error: %s", e)
                        webui_adapter.console.print(f"[red]处理消息时出错: {e}[/red]")
        except KeyboardInterrupt:
            pass
        finally:
            webui_adapter._active = False
            await webui_adapter.stop_server()
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
            self.interaction.end_session(session.session_id)

    def shutdown(self) -> None:
        """关闭系统"""
        self._save_session()

        if self.sandbox_session_manager is not None:
            try:
                import asyncio as _asyncio
                try:
                    loop = _asyncio.get_running_loop()
                    loop.create_task(self.sandbox_session_manager.destroy_all())
                except RuntimeError:
                    _asyncio.get_event_loop().run_until_complete(
                        self.sandbox_session_manager.destroy_all()
                    )
            except Exception:
                pass

        if self.audit_hook is not None:
            self.audit_hook.uninstall()
        self._initialized = False

    def _init_session_system(self) -> None:
        """初始化会话系统：SessionStore + PreferenceStore + DailySummaryStore"""
        if self.workspace is None:
            return

        data_dir = self.workspace.data_dir
        self.session_store = SessionStore(data_dir)
        self.preference_store = PreferenceStore(data_dir)
        self.summary_store = DailySummaryStore(data_dir, self.session_store)
        self.user_profile = UserProfile(data_dir)

        latest = self.session_store.load_latest_session()
        if latest is not None:
            self.active_session = latest
            self._current_session_date = latest.date_str
            logger.info("恢复会话: %s (%d 条消息)", latest.id, latest.message_count)
        else:
            self.active_session = Session()
            self._current_session_date = self.active_session.date_str
            logger.info("创建新会话: %s", self.active_session.id)

        prefs = self.preference_store.get_all()
        if prefs:
            logger.info("已加载 %d 条用户偏好", len(prefs))

        pending = self.summary_store.check_pending()
        if pending:
            logger.info("待补生成摘要的日期: %s", pending)

    def _ensure_session(self) -> Session:
        """确保有活跃会话，检测日期变更"""
        today = Session().date_str
        if self.active_session is None or self._current_session_date != today:
            if self.active_session is not None and self.session_store is not None:
                self.session_store.save(self.active_session)
                if self.summary_store is not None:
                    try:
                        import asyncio
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.ensure_future(
                                self._daily_summary_and_profile(self._current_session_date)
                            )
                        else:
                            loop.run_until_complete(
                                self._daily_summary_and_profile(self._current_session_date)
                            )
                    except Exception as e:
                        logger.warning("日终处理失败: %s", e)

            self.active_session = Session()
            self._current_session_date = today
            logger.info("日期变更，创建新会话: %s", self.active_session.id)

        return self.active_session

    async def _daily_summary_and_profile(self, date_str: str) -> None:
        """日终处理：生成摘要 + 更新用户画像"""
        if self.summary_store is None:
            return

        summary = await self.summary_store.summarize_day(date_str, self.llm)

        if summary and self.user_profile is not None:
            recent_summaries = [s for _, s in self.summary_store.get_recent(days=7)]
            await self.user_profile.extract_from_summaries(recent_summaries, self.llm)

    def _save_session(self) -> None:
        """持久化当前会话"""
        if self.active_session is not None and self.session_store is not None:
            self.session_store.save(self.active_session)
            self._session_dirty = False

        # 异步同步记忆到文件（非阻塞）
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.create_task(self._sync_memory_to_md())
        except RuntimeError:
            pass

    async def _sync_memory_to_md(self) -> None:
        """将长期记忆同步写入 MEMORY.md + 每日日志（时间序列结构）

        文件结构:
        - MEMORY.md: 索引文件（用户偏好 + 最近7天摘要 + 历史摘要）
        - memory/YYYY-MM-DD.md: 每日详细记忆日志
        """
        if self.workspace is None:
            return

        from datetime import datetime, timezone, timedelta

        tz_cn = timezone(timedelta(hours=8))
        now = datetime.now(tz_cn)
        today_str = now.strftime("%Y-%m-%d")

        # --- 写入每日日志 ---
        memory_dir = self.workspace.root / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        daily_path = memory_dir / f"{today_str}.md"

        daily_lines: list[str] = []
        if daily_path.exists():
            try:
                daily_lines = daily_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                pass

        # 追加新记忆到今日日志
        seen_daily: set[str] = set(l.lstrip("- ").strip() for l in daily_lines if l.startswith("- "))
        new_entries: list[str] = []
        if self.memory is not None:
            try:
                all_items = await self.memory.search("", limit=30)
                for item in all_items:
                    content = getattr(item, "content", str(item))[:200]
                    content_norm = content.strip()
                    if not content_norm or content_norm in seen_daily:
                        continue
                    seen_daily.add(content_norm)
                    new_entries.append(f"- {content_norm}")
            except Exception:
                pass

        if new_entries:
            if not daily_lines:
                daily_lines = [f"# 记忆日志 — {today_str}", ""]
            daily_lines.extend(new_entries)
            try:
                daily_path.write_text("\n".join(daily_lines) + "\n", encoding="utf-8")
            except Exception:
                pass

        # --- 写入 MEMORY.md 索引 ---
        memory_path = self.workspace.root / "MEMORY.md"
        lines: list[str] = [
            "# MEMORY.md — Agent 长期记忆索引",
            "",
            "此文件由系统自动维护。详细记忆按日期存储在 memory/ 目录下。",
            "",
            "---",
        ]

        # 用户偏好
        if self.preference_store is not None:
            try:
                prefs = self.preference_store.get_all()
                if prefs:
                    lines.append("")
                    lines.append("## 用户偏好")
                    for k, v in prefs.items():
                        lines.append(f"- {k}: {v}")
            except Exception:
                pass

        # 最近7天记忆摘要
        lines.append("")
        lines.append("## 最近记忆")
        retain_days = 7
        for day_offset in range(retain_days):
            day = now - timedelta(days=day_offset)
            day_str = day.strftime("%Y-%m-%d")
            day_path = memory_dir / f"{day_str}.md"
            if not day_path.exists():
                continue
            try:
                day_content = day_path.read_text(encoding="utf-8").strip()
                day_lines = [l for l in day_content.splitlines() if l.startswith("- ")]
                if day_lines:
                    lines.append(f"### {day_str}")
                    # 最多显示5条摘要
                    for entry in day_lines[:5]:
                        lines.append(entry[:120])
                    if len(day_lines) > 5:
                        lines.append(f"- ... (共{len(day_lines)}条，详见 memory/{day_str}.md)")
            except Exception:
                pass

        # 历史摘要（7天前的记忆合并）
        lines.append("")
        lines.append("## 历史摘要")
        older_entries: list[str] = []
        try:
            for day_file in sorted(memory_dir.glob("*.md"), reverse=True):
                day_str = day_file.stem
                try:
                    day_date = datetime.strptime(day_str, "%Y-%m-%d").replace(tzinfo=tz_cn)
                    if (now - day_date).days < retain_days:
                        continue
                except ValueError:
                    continue
                day_content = day_file.read_text(encoding="utf-8").strip()
                day_lines = [l for l in day_content.splitlines() if l.startswith("- ")]
                older_entries.extend(day_lines[:3])
        except Exception:
            pass

        if older_entries:
            for entry in older_entries[:10]:
                lines.append(entry[:120])
        else:
            lines.append("(暂无)")

        try:
            content_text = "\n".join(lines) + "\n"
            memory_path.write_text(content_text, encoding="utf-8")
        except Exception:
            pass

    def _mark_session_dirty(self) -> None:
        """标记会话为脏（需要持久化）"""
        self._session_dirty = True

    def _save_session_if_dirty(self) -> None:
        """仅在会话有变更时持久化"""
        if self._session_dirty:
            self._save_session()

    def _submit_background_task(self, coro: Any) -> None:
        """提交后台任务（并发上限3个）"""
        self._background_tasks = [t for t in self._background_tasks if not t.done()]
        if len(self._background_tasks) >= self._max_background_tasks:
            logger.debug("后台任务已达上限 %d，跳过", self._max_background_tasks)
            return
        task = asyncio.ensure_future(coro)
        task.add_done_callback(self._on_background_task_done)
        self._background_tasks.append(task)

    def _on_background_task_done(self, task: asyncio.Task) -> None:
        """后台任务完成回调"""
        if task.exception():
            logger.warning("后台任务异常: %s", task.exception())

    async def _auto_eval_conversation(self) -> None:
        """每隔若干轮对话自动运行轻量级评估，记录到优化器"""
        self._conversation_turn_count += 1
        if self._conversation_turn_count % self._auto_eval_interval != 0:
            return
        if self.eval_pipeline is None or self.optimizer is None:
            return
        if self.active_session is None:
            return

        recent = self.active_session.recent_messages(limit=6)
        user_msgs = [m for m in recent if m["role"] == "user"]
        assistant_msgs = [m for m in recent if m["role"] == "assistant"]

        if not user_msgs or not assistant_msgs:
            return

        try:
            from long.eval.report import EvalTask

            last_user = user_msgs[-1]["content"]
            last_assistant = assistant_msgs[-1]["content"]

            task = EvalTask(
                name=f"auto_eval_turn_{self._conversation_turn_count}",
                input=last_user,
                expected=None,
            )

            report = self.eval_pipeline.run(task, output=last_assistant)

            self.optimizer.collector.record_eval_result(
                task_name=task.name,
                score=report.score,
                category=task.category.value,
            )
            self.optimizer.collector.record_execution_metrics(
                step_count=len(recent),
                duration=0.0,
                success=report.score >= 0.5,
            )

            logger.info(
                "自动评估完成: turn=%d, score=%.2f, needs_review=%s",
                self._conversation_turn_count,
                report.score,
                report.needs_human_review,
            )

            if self.optimizer is not None:
                self.optimizer.on_conversation_complete()

            if self.feature_flags.is_enabled("auto_eval_feedback") and report.score < 0.7:
                self.feedback_loop.generate_from_eval({
                    "scores": {"efficiency": report.score},
                    "turn": self._conversation_turn_count,
                })

        except Exception as e:
            logger.warning("自动评估失败: %s", e)

    def _schedule_auto_eval(self) -> None:
        """将自动评估提交为后台任务"""
        self._submit_background_task(self._auto_eval_conversation())

    def _record_llm_stats(self, response: Any) -> None:
        """记录 LLM 调用统计（用于告警和监控）"""
        self._llm_call_total += 1
        if response.usage:
            self._llm_total_tokens += getattr(response.usage, "total_tokens", 0) or 0

    def _record_llm_timeout(self) -> None:
        self._llm_call_timeout += 1
        self._llm_call_total += 1

    def _record_llm_fail(self) -> None:
        self._llm_call_fail += 1
        self._llm_call_total += 1

    def _check_output_safety(self, text: str) -> None:
        """检查输出安全（PII过滤 + 敏感词）"""
        if self.output_guard is None or not self.feature_flags.is_enabled("output_pii_filter"):
            return
        if not text:
            return
        result = self.output_guard.scan(text)
        if result.pii_matches:
            self.alert_manager.trigger(
                "pii_detected",
                f"LLM 输出包含 PII: {result.warning}",
            )
        if result.sensitive_matches:
            self.alert_manager.trigger(
                "sensitive_word",
                f"LLM 输出包含敏感词: {result.warning}",
            )

    def _get_or_create_ws_session(self, ws_session_id: str) -> Session:
        """根据 ws_session_id 获取或创建对应的 Session

        WebUI 模式下，每个 WebSocket 连接有独立的 session_id，
        对话历史按 ws_session_id 隔离，避免跨会话的上下文污染。

        CLI 模式下 ws_session_id 为空，使用默认的 active_session。
        """
        if not ws_session_id:
            # CLI 模式：使用默认的 active_session
            return self._ensure_session()

        if ws_session_id in self._ws_sessions:
            session = self._ws_sessions[ws_session_id]
            self.active_session = session  # 确保 active_session 同步
            # 检测日期变更
            today = Session().date_str
            if session.date_str != today:
                # 保存旧 session
                if self.session_store is not None:
                    self.session_store.save(session)
                # 创建新 session
                new_session = Session()
                self._ws_sessions[ws_session_id] = new_session
                logger.info("WS会话 %s 日期变更，创建新会话: %s", ws_session_id, new_session.id)
                return new_session
            return session

        # 创建新的 session
        session = Session()
        self._ws_sessions[ws_session_id] = session
        self.active_session = session  # 兼容：同时更新 active_session
        logger.info("WS会话 %s 创建新 Session: %s", ws_session_id, session.id)
        return session

    def _trim_history_by_topic(
        self,
        messages: list[dict[str, str]],
        current_user_msg: str,
        max_messages: int = 20,
    ) -> list[dict[str, str]]:
        """根据话题相关性裁剪历史消息

        当用户的新消息与最近历史话题明显不同时，截断不相关的旧历史，
        防止跨话题的上下文污染（如天气对话混入AI报告对话）。

        策略：
        1. 从后往前找到最近的用户消息边界
        2. 如果当前消息与最近一轮对话话题不同，只保留最近一轮
        3. 否则保留最多 max_messages 条消息
        """
        if not messages:
            return messages

        # 如果消息总数不超过 max_messages，直接返回
        if len(messages) <= max_messages:
            return messages

        # 找到所有 user 消息的位置（从后往前）
        user_msg_indices = []
        for i, m in enumerate(messages):
            if m.get("role") == "user":
                user_msg_indices.append(i)

        if not user_msg_indices:
            return messages[-max_messages:]

        # 检测话题切换：当前消息与最近一轮对话的话题是否不同
        last_user_idx = user_msg_indices[-1]
        last_user_content = messages[last_user_idx].get("content", "")

        if self._is_topic_switch(current_user_msg, last_user_content):
            # 话题切换：只保留最近一轮对话（最后一对 user-assistant）
            # 从最后一个 user 消息开始截取
            trimmed = messages[last_user_idx:]
            logger.info(
                "话题切换检测: 当前='%s' vs 历史='%s'，截断历史至最近 %d 条消息",
                current_user_msg[:30], last_user_content[:30], len(trimmed),
            )
            return trimmed

        # 同一话题：保留最近 max_messages 条
        return messages[-max_messages:]

    @staticmethod
    def _is_topic_switch(current_msg: str, last_msg: str) -> bool:
        """检测两条用户消息之间是否存在话题切换

        使用核心词匹配判断：提取每条消息中的核心名词/动词，
        如果核心词重叠度很低，认为是话题切换。
        """
        if not current_msg or not last_msg:
            return False

        # 完全相同的消息不算切换
        if current_msg.strip() == last_msg.strip():
            return False

        # 核心领域词词典：这些词出现则代表特定话题
        _TOPIC_WORDS = {
            # 天气
            "天气", "气温", "温度", "下雨", "晴天", "阴天", "多云", "风力",
            "预报", "降雨", "降雪", "湿度", "气象",
            # 排序算法
            "排序", "冒泡", "快速排序", "归并排序", "桶排序", "树排序",
            "堆排序", "插入排序", "选择排序", "算法",
            # AI/科技
            "人工智能", "AI", "大模型", "机器学习", "深度学习", "神经网络",
            "智能体", "Agent", "GPT", "LLM", "机器人",
            # 报告/PPT
            "报告", "PPT", "pptx", "演示", "幻灯片", "汇报",
            # 编程
            "代码", "编程", "函数", "程序", "Python", "JavaScript",
            # 数据
            "数据", "图表", "可视化", "折线图", "柱状图",
            # 地理
            "杭州", "北京", "上海", "深圳", "阜阳", "临泉", "余杭",
        }

        def _extract_topic_words(text: str) -> set[str]:
            """从文本中提取匹配的核心话题词（大小写不敏感）"""
            text_lower = text.lower()
            found = set()
            for word in _TOPIC_WORDS:
                if word.lower() in text_lower:
                    found.add(word)
            return found

        current_topics = _extract_topic_words(current_msg)
        last_topics = _extract_topic_words(last_msg)

        # 如果两条消息都没有匹配到话题词，用简单的字符重叠判断
        if not current_topics and not last_topics:
            # 简单判断：如果两条消息没有任何2字以上的共同子串
            common = set()
            for i in range(len(current_msg) - 1):
                sub = current_msg[i:i+2]
                if sub in last_msg and sub.strip():
                    common.add(sub)
            return len(common) == 0

        # 如果只有一条消息匹配到话题词，另一条没有，不判断为切换
        if not current_topics or not last_topics:
            return False

        # 两条消息都有话题词：检查是否有交集
        intersection = current_topics & last_topics
        if intersection:
            return False  # 有共同话题词，不是切换

        # 话题词完全不重叠 → 话题切换
        return True

    def _send_trace_to_frontend(self, cli_adapter: Any) -> None:
        """发送最新 Trace 数据到前端（仅 WebUI 适配器）"""
        from long.interaction.adapters.webui import WebUIAdapter
        if not isinstance(cli_adapter, WebUIAdapter):
            return

        traces = self.tracer.get_traces(limit=1)
        if not traces:
            return

        trace = traces[0]
        # 确保 trace 的 end_time 已设置
        if trace.end_time is None:
            trace.finish()

        trace_data = trace.to_dict()
        # 如果没有 span，不发送（避免空 trace）
        if not trace_data.get("spans"):
            return

        cli_adapter.send_event(InteractionEvent(
            type=InteractionEventType.TRACE,
            content="",
            metadata={"trace": trace_data},
        ))

    def _notify_turn_complete(self, cli_adapter: Any) -> None:
        """通知前端本轮对话已结束（仅对 WebUI 适配器生效）"""
        from long.interaction.adapters.webui import WebUIAdapter
        if isinstance(cli_adapter, WebUIAdapter):
            cli_adapter.send_event(InteractionEvent(
                type=InteractionEventType.TURN_COMPLETE,
                content="",
                metadata={},
            ))

    def _check_alerts(self) -> None:
        """在每轮对话后检查告警条件"""
        if self.alert_manager is None:
            return
        self.alert_manager.collect_metrics_alert(
            llm_call_total=self._llm_call_total,
            llm_call_timeout=self._llm_call_timeout,
            total_tokens=self._llm_total_tokens,
            budget_tokens=self._llm_budget_tokens,
        )
        if self._llm_call_fail > 0:
            self.alert_manager.check("consecutive_failures", float(self._llm_call_fail))

    def _register_local_tools(self) -> None:
        """注册本地内置工具（文件操作等）到工具注册表"""
        if self.workspace is None:
            return

        ws = self.workspace
        fs = self._workspace_fs

        async def _list_files(path: str = "", **kwargs: Any) -> str:
            try:
                if fs is not None:
                    entries = await fs.list_dir(path)
                    if not entries:
                        return f"(空目录: {path or '.'})"
                    lines: list[str] = []
                    for e in entries:
                        entry_path = f"{path}/{e}" if path else e
                        is_dir = await fs.is_dir(entry_path)
                        prefix = "[D]" if is_dir else "[F]"
                        lines.append(f"{prefix} {entry_path}")
                    return "\n".join(lines)
                entries = ws.list_files(path)
                if not entries:
                    return f"(空目录: {path or '.'})"
                lines: list[str] = []
                for e in entries:
                    rel = str(e.relative_to(ws.root))
                    prefix = "[D]" if e.is_dir() else "[F]"
                    lines.append(f"{prefix} {rel}")
                return "\n".join(lines)
            except Exception as e:
                return f"list_files 错误: {e}"

        def _resolve_path(path: str) -> str:
            _workspace_prefixes = ("output/", "skills/", "data/", "configs/", "logs/", "traces/", "cache/", "sandbox/", "temp/")
            if not any(path.startswith(p) for p in _workspace_prefixes):
                path = f"output/{path}"

            from pathlib import Path
            resolved = (ws.root / path).resolve()
            if not str(resolved).startswith(str(ws.root.resolve())):
                raise PermissionError(f"路径超出工作区边界: {path}")
            return str(resolved.relative_to(ws.root))

        async def _read_file(path: str = "", **kwargs: Any) -> str:
            try:
                resolved = _resolve_path(path)
                if fs is not None:
                    try:
                        return await fs.read(resolved)
                    except Exception:
                        return await fs.read(path)
                try:
                    return str(ws.read_file(resolved))
                except FileNotFoundError:
                    return str(ws.read_file(path))
            except Exception as e:
                return f"read_file 错误: {e}"

        async def _write_file(path: str = "", content: str = "", **kwargs: Any) -> str:
            try:
                if not path or not path.strip():
                    return "write_file 错误: path 参数不能为空，必须指定文件名（如 'output/report.py'）"
                if path.endswith("/"):
                    return f"write_file 错误: path 不能是目录，必须包含文件名（如 '{path}report.py'）"
                if not content or not content.strip():
                    return f"write_file 错误: content 参数不能为空，必须包含实际的文件内容（代码、文本等），不能只写描述"

                resolved = _resolve_path(path)

                # 对 .py 文件做内容质量预检：检测是否用描述文字代替了代码
                if resolved.endswith(".py") and content:
                    content_stripped = content.lstrip("# \n\r\t")
                    has_code = bool(re.search(r'(?:^|\n)\s*(?:from\s+\w+\s+import|import\s+\w+|def\s+\w+\s*\(|class\s+\w+|\w+\s*=\s*)', content))
                    is_description = content_stripped.startswith(("根据", "基于", "将根据", "代码将", "编写", "# 根据")) and not has_code
                    if is_description:
                        logger.warning("_write_file: %s 内容疑似描述文字而非代码，拒绝写入", resolved)
                        return (
                            f"write_file 错误: content 参数包含的是描述文字而非可执行代码。\n"
                            f"你写的是: '{content[:100]}...'\n"
                            f"正确做法: content 必须包含完整可执行的 Python 代码（以 import/from/def/class 开头），"
                            f"不能写描述性文字。请重新调用 write_file，在 content 中写入实际的 Python 代码。"
                        )

                if self.output_guard is not None and self.feature_flags.is_enabled("output_pii_filter"):
                    guard_result = self.output_guard.scan(content)
                    if guard_result.pii_matches:
                        self.alert_manager.trigger(
                            "pii_detected", f"write_file 写入内容包含 PII: {guard_result.warning}",
                        )

                if fs is not None:
                    await fs.write(resolved, content)
                else:
                    ws.write_file(resolved, content)
                return f"写入成功: {resolved}"
            except Exception as e:
                return f"write_file 错误: {e}"

        async def _delete_file(path: str = "", **kwargs: Any) -> str:
            try:
                resolved = _resolve_path(path)
                if fs is not None:
                    if await fs.exists(resolved):
                        await fs.delete(resolved)
                        return f"删除成功: {path}"
                    if await fs.exists(path):
                        await fs.delete(path)
                        return f"删除成功: {path}"
                    return f"文件不存在: {path}"
                if ws.delete_file(resolved) or ws.delete_file(path):
                    return f"删除成功: {path}"
                return f"文件不存在: {path}"
            except Exception as e:
                return f"delete_file 错误: {e}"

        async def _read_skill_md(skill_name: str = "", **kwargs: Any) -> str:
            """读取指定 skill 的 SKILL.md 文档内容"""
            if not skill_name:
                if self.skill_manager is None:
                    return "Skill 管理器未初始化"
                skills = self.skill_manager.list_skills()
                available = [s.manifest.name for s in skills]
                return f"可用 skills: {', '.join(available)}"

            if self.skill_manager is None:
                return f"Skill '{skill_name}' 未找到（Skill 管理器未初始化）"

            skill_record = self.skill_manager.get_skill(skill_name)
            if skill_record is None:
                _fallback_hints = {
                    "docx": "你可以直接使用 execute_code 工具执行 python-docx 代码来创建/编辑 Word 文档。先 pip install python-docx，然后用 from docx import Document 编写代码。",
                    "xlsx": "你可以直接使用 execute_code 工具执行 openpyxl 代码来创建/编辑 Excel 文件。",
                    "pptx": "你可以直接使用 execute_code 工具执行 python-pptx 代码来创建/编辑 PPT 文件。",
                    "pdf": "你可以直接使用 execute_code 工具执行 reportlab/fpdf 代码来创建/编辑 PDF 文件。",
                }
                _hint = _fallback_hints.get(skill_name, "你可以直接使用 execute_code 工具编写 Python 代码来完成任务。")
                return f"Skill '{skill_name}' 未找到。{_hint}"

            skill_path = Path(skill_record.path)
            skill_md = skill_path / "SKILL.md"
            if not skill_md.exists():
                scripts_dir = skill_path / "scripts"
                if scripts_dir.exists():
                    return (
                        f"Skill '{skill_name}' 的 SKILL.md 不存在。"
                        f"但该 skill 有 scripts/ 目录，包含以下脚本：\n"
                        + "\n".join(f"  - {p.name}" for p in sorted(scripts_dir.rglob("*.py")))
                    )
                return f"Skill '{skill_name}' 既没有 SKILL.md 也没有 scripts/ 目录"

            try:
                return skill_md.read_text(encoding="utf-8")
            except Exception as e:
                return f"读取 SKILL.md 失败: {e}"

        async def _execute_code(code: str = "", language: str = "python", args: str = "", **kwargs: Any) -> str:
            """在沙箱中执行代码并返回结果"""
            if self.sandbox is None:
                return "沙箱未初始化，无法执行代码"

            if not code.strip():
                return "代码内容为空"

            from long.sandbox.base import ExecutionSpec, ResourceLimits

            sandbox_env: dict[str, str] = {}
            for key in ("TAVILY_API_KEY", "LLM_API_KEY", "LLM_BASE_URL", "QWEATHER_PRIVATE_KEY", "QWEATHER_API_HOST", "KEY_ID"):
                val = os.environ.get(key, "")
                if val:
                    sandbox_env[key] = val

            parsed_args: list[str] = []
            if args.strip():
                parsed_args = args.strip().split()

            spec = ExecutionSpec(
                code=code,
                language=language,
                timeout=120.0,
                args=parsed_args,
                env=sandbox_env,
                working_dir=str(self.workspace.root),
                resource_limits=ResourceLimits(network=True),
            )

            try:
                session_id = getattr(self.active_session, "id", "") if self.active_session else ""
                if self.sandbox_session_manager is not None and session_id:
                    sb = await self.sandbox_session_manager.get_or_create(session_id)
                    result, _sandbox_id = await self.sandbox.execute_with_session(
                        spec, session_id, sandbox_lifecycle=sb,
                    )
                else:
                    result = await self.sandbox.execute(spec)

                if result.status.value == "success":
                    output_parts: list[str] = []
                    if result.stdout:
                        output_parts.append(f"输出:\n{result.stdout}")
                    if result.stderr:
                        filtered_lines = []
                        for line in result.stderr.splitlines():
                            if any(skip in line for skip in (
                                "matplotlib", "MPLCONFIGDIR", "fontTools", "PIL",
                                "UserWarning", "FutureWarning", "DeprecationWarning",
                            )):
                                continue
                            filtered_lines.append(line)
                        filtered_stderr = "\n".join(filtered_lines).strip()
                        if filtered_stderr:
                            output_parts.append(f"警告:\n{filtered_stderr}")
                    output_parts.append(f"执行成功 (耗时: {result.duration:.2f}s)")
                    return "\n".join(output_parts) if output_parts else "执行成功（无输出）"

                error_parts: list[str] = [f"❌ 执行失败: {result.status.value}"]
                if result.stdout:
                    error_parts.append(f"标准输出:\n{result.stdout}")
                if result.stderr:
                    error_parts.append(f"错误信息:\n{result.stderr}")
                if result.error:
                    error_parts.append(f"错误详情: {result.error}")
                error_parts.append(f"退出码: {result.exit_code}")
                error_parts.append(
                    "请根据上述错误信息修复代码，然后重新使用 write_file 写入修复后的代码，"
                    "再使用 execute_code 或 execute_file 执行。"
                )
                return "\n".join(error_parts)

            except Exception as e:
                return f"执行异常: {e}\n请修复代码后重试。"

        async def _execute_file(path: str = "", args: str = "", **kwargs: Any) -> str:
            """执行工作区中的代码文件"""
            if self.sandbox is None:
                return "沙箱未初始化，无法执行代码"

            if not path:
                return "文件路径不能为空"

            try:
                resolved = _resolve_path(path)
                try:
                    content = ws.read_file(resolved)
                except FileNotFoundError:
                    content = ws.read_file(path)
                if isinstance(content, bytes):
                    return "不支持执行二进制文件"

                language = "python"
                if path.endswith(".js"):
                    language = "javascript"
                elif path.endswith(".sh") or path.endswith(".bash"):
                    language = "bash"

                result_text = await _execute_code(code=content, language=language, args=args)
                if "执行失败" in result_text or "执行异常" in result_text:
                    result_text += f"\n文件路径: {path}"
                return result_text
            except Exception as e:
                return f"执行文件失败: {e}\n请检查文件路径是否正确。"

        async def _get_current_time(**kwargs: Any) -> str:
            from datetime import datetime

            now = datetime.now()
            weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
            weekday = weekdays[now.weekday()]
            return (
                f"当前时间: {now.strftime('%Y年%m月%d日')} {weekday} "
                f"{now.strftime('%H:%M:%S')}"
            )

        async def _tavily_search(query: str = "", max_results: int = 5, search_depth: str = "basic", **kwargs: Any) -> str:
            # 类型容错：LLM 可能传字符串
            try:
                max_results = int(max_results)
            except (ValueError, TypeError):
                max_results = 5
            script_path = self.workspace.root / "skills" / "tavily-search" / "scripts" / "tavily_search.py"
            if not script_path.exists():
                script_path = self._config_dir.parent.resolve() / "skills" / "tavily-search" / "scripts" / "tavily_search.py"
            if not script_path.exists():
                return f"Tavily 搜索脚本未找到"

            env = dict(os.environ)
            tavily_key = os.environ.get("TAVILY_API_KEY", "")
            if not tavily_key:
                return "TAVILY_API_KEY 未配置"

            max_results = min(max(1, max_results), 5)
            search_depth = search_depth or "basic"

            import sys as _sys

            async def _run_search(q: str, max_res: int, depth: str) -> str:
                cmd = [
                    _sys.executable,
                    str(script_path),
                    "--query", q,
                    "--max-results", str(max(max_res, 1)),
                    "--search-depth", depth,
                    "--format", "md",
                ]
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=env,
                        cwd=str(self.workspace.root),
                    )
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=20,
                    )
                    if proc.returncode != 0:
                        return f"搜索执行失败 (exit={proc.returncode}): {stderr.decode().strip()}"
                    return stdout.decode().strip() or "搜索完成，但无结果返回"
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    return "搜索超时（20秒）"

            output = await _run_search(query, max_results, search_depth)

            if output.startswith("搜索执行失败") or output.startswith("搜索超时"):
                if search_depth == "advanced":
                    logger.info("Tavily 搜索失败，降级为 basic 模式重试")
                    output = await _run_search(query, max_results, "basic")
                if output.startswith("搜索执行失败") and max_results > 3:
                    logger.info("Tavily 搜索仍失败，减少结果数重试")
                    output = await _run_search(query, 3, "basic")

            return output

        async def _query_weather(city: str = "", **kwargs: Any) -> str:
            if not city:
                return "城市名不能为空，请提供城市名（如：杭州、北京）"
            result = await _execute_file(path="skills/qweather/query_weather.py", args=city)
            if not result.startswith(("执行失败", "执行异常", "文件路径")):
                cities_list = [c.strip() for c in city.replace("，", ",").replace("、", ",").split(",") if c.strip()]
                if len(cities_list) > 1:
                    result += f"\n\n[提示] 以上已包含 {', '.join(cities_list)} 共{len(cities_list)}个城市的完整天气数据，请直接使用以上数据回答，不要再调用 query_weather 重复查询。"
                else:
                    result += "\n\n[提示] 以上已包含该城市的完整天气数据，请直接使用以上数据回答，不要再调用 query_weather 重复查询。"
            return result

        from long.capabilities.unified_tool_registry import ToolDefinition, ToolSource

        tools_config = [
            ("list_files", "列出工作区目录下的文件和子目录", _list_files),
            ("read_file", "读取工作区中指定文件的内容（纯文本文件）", _read_file),
            ("write_file", "将内容写入工作区中的文件（会自动创建父目录）", _write_file),
            ("delete_file", "删除工作区中的文件或目录", _delete_file),
            ("read_skill_md", "读取指定 skill 的 SKILL.md 文档内容（当需要使用某个 skill 时先读取其文档）", _read_skill_md),
            ("execute_code", "在沙箱中执行代码并返回结果（支持 Python/JavaScript/Bash）", _execute_code),
            ("execute_file", "执行工作区中已有的代码文件（如 output/ 目录下的 .py 文件）", _execute_file),
            ("get_current_time", "获取当前日期和时间（当需要知道今天日期、当前时间时使用此工具）", _get_current_time),
            ("query_weather", "查询指定城市的实时天气和7天预报。⚠️ 天气查询必须使用此工具，禁止使用 tavily_search 查天气。参数: city(城市名，多城市用英文逗号分隔，如'北京,上海')", _query_weather),
            ("tavily_search", "使用 Tavily API 搜索网络获取最新信息。⚠️ 禁止用于天气查询（请用 query_weather）。严格限制：每个主题最多搜索2次，系统会拦截超过限制的调用。", _tavily_search),
        ]

        for name, desc, handler in tools_config:
            if name in self._denied_tools:
                logger.info("安全模式 [%s]: 工具 %s 已禁用", self._security_mode, name)
                continue
            tool_def = ToolDefinition(
                name=name,
                description=desc,
                source=ToolSource.LOCAL,
            )
            self.tool_registry.register_local(tool_def, handler)

        self._register_subagent_tools()

    def _register_subagent_tools(self) -> None:
        """注册子 Agent 编排工具（Harness Engineering: 声明式子任务）"""
        if self.subagent_registry is None or self.task_orchestrator is None:
            return

        from long.capabilities.unified_tool_registry import ToolDefinition, ToolSource

        subagent_names = self.subagent_registry.list_names()
        subagent_desc = (
            f"可用的子 Agent: {', '.join(subagent_names)}。每个子 Agent 可独立处理子任务。"
            if subagent_names
            else "当前无可用子 Agent。可在 workspace/subagents/ 目录下创建声明文件。"
        )

        async def _delegate_task(
            sub_agent_name: str = "",
            instruction: str = "",
            timeout: float = 120.0,
        ) -> str:
            sub_agent_name = sub_agent_name.strip()
            instruction = instruction.strip()
            if not sub_agent_name or not instruction:
                return "错误: 必须指定 sub_agent_name 和 instruction"
            spec = self.subagent_registry.get(sub_agent_name)
            if spec is None:
                return f"错误: 子 Agent '{sub_agent_name}' 未找到。可用: {', '.join(subagent_names)}"

            session_id = getattr(self.active_session, "id", "") if self.active_session else ""
            task = self.task_orchestrator.submit(
                sub_agent_name=sub_agent_name,
                instruction=instruction,
                timeout=timeout,
                parent_session_id=session_id,
            )

            async def _exec(tsk: Any) -> str:
                spec = self.subagent_registry.get(tsk.sub_agent_name)
                sub_prompt = spec.prompt if spec else ""
                sub_messages: list[dict[str, str]] = [
                    {"role": "system", "content": sub_prompt or "你是一个子任务执行助手。"},
                    {"role": "user", "content": tsk.instruction},
                ]

                sub_tool_names = spec.tools if spec else []
                sub_tools: list[dict[str, Any]] = []
                if sub_tool_names and self.tool_registry is not None:
                    all_tools = self._gather_tools()
                    sub_tools = [t for t in all_tools if t.get("function", {}).get("name") in sub_tool_names]

                try:
                    if sub_tools:
                        sub_response = await asyncio.wait_for(
                            self.llm.chat_with_tools(
                                [LLMMessage(role=m["role"], content=m["content"]) for m in sub_messages],
                                sub_tools,
                                purpose="chat",
                                max_tokens=1024,
                                model=spec.model or "",
                            ),
                            timeout=tsk.timeout,
                        )
                        result_text = sub_response.content or ""
                    else:
                        sub_response = await asyncio.wait_for(
                            self.llm.chat(
                                [LLMMessage(role=m["role"], content=m["content"]) for m in sub_messages],
                                purpose="chat",
                                max_tokens=1024,
                                model=spec.model or "",
                            ),
                            timeout=tsk.timeout,
                        )
                        result_text = sub_response.content or ""

                    self._record_llm_stats(sub_response)
                    return result_text
                except asyncio.TimeoutError:
                    self._record_llm_timeout()
                    return f"任务 {tsk.task_id} 超时 ({tsk.timeout}s)"
                except Exception as exc:
                    self._record_llm_fail()
                    return f"任务 {tsk.task_id} 失败: {exc}"

            _ = asyncio.ensure_future(
                self.task_orchestrator.execute(task, _exec)
            )
            return (
                f"子任务已提交: task_id={task.task_id}, sub_agent={sub_agent_name}, "
                f"状态=pending。使用 check_task(task_id='{task.task_id}') 查询结果。"
            )

        async def _check_task(task_id: str = "") -> str:
            task_id = task_id.strip()
            if not task_id:
                pending = self.task_orchestrator.list_tasks()
                if not pending:
                    return "无活跃任务"
                lines = [f"- {t.task_id}: [{t.status.value}] {t.sub_agent_name} → {t.instruction[:100]}"
                         for t in pending]
                return "活跃任务:\n" + "\n".join(lines)
            task = self.task_orchestrator.get_task(task_id)
            if task is None:
                return f"任务 '{task_id}' 未找到"
            status_info = f"task_id={task.task_id}, status={task.status.value}, sub_agent={task.sub_agent_name}"
            if task.status.value == "completed":
                status_info += f"\n结果: {str(task.result)[:500]}"
            elif task.status.value == "failed":
                status_info += f"\n错误: {task.error}"
            elif task.status.value == "timeout":
                status_info += f"\n超时: {task.timeout}s"
            return status_info

        async def _cancel_task(task_id: str = "") -> str:
            task_id = task_id.strip()
            if not task_id:
                return "错误: 必须指定 task_id"
            if self.task_orchestrator.cancel(task_id):
                return f"任务 '{task_id}' 已取消"
            return f"任务 '{task_id}' 取消失败（可能已不存在或已执行完毕）"

        subagent_tools = [
            ("delegate_task", (
                f"委派子任务给指定的子 Agent 异步执行。{subagent_desc}"
                "参数: sub_agent_name(子Agent名称), instruction(任务指令), timeout(超时秒数，默认120)"
            ), _delegate_task),
            ("check_task", (
                "查询异步子任务的状态。不带 task_id 则列出所有任务。"
                "参数: task_id(可选，任务ID) 使用 delegate_task 返回的 task_id 查询具体任务结果"
            ), _check_task),
            ("cancel_task", (
                "取消指定的异步子任务。参数: task_id(任务ID)"
            ), _cancel_task),
        ]

        for name, desc, handler in subagent_tools:
            if name in self._denied_tools:
                continue
            tool_def = ToolDefinition(
                name=name, description=desc, source=ToolSource.LOCAL,
            )
            self.tool_registry.register_local(tool_def, handler)

        logger.info("子 Agent 编排工具已注册: %s", ", ".join(t[0] for t in subagent_tools))

    def _auto_discover_skills(self) -> None:
        """自动发现并加载 Skills

        加载顺序（后加载的覆盖先加载的同名 Skill）：
        1. search_paths（项目根目录 skills/）— 先加载，低优先级
        2. workspace/skills — 后加载，高优先级（覆盖同名）
        """
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
                search_dir = self._config_dir.parent.resolve() / search_path
            if search_dir.exists() and search_dir.is_dir():
                for skill_path in search_dir.iterdir():
                    if skill_path.is_dir() and self.skill_manager._loader.has_valid_skill_format(skill_path):
                        record = self.skill_manager.load_skill(skill_path)
                        if record and record.state != SkillState.ERROR:
                            self.skill_manager.enable_skill(record.manifest.name)
                            logger.info("Skill 已加载: %s (来自 %s)", record.manifest.name, search_path)

        # 加载 .trae/skills/ 目录下的 skill（如 docx）
        _trae_skills_dir = self._config_dir.parent.resolve() / ".trae" / "skills"
        if _trae_skills_dir.exists() and _trae_skills_dir.is_dir():
            for _skill_path in sorted(_trae_skills_dir.iterdir()):
                if _skill_path.is_dir() and self.skill_manager._loader.has_valid_skill_format(_skill_path):
                    try:
                        _record = self.skill_manager.load_skill(_skill_path)
                        if _record and _record.state != SkillState.ERROR:
                            self.skill_manager.enable_skill(_record.manifest.name)
                            logger.info("Skill 已加载: %s (来自 .trae/skills)", _record.manifest.name)
                    except Exception as _e:
                        logger.warning("加载 .trae/skills/%s 失败: %s", _skill_path.name, _e)

        loaded = self.skill_manager.auto_discover()
        logger.info("从 workspace/skills 发现 %d 个 Skill（同名覆盖项目根目录）", len(loaded))

        total = len(self.skill_manager.list_skills())
        logger.info("总共加载 %d 个 Skill", total)

    async def _connect_mcp_servers(self) -> None:
        """连接 mcp.yaml 中配置的所有 MCP 服务器"""
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

    def _register_cli_commands(self, cli_adapter: Any) -> None:
        """向 CLI 适配器注册命令处理器"""
        from rich.table import Table

        def show_status(*args: Any) -> str | None:
            """显示系统状态"""
            table = Table(title="系统状态", show_header=False, border_style="blue")
            table.add_column("属性", style="cyan")
            table.add_column("值")

            if self.llm:
                table.add_row("LLM 模型", self.llm.config.model)
                api_key = self.llm.config.resolve_api_key()
                table.add_row("API Key", f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 8 else "(未配置)")
                table.add_row("Base URL", self.llm.config.resolve_base_url() or "(默认)")
            else:
                table.add_row("LLM", "[red]未配置[/red]")

            table.add_row("对话历史", f"{self.active_session.message_count if self.active_session else 0} 条")

            if self.memory:
                table.add_row("记忆系统", "已就绪")
            if self.sandbox:
                table.add_row("沙箱", "已就绪")
            if self.workspace:
                table.add_row("工作区", str(self.workspace.root))
            if self.skill_manager:
                skills = self.skill_manager.list_skills()
                table.add_row("已加载 Skills", f"{len(skills)} 个")
            if self.mcp_client:
                table.add_row("MCP", "已就绪")

            cli_adapter.console.print(table)
            return None

        def manage_skill(*args: Any) -> str | None:
            """Skill 管理 (list / enable / disable)"""
            if not self.skill_manager:
                cli_adapter.console.print("[yellow]Skill 管理器未初始化[/yellow]")
                return None

            skills = self.skill_manager.list_skills()
            if not skills:
                cli_adapter.console.print("[dim]没有已加载的 Skill[/dim]")
                return None

            table = Table(title="已加载 Skills", show_header=True, header_style="bold")
            table.add_column("名称", style="cyan")
            table.add_column("状态")
            table.add_column("描述")

            for s in skills:
                status_text = "[green]已启用[/green]" if s.state == SkillState.ENABLED else f"[dim]{s.state.value}[/dim]"
                table.add_row(s.manifest.name, status_text, s.manifest.description or "-")

            cli_adapter.console.print(table)
            cli_adapter.console.print("[dim]用法: /skill list | /skill enable <name> | /skill disable <name>[/dim]")
            return None

        def manage_mcp(*args: Any) -> str | None:
            """MCP 服务器管理"""
            if not self.mcp_client:
                cli_adapter.console.print("[yellow]MCP 客户端未初始化[/yellow]")
                return None

            server_count = len(self.mcp_client._servers) if hasattr(self.mcp_client, "_servers") else 0
            if server_count == 0:
                cli_adapter.console.print("[dim]没有已连接的 MCP 服务器[/dim]")
                cli_adapter.console.print("[dim]用法: /mcp list | /mcp connect <config>[/dim]")
                return None

            table = Table(title="MCP 服务器", show_header=True, header_style="bold")
            table.add_column("名称", style="cyan")
            table.add_column("状态")

            for name in self.mcp_client._servers:
                alive = "[green]已连接[/green]"
                table.add_row(name, alive)

            cli_adapter.console.print(table)
            return None

        def show_history(*args: Any) -> str | None:
            """查看对话历史"""
            if not self.active_session or not self.active_session.messages:
                cli_adapter.console.print("[dim]暂无对话历史[/dim]")
                return None

            msgs = self.active_session.messages
            cli_adapter.console.print(f"[bold]对话历史 ({len(msgs)} 条):[/bold]")
            for i, msg in enumerate(msgs):
                role = "[blue]用户[/blue]" if msg["role"] == "user" else "[green]助手[/green]"
                content = msg["content"][:120] + "..." if len(msg["content"]) > 120 else msg["content"]
                cli_adapter.console.print(f"  {i + 1}. {role}: {content}")
            return None

        def run_eval(*args: Any) -> str | None:
            """运行评估"""
            if self.eval_pipeline is None:
                cli_adapter.console.print("[yellow]评估系统未初始化[/yellow]")
                return None

            from long.eval.report import EvalTask, EvalCategory

            if not self.active_session or not self.active_session.messages:
                cli_adapter.console.print("[dim]暂无对话历史，无法评估[/dim]")
                return None

            cli_adapter.console.print("[bold]开始评估最近对话...[/bold]")

            recent = self.active_session.recent_messages(limit=10)
            user_msgs = [m for m in recent if m["role"] == "user"]
            assistant_msgs = [m for m in recent if m["role"] == "assistant"]

            if not user_msgs or not assistant_msgs:
                cli_adapter.console.print("[dim]对话轮次不足，无法评估[/dim]")
                return None

            last_user = user_msgs[-1]["content"]
            last_assistant = assistant_msgs[-1]["content"]

            task = EvalTask(
                name="recent_conversation",
                input=last_user,
                expected=None,
                category=EvalCategory.NORMAL,
            )

            report = self.eval_pipeline.run(task, output=last_assistant)

            table = Table(title="评估报告", show_header=True, header_style="bold")
            table.add_column("指标", style="cyan")
            table.add_column("值")

            table.add_row("综合分数", f"{report.score:.2f}")
            table.add_row("结果层准确性", f"{report.outcome.accuracy:.2f}")
            table.add_row("过程层分数", f"{report.process.score:.2f}")
            table.add_row("过程层效率", f"{report.process.efficiency:.2f}")
            table.add_row("Schema 合法", "✅" if report.outcome.schema_valid else "❌")
            table.add_row("约束满足", "✅" if report.outcome.constraint_satisfied else "❌")
            table.add_row("规则违反数", str(report.process.rule_violations))
            table.add_row("需要人工审核", "是" if report.needs_human_review else "否")
            table.add_row("自动审核通过", "✅" if report.auto_reviewed else "❌")

            if report.process.details:
                table.add_row("过程详情", str(report.process.details)[:100])

            cli_adapter.console.print(table)

            if self.optimizer is not None:
                self.optimizer.collector.record_eval_result(
                    task_name=task.name,
                    score=report.score,
                    category=task.category.value,
                )
                cli_adapter.console.print("[dim]评估结果已记录到优化器指标[/dim]")

            return None

        cli_adapter.register_command("/status", show_status)
        cli_adapter.register_command("/skill", manage_skill)
        cli_adapter.register_command("/mcp", manage_mcp)
        cli_adapter.register_command("/history", show_history)
        cli_adapter.register_command("/eval", run_eval)

        def show_health(*args: Any) -> str | None:
            """显示系统健康报告"""
            from long.observability.dashboard import HealthDashboard

            dashboard = HealthDashboard(
                collector=self.optimizer.collector if self.optimizer else None,
                tracer=self.tracer,
            )
            report = dashboard.format_report()
            cli_adapter.console.print(report)
            return None

        def show_traces(*args: Any) -> str | None:
            """显示最近的 Trace 记录"""
            traces = self.tracer.get_traces(limit=10)
            if not traces:
                cli_adapter.console.print("[dim]暂无 Trace 记录[/dim]")
                return None

            table = Table(title="最近 Trace 记录", show_header=True, header_style="bold")
            table.add_column("Trace ID", style="cyan")
            table.add_column("名称")
            table.add_column("耗时(ms)")
            table.add_column("Span数")
            table.add_column("失败Span")

            for t in traces:
                failed = len(t.failed_spans())
                failed_style = f"[red]{failed}[/red]" if failed > 0 else "0"
                table.add_row(
                    t.trace_id[:12],
                    t.name,
                    f"{t.duration_ms:.0f}",
                    str(len(t.spans)),
                    failed_style,
                )

            cli_adapter.console.print(table)
            return None

        def show_optimization(*args: Any) -> str | None:
            """显示优化器状态"""
            if self.optimizer is None:
                cli_adapter.console.print("[yellow]优化器未初始化[/yellow]")
                return None

            status = self.optimizer.get_status()
            table = Table(title="优化器状态", show_header=False, border_style="green")
            table.add_column("属性", style="cyan")
            table.add_column("值")

            table.add_row("对话计数", str(status.get("conversation_count", 0)))
            table.add_row("上次循环", f"{time.time() - status.get('last_cycle_time', time.time()):.0f}s 前")
            table.add_row("循环次数", str(status.get("cycle_count", 0)))
            table.add_row("已应用变更", str(len(status.get("applied_changes", []))))
            table.add_row("指标名称", ", ".join(status.get("metric_names", [])[:5]))
            table.add_row("后台运行", "是" if status.get("background_running") else "否")

            cli_adapter.console.print(table)

            history = self.optimizer.get_cycle_history(limit=5)
            if history:
                cli_adapter.console.print("\n[bold]最近 OODA 循环:[/bold]")
                for h in history:
                    from datetime import datetime
                    ts = datetime.fromtimestamp(h["timestamp"]).strftime("%H:%M:%S")
                    cli_adapter.console.print(
                        f"  {ts} - {h['status']} (提案:{h['proposals_count']}, 批准:{h['approved_count']})"
                    )

            return None

        cli_adapter.register_command("/health", show_health)
        cli_adapter.register_command("/traces", show_traces)
        cli_adapter.register_command("/optimization", show_optimization)

    def _build_system_prompt(self) -> str:
        """构建系统提示词

        精简原则：只告诉模型"做什么"，不告诉"怎么做"。
        模型本身知道如何调用工具，过多的规则只会造成冲突。
        """
        _skill_file_map = {
            "xlsx": (".xlsx / .xls", "openpyxl"),
            "docx": (".docx", "python-docx"),
            "pptx": (".pptx", "python-pptx"),
            "pdf": (".pdf", "reportlab/fpdf"),
        }
        _registered_skills = {s.manifest.name for s in self.skill_manager.list_skills() if s.state.value == "enabled"} if self.skill_manager else set()

        parts: list[str] = [
            "你是 Long，一个智能 AI 助手。你有工具可以使用。",
            "",
            "## 核心原则",
            "1. **有工具直接调用** — 需要实时数据（新闻、天气、搜索等）时，直接调用对应工具，不要先回复文字。",
            "2. **代码写完后必须执行** — 用 write_file 写代码，用 execute_file 执行。不要只写不执行。",
            "3. **不要编造结果** — 没有调用工具就不要说有结果。所有输出必须来自工具的真实返回值。",
            "4. **工具失败时换方法** — 搜索不到就换关键词，代码报错就修。不要放弃。",
            "",
            "## 数据源选择",
            "- 天气 → query_weather(city='城市名')",
            "- 实时信息（新闻、股价、赛事等）→ tavily_search(query='关键词')",
            "- 当前时间 → get_current_time()",
            "- 文件读写 → read_file / write_file",
            "- 执行代码 → execute_file(path='output/xxx.py')",
            "",
            "## 文件处理",
            "处理文件时，先检查是否有对应的 Skill（read_skill_md），有则按 Skill 指引操作，没有则直接用 Python 库。",
        ]

        # 动态生成 skill 引用
        for _skill_name, (_ext, _lib) in _skill_file_map.items():
            if _skill_name in _registered_skills:
                parts.append(f"- **{_ext}** → 先 `read_skill_md(skill_name='{_skill_name}')` 读取指引，再按指引操作。")
            else:
                parts.append(f"- **{_ext}** → 用 `execute_code` + {_lib} 库处理。")

        parts += [
            "- 纯文本文件（.txt/.md/.py/.json 等）→ `read_file` 读取",
            "- 代码文件保存到 `output/` 目录，路径用相对路径",
            "- matplotlib 中文字体: `plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']`",
            "",
            "## 工作流示例",
            "需要搜索信息并生成报告时：",
            "1. tavily_search 获取数据",
            "2. write_file 写入代码",
            "3. execute_file 执行代码生成文件",
            "4. 向用户展示结果",
        ]

        semistatic = self._build_semistatic_prompt()
        if semistatic:
            parts.append(semistatic)

        if self.workspace is not None:
            memory_path = self.workspace.root / "MEMORY.md"
            if memory_path.exists():
                try:
                    mem_content = memory_path.read_text(encoding="utf-8").strip()
                    if mem_content:
                        parts.append("")
                        parts.append(f"## 长期记忆\n{mem_content}")
                except Exception:
                    pass

            knowledge_dir = self.workspace.root / "knowledge"
            if knowledge_dir.is_dir():
                try:
                    knowledge_parts: list[str] = []
                    for kf in sorted(knowledge_dir.glob("*.md")):
                        kf_content = kf.read_text(encoding="utf-8")[:2000]
                        if kf_content.strip():
                            knowledge_parts.append(f"### {kf.stem}\n{kf_content.strip()}")
                    if knowledge_parts:
                        parts.append("")
                        parts.append("## 领域知识\n" + "\n\n".join(knowledge_parts))
                except Exception:
                    pass

        # 记忆注入已移至 _handle_user_message 中的 memory.search() 动态检索
        # 保留 MEMORY.md 的"长期记忆"部分作为兜底（上方 line ~1960 已读取）

        return "\n".join(parts)

    def _build_static_prompt(self) -> str:
        """构建静态提示词层（规则文本，会话期间不变）

        支持渐进式发布：通过 FeatureFlag 按 session 灰度分流 Prompt 版本。
        Harness Engineering: AGENTS.md 是 Agent 人格的唯一事实来源。
        """
        session_id = getattr(self.active_session, "id", "") if self.active_session else ""

        if self.prompt_version is not None:
            agents_md = self.prompt_version.get_prompt(session_id)
            if agents_md:
                return agents_md

        agents_md = ""
        if self.workspace is not None:
            agents_path = self.workspace.root / "AGENTS.md"
            if agents_path.exists():
                try:
                    agents_md = agents_path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

        if agents_md:
            return agents_md

        parts: list[str] = [
            "你是 Long，一个智能 AI 助手，可以调用工具和技能来帮助用户完成任务。",
            "",
            "## 🚨 核心规则：工具调用效率约束",
            "你受到严格的工具调用次数限制，违反会被系统拦截：",
            "- **🔍 搜索绝对限制**：tavily_search 最多只能调用2次！首次搜索后必须使用结果，不得换关键词重复搜索。",
            "- **🚫 禁止无意义探索**：不得为了'看看有什么'调用 list_files/read_skill_md，除非任务确实需要。",
            "- **💻 写代码后必须执行**：write_file 后必须立即用 execute_code/execute_file 执行，不得只写不执行。",
            "- **📄 报告格式**：使用 Markdown(.md)/HTML(.html)/PDF 格式，不要用 xlsx（除非用户明确要求 Excel）。",
            "",
            "## 🚨 核心规则：代码重试与错误恢复",
            "- **执行失败最多重试2次**：execute_code 失败后，分析错误原因再重试，同一段代码最多执行2次。",
            "- **2次失败后必须换方法**：不是继续重试同一代码，而是换方案（修改代码逻辑/读取 Skill 文档/用其他工具）。",
            "- **禁止连续4次执行相同代码**：如果你发现自己在重复执行同样的代码，立即停下来换方法。",
            "- **Skill 中的代码可直接复制执行**：读 read_skill_md 后，skill 内的完整代码块可以直接拷贝到 execute_code，无需自己重新编写。",
            "",
            "## 重要：文件类型处理规则",
            "处理不同类型的文件时，请遵循以下规则：",
            "- **.xlsx / .xls** → 先用 `read_skill_md(skill_name='xlsx')` 读取对应 Skill 的 SKILL.md 文档，然后按照文档中的指令来处理文件。",
            "- **.docx** → 先用 `read_skill_md(skill_name='docx')` 读取对应 Skill 的 SKILL.md 文档，然后按照文档中的指令（使用 python-docx 编写 Python 代码）来处理文件。",
            "- **.pptx** → 先用 `read_skill_md(skill_name='pptx')` 读取对应 Skill 的 SKILL.md 文档，然后按照文档中的指令（使用 python-pptx 编写 Python 代码）来处理文件。",
            "- **如果 read_skill_md 返回 skill 未找到** → 直接使用 execute_code 工具，用 python-docx/python-pptx/openpyxl 等 Python 库编写代码来处理文件。必须实际执行代码，不要只描述代码！",
            "- **如果尝试读取非文本文件报错** → 明确告知用户该文件是二进制格式，并建议使用对应 Skill 或解释原因。",
            "- **.txt / .md / .py / .yaml / .json 等纯文本** → 用 `read_file` 读取。",
            "- **未知文件类型** → 先用 `list_files` 查看，再决定。",
            "",
            "## 重要：代码生成与执行规则",
            "当用户要求生成并执行代码时，你必须遵循以下流程：",
            "1. 使用 `write_file` 将代码写入文件（文件会自动保存到 output/ 目录）",
            "2. 使用 `execute_file` 工具执行代码文件（传入 path 参数），获取执行结果",
            "3. 如果执行失败（有错误），分析错误信息，修复代码后重新写入并执行",
            "4. 重复步骤3直到代码执行成功，最多重试2次",
            "5. 将最终执行结果展示给用户",
            "**绝对不要只生成代码而不执行！** 用户要求执行时，必须调用 `execute_file` 工具执行代码文件。",
            "",
            "### ⚠️ 代码执行路径规则",
            "- 保存文件时使用相对路径，如 `output/report.html` 或 `output/chart.png`",
            "- 环境变量 `OUTPUT_DIR` 指向输出目录，可用 `os.environ.get('OUTPUT_DIR', 'output')`",
            "- **禁止使用绝对路径** 如 `/home/user/output/` 或 `/workspace/output/`，这些路径不存在",
            "- 保存图片前确保目录存在：`os.makedirs('output', exist_ok=True)`",
            "- matplotlib 保存图片示例: `plt.savefig('output/chart.png', dpi=150)`",
            "- 中文字体使用: `plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']`",
            "",
            "## 🔍 搜索+代码标准工作流",
            "任务需要最新信息时，选择合适的数据源：",
            "- **天气查询**：必须使用 query_weather 工具！调用 query_weather(city='城市名')。多城市用逗号分隔，如 city='北京,上海,杭州'。",
            "- **天气查询绝对禁止使用 tavily_search**：凡是天气相关查询，一律不许使用搜索工具。",
            "- **天气查询绝对禁止使用 execute_code 直接调 API**：不要自己写 Python 代码调用和风天气 API，必须用 query_weather 工具，它会返回格式化的简洁文本。",
            "- 新闻/股价/赛事等：tavily_search 搜索获取数据",
            "获取数据后：",
            "1. **直接基于结果**用 write_file + execute_file 生成报告/图表",
            "2. 如果结果不够详细，用已有数据合理推断（并注明来源），不要尝试抓取网页",
            "3. 向用户展示最终结果",
            "⚠️ 禁止连续搜索！每次搜索后必须先分析结果再决定下一步。",
            "⚠️ 如果搜索失败：尝试其他工具或方法，绝对不要编造数据，也不要轻易放弃！",
            "",
            "## ⚠️ 搜索决策规则",
            "并非所有问题都需要搜索！遵循以下规则：",
            "- **天气查询**：必须使用 query_weather 工具（调用 query_weather(city='城市名')），绝对不许使用 tavily_search 或 execute_code 直接调 API",
            "- **需要搜索**：新闻、实时数据、最新政策、股价、近期事件等有时效性的信息",
            "- **不需要搜索**：知识问答、算法实现、代码编写、数学计算、概念解释、翻译、写作、通用技术问题等",
            "- **判断标准**：如果你的训练数据中已有足够知识回答，就不要搜索。只有当答案可能随时间变化时才搜索。",
            "- **错误示例**：用户问\"快速排序算法\"→ 不需要搜索（这是固定知识）；用户问\"Python如何读取CSV\"→ 不需要搜索",
            "- **正确示例**：用户问\"杭州今天天气\"→ 调用 query_weather(city='杭州')；用户问\"2026年AI最新进展\"→ 需要搜索（时效性信息）",
            "",
            "## 重要：Skill 创建规则",
            "当用户要求创建 Skill 时，直接按以下结构创建，不要先探索现有 Skill：",
            "1. 创建 `skills/<skill-name>/SKILL.md`：包含 YAML front matter（name, description）和完整 Skill 文档",
            "2. 创建 `skills/<skill-name>/__init__.py`：包含 SKILL_NAME、SKILL_VERSION、SKILL_DESCRIPTION 等元数据",
            "3. 如有需要，创建 `skills/<skill-name>/scripts/` 目录存放脚本",
            "SKILL.md 的 YAML front matter 格式：",
            "```yaml",
            "---",
            "name: skill-name",
            "description: |",
            "  Skill 的详细描述，包含触发条件说明",
            "---",
            "```",
            "__init__.py 的格式：",
            "```python",
            'SKILL_NAME = "skill-name"',
            'SKILL_VERSION = "1.0.0"',
            'SKILL_DESCRIPTION = "Skill 描述"',
            'SKILL_PERMISSIONS = ["compute.cpu"]',
            "SKILL_TOOLS = []",
            "SKILL_DEPENDENCIES = []",
            'SKILL_ENTRY_POINT = ""',
            "```",
            "**创建 Skill 时不要先 list_files 或 read_file 探索现有 Skill，直接创建即可！**",
        ]
        return "\n".join(parts)

    def _build_semistatic_prompt(self) -> str:
        """构建半静态提示词层（偏好/画像/技能，变更时标记 dirty）"""
        parts: list[str] = []

        if self.preference_store is not None:
            pref_text = self.preference_store.format_for_prompt()
            if pref_text:
                parts.append("")
                parts.append(pref_text)

        if self.user_profile is not None:
            profile_text = self.user_profile.format_for_prompt()
            if profile_text:
                parts.append("")
                parts.append(profile_text)

        if self.summary_store is not None:
            summary_text = self.summary_store.format_for_prompt(days=7)
            if summary_text:
                parts.append("")
                parts.append(summary_text)

        if self.skill_manager is not None:
            skills = [s for s in self.skill_manager.list_skills() if s.state.value == "enabled"]
            if skills:
                skill_lines: list[str] = []
                for s in skills:
                    desc = s.manifest.description.strip().replace("\n", " ")
                    if len(desc) > 150:
                        desc = desc[:147] + "..."
                    has_tools = "🔧" if s.manifest.tools else "📄"
                    skill_lines.append(f"- {has_tools} **{s.manifest.name}**: {desc}")
                parts.append("\n## 可用技能\n" + "\n".join(skill_lines))
                parts.append(
                    "\n当用户的请求与上述技能描述匹配时，你应该使用对应的工具函数。"
                    "对于标记为 🔧 的技能，必须调用对应工具执行；对于标记为 📄 的技能，按技能文档指引执行代码。"
                )

                # 自动注入已启用文档类 skill 的 SKILL.md 内容到系统提示词
                _DOC_SKILL_NAMES = ("pptx", "docx", "xlsx", "qweather")
                for s in skills:
                    # 只注入已启用（enabled）的 skill
                    if s.manifest.name in _DOC_SKILL_NAMES and s.module is None and s.state.value == "enabled":
                        skill_md_path = Path(s.path) / "SKILL.md"
                        if skill_md_path.exists():
                            try:
                                doc = skill_md_path.read_text(encoding="utf-8")
                                if len(doc) > 3000:
                                    doc = doc[:2997] + "..."
                                parts.append(f"\n## {s.manifest.name.upper()} Skill 指引\n{doc}")
                            except Exception:
                                pass

        return "\n".join(parts)

    def _build_dynamic_time(self) -> str:
        """构建动态提示词层（当前时间 + 语义时间归一化上下文，每次请求时更新）

        遵循前缀缓存原则：此层放在 prompt 末尾，仅使末尾 token 失效。
        """
        from datetime import datetime, timezone, timedelta as td

        now = datetime.now(timezone(td(hours=8)))
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        current_date = f"{now.strftime('%Y年%m月%d日')} {weekdays[now.weekday()]} {now.strftime('%H:%M')}"
        iso_date = now.strftime("%Y-%m-%d")
        unix_ts = int(now.timestamp())

        return (
            f"\n\n## 当前时间\n"
            f"- 中文: {current_date}\n"
            f"- ISO: {iso_date}\n"
            f"- Unix: {unix_ts}\n"
            f"- 时区: UTC+08:00\n"
            f"以上时间是系统确认的当前真实时间。当用户提到'今天'、'现在'、'最近'等时间词时，"
            f"请使用上面的时间。搜索结果中出现同年同月的数据即为有效数据，请勿误判为过期。"
            f"当需要实时数据时（如天气、股价），如果搜索不到当日数据，最近的可用数据也可作为参考。"
        )

    def _gather_tools(self) -> list[dict[str, Any]]:
        """收集所有可用工具，格式化为 OpenAI function definitions

        每个工具的 metadata 中包含 _source 字段用于标识来源：
        - "local": 本地内置工具
        - "skill": Skill 工具
        - "mcp": MCP 服务器工具
        
        去重规则：同名工具，local 优先于 skill（local 有更详细的参数定义）
        """
        tools: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        # 先收集 local 工具（优先级最高，参数定义更详细）
        local_tools: list[dict[str, Any]] = []
        if self.tool_registry is not None:
            file_tools = {
                "list_files": {
                    "name": "list_files",
                    "description": "列出工作区目录下的文件和子目录",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "目录路径（相对于工作区根目录，留空表示根目录）",
                                "default": "",
                            },
                        },
                    },
                },
                "read_file": {
                    "name": "read_file",
                    "description": "读取工作区中指定文件的内容",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "文件路径（相对于工作区根目录）"},
                        },
                        "required": ["path"],
                    },
                },
                "write_file": {
                    "name": "write_file",
                    "description": "将内容写入工作区中的文件（会自动创建父目录）。生成的代码文件默认保存到 output/ 目录；skills 保存到 skills/ 目录；其他已知目录（data/、configs/ 等）保持原路径。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "文件路径。代码文件写 'fibonacci.py' 会自动保存为 'output/fibonacci.py'；skill 文件写 'skills/xxx/SKILL.md' 保持原路径。"},
                            "content": {"type": "string", "description": "要写入的内容"},
                        },
                        "required": ["path", "content"],
                    },
                },
                "delete_file": {
                    "name": "delete_file",
                    "description": "删除工作区中的文件或目录",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "文件或目录路径（相对于工作区根目录）"},
                        },
                        "required": ["path"],
                    },
                },
                "read_skill_md": {
                    "name": "read_skill_md",
                    "description": "读取指定 skill 的 SKILL.md 文档。当需要使用某个 skill（如 xlsx、docx）时，必须先调用此工具读取其文档，了解如何正确处理该类文件。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_name": {
                                "type": "string",
                                "description": "skill 名称（如 xlsx、docx、calculator 等）",
                            },
                        },
                        "required": ["skill_name"],
                    },
                },
                "execute_code": {
                    "name": "execute_code",
                    "description": "在沙箱中执行代码并返回结果。当用户要求执行代码时必须使用此工具。支持 Python、JavaScript、Bash。可通过 args 传入命令行参数。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "要执行的代码内容",
                            },
                            "language": {
                                "type": "string",
                                "description": "编程语言（python/javascript/bash），默认 python",
                                "default": "python",
                            },
                            "args": {
                                "type": "string",
                                "description": "命令行参数（空格分隔），如 '杭州' 或 '--output json'",
                                "default": "",
                            },
                        },
                        "required": ["code"],
                    },
                },
                "execute_file": {
                    "name": "execute_file",
                    "description": "执行工作区中已有的代码文件。当用户要求执行已写入的文件时使用此工具。可通过 args 传入命令行参数。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "要执行的代码文件路径（如 'output/fibonacci.py'）",
                            },
                            "args": {
                                "type": "string",
                                "description": "命令行参数（空格分隔），如 '杭州' 或 '北京,上海'",
                                "default": "",
                            },
                        },
                        "required": ["path"],
                    },
                },
                "get_current_time": {
                    "name": "get_current_time",
                    "description": "获取精确的当前日期和时间。注意：system prompt 中已包含当前时间，通常不需要调用此工具，除非需要精确到秒的时间。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
                "query_weather": {
                    "name": "query_weather",
                    "description": "查询指定城市的实时天气和7天预报（使用和风天气API）。⚠️ 天气查询必须使用此工具，禁止使用 tavily_search 查天气。返回结构化天气数据：温度、湿度、风力、降水等。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名，多城市用英文逗号分隔（如'杭州'或'北京,上海,广州'）",
                            },
                        },
                        "required": ["city"],
                    },
                },
                "tavily_search": {
                    "name": "tavily_search",
                    "description": "使用 Tavily API 搜索网络获取最新信息。⚠️ 禁止用于天气查询（请用 query_weather 工具）。严格限制：每个主题最多搜索2次，系统会拦截超过限制的调用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索关键词（请用简洁精准的关键词，避免重复搜索相同内容。天气查询请用 query_weather 工具）",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "最大返回结果数（1-5，默认5）",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
            for tool_def in file_tools.values():
                if self.tool_registry.get_tool(tool_def["name"]):
                    local_tools.append({"type": "function", "function": tool_def, "_source": "local"})

        # 添加 local 工具（优先级最高）
        for t in local_tools:
            name = t["function"]["name"]
            if name not in seen_names:
                tools.append(t)
                seen_names.add(name)

        # 添加 Skill 工具（跳过已存在的同名工具）
        if self.skill_manager is not None:
            for skill in self.skill_manager.list_skills():
                for tool_name in skill.manifest.tools:
                    if tool_name in seen_names:
                        continue
                    if not self.tool_registry.get_tool(tool_name):
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

    def _clean_tools_for_api(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """清理工具定义中的内部字段，确保 API 兼容"""
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

    _MAX_TOOL_RESULT_LEN = 8000

    async def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """执行工具调用并返回结果文本"""
        # 检查工具结果缓存
        if self._tool_cache is not None:
            cached = self._tool_cache.get(tool_name, arguments)
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
                            output = str(result.output)

                            if tool_name == "write_file":
                                file_path = arguments.get("path", "")
                                file_content = arguments.get("content", "")
                                if file_path and file_content:
                                    from long.eval.content_validator import validate_content
                                    passed, feedback = validate_content(file_path, file_content)
                                    if not passed:
                                        output += feedback

                            return output
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

        if len(result) > self._MAX_TOOL_RESULT_LEN:
            result = result[:self._MAX_TOOL_RESULT_LEN] + "\n...[结果已截断]"

        # 存入缓存
        if self._tool_cache is not None and not result.startswith(("未知工具:", "工具执行失败:", "工具异常:", "MCP工具异常:")):
            self._tool_cache.put(tool_name, arguments, result)

        return result

    async def _add_message_to_session_and_memory(self, role: str, content: str) -> None:
        """同时写入 session 和 memory 的辅助方法"""
        if self.active_session is not None:
            self.active_session.add_message(role, content)
        if self.memory is not None:
            try:
                await self.memory.add_message(role, content)
            except Exception:
                pass

    async def _handle_user_message(self, cli_adapter: Any, message: str, *, ws_session_id: str = "") -> None:
        """处理用户非命令消息，支持 tool-calling

        Args:
            cli_adapter: CLI/WebUI 适配器
            message: 用户消息
            ws_session_id: WebSocket 会话ID（WebUI 模式下用于隔离对话历史）
        """
        if self.llm is not None:
            try:
                from long.llm.base import LLMMessage

                api_key = self.llm.config.resolve_api_key()
                if not api_key:
                    cli_adapter.console.print("[red]API Key 未配置，请在 .env 中设置 LLM_API_KEY[/red]")
                    return

                with self.tracer.trace(
                    "user_message",
                    attributes={"user_input_length": len(message)},
                ) as trace:
                    # 记录会话开始时间戳，用于过滤历史输出文件
                    self._session_start_ts = time.time()

                    # 按 ws_session_id 获取或创建对应的 Session
                    session = self._get_or_create_ws_session(ws_session_id)
                    await self._add_message_to_session_and_memory("user", message)

                    with cli_adapter.console.status("[bold cyan]⏳ 正在处理输入...[/bold cyan]", spinner="dots"):

                        async def _detect_preferences() -> None:
                            if self.preference_store is not None:
                                detections = self.preference_store.detect_preferences(message)
                                if detections:
                                    count = self.preference_store.apply_detected(detections)
                                    if count > 0:
                                        logger.info("检测到 %d 条新偏好", count)
                                        if self._prompt_cache is not None:
                                            self._prompt_cache.mark_dirty()

                        async def _store_memory() -> None:
                            if self.memory is not None:
                                try:
                                    await self.memory.store(
                                        f"user: {message}",
                                        memory_type=MemoryType.EPISODIC,
                                        importance=0.6,
                                    )
                                except Exception:
                                    pass

                        def _gather_tools_sync() -> list[dict[str, Any]]:
                            if self._tools_cache is not None:
                                return self._tools_cache
                            result = self._gather_tools()
                            self._tools_cache = result
                            return result

                        def _build_prompt_sync() -> str:
                            if self._prompt_cache is not None:
                                return self._prompt_cache.build(
                                    semistatic_builder=self._build_semistatic_prompt,
                                    dynamic_time=self._build_dynamic_time(),
                                )
                            return self._build_system_prompt()

                        self.llm.reset_task_budget()

                        await asyncio.gather(_detect_preferences(), _store_memory())

                        tools = _gather_tools_sync()
                        raw_history = session.recent_messages(limit=40)
                        history_msgs = self._trim_history_by_topic(raw_history, message)

                        # 从记忆系统检索相关记忆
                        memory_context = ""
                        if self.memory is not None:
                            try:
                                # 对短消息（如"生成pdf"）扩展搜索查询
                                search_query = message
                                if len(message) < 15:
                                    # 从对话历史中提取最近的助手回复作为搜索上下文
                                    recent_assistant = [
                                        m for m in raw_history
                                        if m.get("role") == "assistant" and len(m.get("content", "")) > 50
                                    ]
                                    if recent_assistant:
                                        last_topic = recent_assistant[-1].get("content", "")[:100]
                                        search_query = f"{message} {last_topic}"

                                relevant = await self.memory.search(search_query, limit=5)
                                if relevant:
                                    mem_lines = []
                                    for item in relevant[:5]:
                                        content = getattr(item, "content", str(item))[:200].strip()
                                        if content:
                                            mem_lines.append(f"- {content}")
                                    if mem_lines:
                                        memory_context = (
                                            "\n## 相关记忆\n"
                                            "以下是历史积累的知识和经验，请参考：\n"
                                            + "\n".join(mem_lines) + "\n"
                                        )
                                    logger.info(
                                        "记忆检索: 查询='%s', 命中%d条, 上下文长度=%d",
                                        message[:30], len(relevant), len(memory_context),
                                    )
                            except Exception:
                                pass

                        system_prompt = _build_prompt_sync()
                        if memory_context:
                            system_prompt = system_prompt + "\n" + memory_context
                        history_msgs = [{"role": "system", "content": system_prompt}] + history_msgs

                    # 所有查询统一走 LLM，让 Agent 自主决策工具调用
                    if tools:
                        await self._chat_with_tools_loop(cli_adapter, history_msgs, tools)
                    else:
                        await self._chat_stream(cli_adapter, history_msgs)

                    self._save_session()

            except LLMBudgetExceededError as e:
                from rich.panel import Panel
                cli_adapter.console.print(
                    Panel(f"[yellow]预算超限: {e}[/yellow]", title="Budget", border_style="yellow")
                )
                if self.active_session is not None:
                    self.active_session.add_message(
                        "assistant", f"[系统提示: 预算超限，请稍后再试]"
                    )
                    self._save_session()

            except LLMTimeoutError as e:
                from rich.panel import Panel
                cli_adapter.console.print(
                    Panel(f"[yellow]请求超时: {e}[/yellow]", title="Timeout", border_style="yellow")
                )
                if self.active_session is not None:
                    self.active_session.pop_last_user_message()
                    self._save_session()

            except asyncio.TimeoutError:
                self._record_llm_timeout()
                from rich.panel import Panel
                cli_adapter.console.print(
                    Panel(
                        "[yellow]操作超时，LLM 服务响应过慢。请稍后重试或简化问题。[/yellow]",
                        title="Timeout",
                        border_style="yellow",
                    )
                )
                if self.active_session is not None:
                    self.active_session.pop_last_user_message()
                    self._save_session()

            except LLMError as e:
                import traceback
                error_detail = f"{type(e).__name__}: {e}"
                logger.error("LLM 调用失败: %s\n%s", e, traceback.format_exc())
                from rich.panel import Panel

                cli_adapter.console.print(
                    Panel(f"[red]LLM 调用失败: {error_detail}[/red]", title="Error", border_style="red")
                )

                if self.active_session is not None:
                    self.active_session.add_message(
                        "assistant", f"[系统错误: {error_detail}，请重新提问]"
                    )
                    self._save_session()

            except Exception as e:
                import traceback
                error_detail = f"{type(e).__name__}: {e}"
                logger.error("处理消息失败: %s\n%s", e, traceback.format_exc())
                from rich.panel import Panel

                cli_adapter.console.print(
                    Panel(f"[red]处理失败: {error_detail}[/red]", title="Error", border_style="red")
                )

                if self.active_session is not None:
                    self.active_session.add_message(
                        "assistant", f"[系统错误: {error_detail}，请重新提问]"
                    )
                    self._save_session()
            finally:
                # 先发送追踪数据到前端（在 turn_complete 之前）
                self._send_trace_to_frontend(cli_adapter)
                # 通知前端本轮对话已结束
                self._notify_turn_complete(cli_adapter)
                # 记忆生命周期管理
                if self.memory is not None:
                    try:
                        await self.memory.auto_promote()
                    except Exception:
                        pass
                    # 每10轮对话清理一次过期记忆
                    if self.active_session and self.active_session.message_count % 10 == 0:
                        try:
                            await self.memory.cleanup_expired()
                        except Exception:
                            pass
            self._check_alerts()
        else:
            cli_adapter.console.print(
                f"[dim]收到: {message}[/dim] "
                "[yellow](LLM 未配置，请配置 .env 中的 LLM_API_KEY)[/yellow]"
            )

    async def _chat_with_tools_loop(
        self, cli_adapter: Any, history_msgs: list[dict[str, str]], tools: list[dict[str, Any]]
    ) -> None:
        """带工具调用的聊天循环

        执行策略（优先级从高到低）：
        1. PlanIR 结构化计划 — 所有请求先生成计划，再逐步骤执行
        2. 降级模式（Fallback Loop）— PlanIR 失败时兜底
        """
        from long.llm.base import LLMMessage

        # 优先使用 PlanIR 结构化计划
        plan_ok = await self._try_plan_execution(
            cli_adapter, history_msgs, tools
        )
        if plan_ok:
            return

        # 降级模式
        await self._fallback_tool_call_loop(cli_adapter, history_msgs, tools)

    @staticmethod
    def _to_llm_messages(messages: list[dict[str, Any]]) -> list:
        from long.llm.base import LLMMessage
        result = []
        for m in messages:
            if isinstance(m, LLMMessage):
                result.append(m)
            elif isinstance(m, dict):
                result.append(LLMMessage(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                    name=m.get("name"),
                ))
            else:
                result.append(m)
        return result

    async def _cognitive_runtime_loop(
        self, cli_adapter: Any, history_msgs: list[dict[str, str]], tools: list[dict[str, Any]]
    ) -> bool:
        """认知运行时模式 — 基于 State Graph 的执行循环

        替代 while True 循环，支持：
        - THINK → ACT → OBSERVE → REFLECT → PLAN → OUTPUT
        - 分支、回滚、条件跳转
        - 反思和自修正
        - Checkpoint/Resume

        Returns:
            True 如果成功完成，False 如果需要降级
        """
        try:
            from long.cognitive.runtime import (
                CognitiveRuntime, CognitiveContext,
            )
        except ImportError:
            logger.warning("Cognitive Runtime 不可用，降级到 Fallback 模式")
            return False

        import time as _time
        _last_displayed_round = 0

        def _maybe_show_round():
            nonlocal _last_displayed_round
            r = context.round_count
            if r > _last_displayed_round:
                _last_displayed_round = r
                cli_adapter.console.print()
                cli_adapter.console.print(
                    f"━━━ [bold blue]Round {r}[/bold blue] [dim](最多{context.max_rounds}轮, 搜索:{context.search_count})[/dim] ━━━"
                )

        async def llm_chat_fn(messages, purpose="chat"):
            _maybe_show_round()
            cli_adapter.console.print("[dim]  🤔 LLM 思考中...[/dim]", end="\r")
            _think_start = _time.monotonic()
            llm_msgs = self._to_llm_messages(messages)
            response = await self.llm.chat(llm_msgs, purpose=purpose)
            self._record_llm_stats(response)
            _think_elapsed = _time.monotonic() - _think_start
            cli_adapter.console.print(f"[dim]  🤔 LLM 思考完成 ({_think_elapsed:.1f}s)[/dim]")
            return response

        async def llm_chat_with_tools_fn(messages, tools_list, purpose="chat", **kwargs):
            _maybe_show_round()
            cli_adapter.console.print("[dim]  🤔 LLM 思考中...[/dim]", end="\r")
            _think_start = _time.monotonic()
            llm_msgs = self._to_llm_messages(messages)
            response = await self.llm.chat_with_tools(llm_msgs, tools_list, purpose=purpose, **kwargs)
            self._record_llm_stats(response)
            _think_elapsed = _time.monotonic() - _think_start
            cli_adapter.console.print(f"[dim]  🤔 LLM 思考完成 ({_think_elapsed:.1f}s)[/dim]")
            return response

        async def tool_execute_fn(tool_name, arguments):
            import json as _json
            from long.cognitive.compression import SemanticCompressor as _SC
            _tool_compressor = _SC()
            display_args = {}
            for k, v in (arguments or {}).items():
                if isinstance(v, str) and len(v) > 60:
                    display_args[k] = v[:60] + "..."
                else:
                    display_args[k] = v
            param_str = ", ".join(f"{k}={v!r}" for k, v in display_args.items())
            cli_adapter.console.print(
                f"[bold yellow]🔧[/bold yellow] [bold cyan]{tool_name}({param_str})[/bold cyan]"
            )
            exec_start = _time.monotonic()
            result = await self._execute_tool(tool_name, arguments)
            exec_elapsed = _time.monotonic() - exec_start

            # 压缩结果（用于传给 LLM 和显示）
            original_len = len(result)
            compressed_result = _tool_compressor.compress(tool_name, result)

            # 天气查询调试日志（可按需开启为 DEBUG）
            if tool_name == "execute_file" and ("天气" in str(arguments) or "qweather" in str(arguments)):
                _cities_found = [c for c in ("北京", "上海", "阜阳", "杭州", "广州", "深圳", "成都", "武汉", "南京", "重庆") if c in result]
                _cities_after = [c for c in ("北京", "上海", "阜阳", "杭州", "广州", "深圳", "成都", "武汉", "南京", "重庆") if c in compressed_result]
                logger.debug(
                    "天气查询压缩: original=%d→compressed=%d, cities_before=%s, cities_after=%s",
                    original_len, len(compressed_result), _cities_found, _cities_after,
                )

            is_error = result.startswith((
                "未知工具:", "工具执行失败:", "工具异常:", "MCP工具异常:",
                "Skill 执行错误:", "Skill 工具", "沙箱未初始化",
                "TAVILY_API_KEY 未配置", "搜索执行失败", "搜索超时",
            )) or any(
                result.startswith(f"{prefix} 错误:")
                for prefix in ("list_files", "read_file", "write_file", "delete_file")
            ) or any(
                result.startswith(f"{prefix} 失败:")
                for prefix in ("执行文件", "读取 SKILL.md")
            )
            if is_error:
                cli_adapter.console.print(f"  [bold red]❌[/bold red] [dim]{result[:100]}[/dim] [dim]({exec_elapsed:.1f}s)[/dim]")
            else:
                # 对 execute_file/execute_code 的结果做简要预览
                if tool_name in ("execute_file", "execute_code") and len(compressed_result) > 300:
                    key_lines = []
                    for line in compressed_result.split("\n"):
                        stripped = line.strip()
                        if stripped.startswith(("【", "实况", "预报", "⚠", "输出:", "执行成功")):
                            key_lines.append(stripped)
                    if key_lines:
                        preview = " | ".join(key_lines[:3])
                        if len(preview) > 80:
                            preview = preview[:77] + "..."
                    else:
                        preview = compressed_result[:80].replace("\n", " ")
                    len_info = f" ({len(compressed_result)} 字符" + (f", 原始{original_len}" if original_len != len(compressed_result) else "") + ")"
                    cli_adapter.console.print(f"  [bold green]→[/bold green] [dim]{preview}...[/dim][dim]{len_info} ({exec_elapsed:.1f}s)[/dim]")
                else:
                    result_preview = compressed_result[:80].replace("\n", " ")
                    len_info = f" ({len(compressed_result)} 字符)" if len(compressed_result) > 80 else ""
                    cli_adapter.console.print(f"  [bold green]→[/bold green] [dim]{result_preview}{'...' if len(compressed_result) > 80 else ''}[/dim][dim]{len_info} ({exec_elapsed:.1f}s)[/dim]")
            return compressed_result

        def _strip_code_and_tool_calls(text: str) -> str:
            """从 LLM 输出中剥离原始代码块和 tool_calls XML，避免显示到前端"""
            result = text
            # 剥离 <tool_calls>...</tool_calls>完整块
            result = re.sub(r'<tool_calls>[\s\S]*?</tool_calls>', '', result, flags=re.IGNORECASE)
            # 剥离残留的 <tool>...</tool> 标签
            result = re.sub(r'<tool>[\s\S]*?</tool>', '', result, flags=re.IGNORECASE)
            result = re.sub(r'<code>[\s\S]*?</code>', '', result, flags=re.IGNORECASE)
            # 剥离大的 Python 代码块
            result = re.sub(r'```python[\s\S]*?```', '', result)
            result = re.sub(r'```[\s\S]*?```', '', result)
            # 清理多余空行
            result = re.sub(r'\n{3,}', '\n\n', result).strip()
            return result

        def _scan_output_files(output_dir: str) -> list[str]:
            """扫描 output 目录，只返回当前会话生成后创建/修改的文件"""
            if not os.path.isdir(output_dir):
                return []
            binary_exts = {'.pptx', '.pdf', '.docx', '.xlsx', '.xls', '.zip'}
            files = []
            session_start = self._session_start_ts
            try:
                for fname in sorted(os.listdir(output_dir)):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in binary_exts:
                        continue
                    fpath = os.path.join(output_dir, fname)
                    if not os.path.isfile(fpath) or os.path.getsize(fpath) <= 0:
                        continue
                    # 只返回当前会话开始后创建/修改的文件
                    if session_start > 0:
                        mtime = os.path.getmtime(fpath)
                        if mtime < session_start - 5:  # 允许5秒容差
                            continue
                    files.append(fname)
            except Exception:
                pass
            return files

        async def output_fn(text):
            if not text:
                return
            # 过滤掉原始代码和 tool_calls XML
            cleaned = _strip_code_and_tool_calls(text)
            if cleaned:
                cli_adapter.console.print(cleaned)
                if self.active_session is not None:
                    self.active_session.add_message("assistant", cleaned)
                    if self.memory is not None:
                        try:
                            await self.memory.add_message("assistant", cleaned)
                        except Exception:
                            pass
                    self._save_session()

        tool_capability_registry = None
        try:
            from long.capabilities.tool_capability import ToolCapabilityRegistry
            tool_capability_registry = ToolCapabilityRegistry()
        except ImportError:
            pass

        async def on_span_created_fn(span_data: dict[str, Any]) -> None:
            """增量推送 trace span 到前端"""
            from long.interaction.adapters.webui import WebUIAdapter
            if not isinstance(cli_adapter, WebUIAdapter):
                return
            trace = self.tracer.get_traces(limit=1)
            if not trace:
                return
            trace_data = trace[0].to_dict()
            cli_adapter.send_event(InteractionEvent(
                type=InteractionEventType.TRACE,
                content="",
                metadata={"trace": trace_data},
            ))

        runtime = CognitiveRuntime(
            llm_chat_fn=llm_chat_fn,
            llm_chat_with_tools_fn=llm_chat_with_tools_fn,
            tool_execute_fn=tool_execute_fn,
            output_fn=output_fn,
            memory_controller=self.memory,
            tool_capability_registry=tool_capability_registry,
            on_span_created=on_span_created_fn,
            plan_executor=self.plan_executor,
        )

        context = CognitiveContext(
            user_message=history_msgs[-1].get("content", "") if history_msgs else "",
            messages=list(history_msgs),
            max_rounds=8,
        )

        # 检查对话历史中是否已有相关内容，如果有则添加提示
        if len(history_msgs) > 3:
            assistant_msgs = [m for m in history_msgs if m.get("role") == "assistant" and len(m.get("content", "")) > 100]
            if assistant_msgs:
                history_hint = (
                    "\n\n## 重要提示\n对话历史中已经包含了用户之前讨论的内容。"
                    "如果用户当前请求可以基于已有内容完成（如'根据此生成PPT'），"
                    "请直接使用对话历史中的信息，不要再调用 tavily_search 搜索。"
                )
                # 追加到 system message
                for i, msg in enumerate(context.messages):
                    if msg.get("role") == "system":
                        context.messages[i] = {
                            "role": "system",
                            "content": msg.get("content", "") + history_hint,
                        }
                        break

        graph_context = {
            "_cognitive_context": context,
            "_tools": self._clean_tools_for_api(tools),
        }

        try:
            result_context = await runtime.run(
                context, extra={"_tools": self._clean_tools_for_api(tools)}
            )
            # 认知运行时完成后，扫描 output 目录是否有生成的二进制文件
            output_dir = os.path.join(self.workspace.root, "output") if self.workspace else "output"
            output_files = _scan_output_files(output_dir)
            if output_files:
                links = []
                for fname in output_files:
                    links.append(f"[{fname}](/output/{fname})")
                file_section = "\n\n📁 生成的文件：\n" + "\n".join(f"  - {link}" for link in links)
                cli_adapter.console.print(file_section)
            
            return result_context.is_complete
        except Exception as e:
            logger.warning("Cognitive Runtime 执行失败: %s，降级到 Fallback", e)
            return False

    async def _print_generated_files(self, plan: Any, cli_adapter: Any) -> str | None:
        """读取计划执行生成的报告/图表文件内容并输出到前端

        只输出当前计划步骤中明确写入的文件，不扫描整个 output 目录，
        避免将之前任务的文件混入当前输出。

        返回完整的报告内容字符串（用于保存到 session），如果没有报告则返回 None。
        """
        import os as _os

        output_dir = _os.path.join(self.workspace.root, "output") if self.workspace else "output"
        if not _os.path.isdir(output_dir):
            return None

        # 收集计划中写入的文件路径
        written_files: list[str] = []
        for step in plan.steps:
            args = step.args or {}
            action = step.action.lower() if hasattr(step, 'action') and step.action else ""
            if action == "write_file" and args.get("path"):
                written_files.append(args["path"])
            elif action == "call_tool":
                tool_name = args.get("tool_name", "")
                params = args.get("parameters", {})
                if not isinstance(params, dict):
                    params = {}
                for key, value in args.items():
                    if key not in ("tool_name", "parameters") and key not in params:
                        params[key] = value
                if tool_name == "write_file" and params.get("path"):
                    written_files.append(params["path"])

        # 同时收集 execute_file 步骤生成的文件（通过脚本路径推断输出文件）
        executed_scripts: list[str] = []
        for step in plan.steps:
            args = step.args or {}
            action = step.action.lower() if hasattr(step, 'action') and step.action else ""
            if action == "execute_file" and args.get("path"):
                executed_scripts.append(args["path"])
            elif action == "call_tool":
                tool_name = args.get("tool_name", "")
                params = args.get("parameters", {})
                if not isinstance(params, dict):
                    params = {}
                for key, value in args.items():
                    if key not in ("tool_name", "parameters") and key not in params:
                        params[key] = value
                if tool_name == "execute_file" and params.get("path"):
                    executed_scripts.append(params["path"])

        # 读取 .md 报告文件内容（仅当前计划写入的文件）
        report_parts: list[str] = []
        for file_path in written_files:
            ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
            if ext != "md":
                continue
            full_path = _os.path.join(self.workspace.root, file_path) if self.workspace and not _os.path.isabs(file_path) else file_path
            if not _os.path.exists(full_path):
                continue
            try:
                with open(full_path, encoding="utf-8") as f:
                    content = f.read()
                if content.strip():
                    import re as _re
                    content = _re.sub(
                        r'!\[([^\]]*)\]\((?!https?://|/)([^)]+)\)',
                        r'![\1](/output/\2)',
                        content,
                    )
                    report_parts.append(content)
            except Exception:
                pass

        # 收集当前计划生成的图表文件（仅 written_files 中的图片）
        chart_files: list[str] = []
        for file_path in written_files:
            ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
            if ext in ("png", "jpg", "jpeg", "svg", "gif"):
                chart_files.append(_os.path.basename(file_path))

        # 收集当前计划生成的 PPTX 文件
        pptx_files: list[str] = []
        for file_path in written_files:
            ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
            if ext == "pptx":
                pptx_files.append(_os.path.basename(file_path))

        # 如果计划步骤没找到 PPTX/图表文件，检查 execute_file 步骤的脚本可能生成的输出
        # 通过读取脚本内容推断输出文件名
        if (not pptx_files and not chart_files) and executed_scripts:
            for script_path in executed_scripts:
                full_path = _os.path.join(self.workspace.root, script_path) if self.workspace and not _os.path.isabs(script_path) else script_path
                if not _os.path.exists(full_path):
                    logger.debug("脚本文件不存在: %s", script_path)
                    continue
                try:
                    with open(full_path, encoding="utf-8") as f:
                        script_content = f.read()
                    # 从脚本中提取 save 路径
                    import re as _re
                    save_matches = _re.findall(r'\.save\([\'"]([^\'"]+\.pptx)[\'"]\)', script_content)
                    save_matches += _re.findall(r'\.save\([\'"]([^\'"]+\.png)[\'"]\)', script_content)
                    save_matches += _re.findall(r'savefig\([\'"]([^\'"]+\.png)[\'"]\s*', script_content)
                    logger.info("从脚本 %s 推断输出文件: %s", script_path, save_matches)
                    for match in save_matches:
                        basename = _os.path.basename(match)
                        ext = basename.rsplit(".", 1)[-1].lower()
                        if ext == "pptx":
                            pptx_files.append(basename)
                        elif ext in ("png", "jpg", "jpeg", "svg", "gif"):
                            chart_files.append(basename)
                        elif ext == "md":
                            # 读取推断出的 .md 文件
                            md_path = _os.path.join(output_dir, basename)
                            if _os.path.exists(md_path):
                                with open(md_path, encoding="utf-8") as f:
                                    content = f.read()
                                if content.strip():
                                    report_parts.append(content)
                except Exception:
                    pass

        combined_report = "\n\n".join(report_parts) if report_parts else ""
        for chart_name in chart_files:
            if chart_name not in combined_report:
                combined_report += f"\n\n![{chart_name}](/output/{chart_name})\n"

        # 添加 PPTX 文件下载提示
        for pptx_name in pptx_files:
            combined_report += f"\n\n📊 PPT已生成：[{pptx_name}](/output/{pptx_name})\n"

        # 组合完整报告内容并输出
        full_report = combined_report if combined_report.strip() else None

        # 如果推断失败（没有找到 PPTX/图表/报告），扫描 output 目录中当前会话的文件
        if not full_report:
            session_start = self._session_start_ts
            binary_exts = {'.pptx', '.pdf', '.docx', '.xlsx', '.xls', '.zip', '.png', '.jpg', '.jpeg', '.svg', '.gif'}
            try:
                for fname in sorted(_os.listdir(output_dir)):
                    ext = _os.path.splitext(fname)[1].lower()
                    if ext not in binary_exts:
                        continue
                    fpath = _os.path.join(output_dir, fname)
                    if not _os.path.isfile(fpath) or _os.path.getsize(fpath) <= 0:
                        continue
                    # 只返回当前会话开始后创建/修改的文件
                    if session_start > 0:
                        mtime = _os.path.getmtime(fpath)
                        if mtime < session_start - 5:
                            continue
                    # 添加下载链接
                    if ext == '.pptx':
                        combined_report += f"\n\n📊 PPT已生成：[{fname}](/output/{fname})\n"
                    elif ext in ('.docx', '.pdf'):
                        combined_report += f"\n\n📄 文档已生成：[{fname}](/output/{fname})\n"
                    elif ext in ('.xlsx', '.xls'):
                        combined_report += f"\n\n📊 表格已生成：[{fname}](/output/{fname})\n"
                    elif ext in ('.png', '.jpg', '.jpeg', '.svg', '.gif'):
                        combined_report += f"\n\n![{fname}](/output/{fname})\n"
                    else:
                        combined_report += f"\n\n📦 文件已生成：[{fname}](/output/{fname})\n"
                full_report = combined_report if combined_report.strip() else None
            except Exception:
                pass

        logger.info("_print_generated_files: pptx_files=%s, chart_files=%s, report_parts=%d, full_report=%s",
                     pptx_files, chart_files, len(report_parts), "有内容" if full_report else "空")
        if full_report:
            cli_adapter.console.print(full_report)
        return full_report

    async def _verify_plan_deliverables(
        self, plan: Any, cli_adapter: Any, exec_result: Any
    ) -> bool:
        """验证计划执行后交付物是否真实存在且内容完整

        检查：
        1. 计划中 write_file 步骤的文件是否真的被写入
        2. 生成的文件是否通过内容质量校验
        3. 是否产生了有效的输出内容（不只是缝合的中间结果）
        """
        import os as _os
        from long.eval.content_validator import content_validator

        executed_files = []  # execute_file 执行的脚本路径
        written_files = []
        has_report_step = False
        has_chart_step = False

        for step in plan.steps:
            desc = (step.description or "").lower()
            action = step.action.lower() if hasattr(step, 'action') else ""
            args = step.args or {}

            if "报告" in desc or "report" in desc:
                has_report_step = True
            if "图表" in desc or "chart" in desc or "chart" in action:
                has_chart_step = True

            # 检测 write_file 步骤的输出文件路径
            if action == "write_file" and args.get("path"):
                written_files.append(args["path"])
            elif action == "call_tool":
                # call_tool 动作：path 可能在 args 顶层或 parameters 子字典中
                tool_name = args.get("tool_name", "")
                parameters = args.get("parameters", {})
                if not isinstance(parameters, dict):
                    parameters = {}
                # 合并 args 顶层非保留键到 parameters
                for key, value in args.items():
                    if key not in ("tool_name", "parameters") and key not in parameters:
                        parameters[key] = value

                if tool_name == "write_file" and parameters.get("path"):
                    written_files.append(parameters["path"])
                elif tool_name == "execute_file" and parameters.get("path"):
                    executed_files.append(parameters["path"])
            elif action == "execute_file" and args.get("path"):
                executed_files.append(args["path"])
            if action == "output":
                out_path = args.get("path") or args.get("output_path", "")
                if out_path:
                    written_files.append(out_path)

        if not written_files and not executed_files and not exec_result.output_text:
            return True

        missing_files: list[str] = []
        bad_content_files: list[str] = []

        for file_path in written_files:
            full_path = _os.path.join(self.workspace.root, file_path) if not _os.path.isabs(file_path) else file_path
            if not _os.path.exists(full_path):
                missing_files.append(file_path)
                continue

            try:
                with open(full_path) as f:
                    file_content = f.read()
                validation = content_validator.validate(file_path, file_content)
                if not validation.passed:
                    bad_content_files.append(file_path)
                    cli_adapter.console.print(
                        f"[dim]  📄 {file_path}: {validation.format_summary()}[/dim]"
                    )
            except Exception:
                missing_files.append(file_path)

        # 检查 execute_file 步骤执行的脚本是否生成了输出文件
        import re as _content_re
        output_dir = _os.path.join(self.workspace.root, "output")

        for file_path in written_files:
            full_path = _os.path.join(self.workspace.root, file_path) if self.workspace and not _os.path.isabs(file_path) else file_path
            if not _os.path.exists(full_path):
                missing_files.append(file_path)
                continue

            # 对 .py 文件做内容质量检测：内容必须是可执行代码，不能是描述文字
            if file_path.endswith(".py"):
                try:
                    with open(full_path, encoding="utf-8") as f:
                        content = f.read()
                    if len(content.strip()) == 0:
                        logger.warning("检测到 write_file 内容为空: %s", file_path)
                        cli_adapter.console.print(
                            f"[yellow]⚠ {file_path} 内容为空（0字符），该脚本无法生成预期输出[/yellow]"
                        )
                        bad_content_files.append(file_path)
                        continue
                    content_stripped = content.lstrip("# \n\r\t")
                    has_import = bool(_content_re.search(r'(?:^|\n)\s*(?:from\s+\w+\s+import|import\s+\w+)', content))
                    has_def = bool(_content_re.search(r'(?:^|\n)\s*def\s+\w+\s*\(', content))
                    has_class = bool(_content_re.search(r'(?:^|\n)\s*class\s+\w+', content))
                    has_assign = bool(_content_re.search(r'(?:^|\n)\s*\w+\s*[=]\s*', content_stripped))
                    is_description_only = (content_stripped.startswith(("根据", "基于", "将根据", "代码将", "编写", "Python", "python"))) and not (has_import or has_def or has_class or has_assign)
                    if is_description_only:
                        logger.warning("检测到 write_file 内容为描述文字而非代码: %s", file_path)
                        cli_adapter.console.print(
                            f"[yellow]⚠ {file_path} 内容为描述文字而非可执行代码，该脚本无法生成预期输出[/yellow]"
                        )
                        bad_content_files.append(file_path)
                except Exception:
                    pass
        for script_path in executed_files:
            full_script = _os.path.join(self.workspace.root, script_path) if not _os.path.isabs(script_path) else script_path
            if not _os.path.exists(full_script):
                missing_files.append(script_path)
                cli_adapter.console.print(
                    f"[yellow]⚠ 脚本文件不存在: {script_path}[/yellow]"
                )

        # 扫描 output 目录检查是否有新增的报告/图表文件
        if (has_report_step or has_chart_step) and _os.path.isdir(output_dir):
            found_report = False
            found_chart = False
            try:
                session_start = getattr(self, '_session_start_ts', 0)
                for fname in _os.listdir(output_dir):
                    fpath = _os.path.join(output_dir, fname)
                    if not _os.path.isfile(fpath):
                        continue
                    try:
                        st = _os.stat(fpath)
                        if st.st_size == 0:
                            continue
                        if session_start > 0 and st.st_mtime < session_start:
                            continue
                    except OSError:
                        continue
                    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                    if ext in ("md", "docx", "pdf", "txt", "html", "pptx"):
                        found_report = True
                    if ext in ("png", "jpg", "jpeg", "svg", "gif"):
                        found_chart = True
                if has_report_step and found_report:
                    cli_adapter.console.print(
                        "[dim]  ✅ 检测到报告文件已生成[/dim]"
                    )
                elif has_report_step and not found_report:
                    cli_adapter.console.print(
                        "[yellow]⚠ 任务要求生成报告，但未检测到有效的报告文件[/yellow]"
                    )
                if has_chart_step and found_chart:
                    cli_adapter.console.print(
                        "[dim]  ✅ 检测到图表文件已生成[/dim]"
                    )
                elif has_chart_step and not found_chart:
                    cli_adapter.console.print(
                        "[yellow]⚠ 任务要求生成图表，但未检测到有效的图表文件[/yellow]"
                    )
            except Exception:
                pass

        if missing_files:
            cli_adapter.console.print(
                f"[yellow]⚠ 计划要求生成以下文件但未找到: {', '.join(missing_files)}[/yellow]"
            )

        if bad_content_files:
            cli_adapter.console.print(
                f"[yellow]⚠ 以下文件内容不达标: {', '.join(bad_content_files)}[/yellow]"
            )

        if (missing_files and len(missing_files) >= len(written_files)) or \
           (bad_content_files and len(bad_content_files) >= len(written_files)):
            return False

        if has_report_step and not written_files and not executed_files:
            cli_adapter.console.print(
                "[yellow]⚠ 任务要求生成报告，但计划中无 write_file 或 execute_file 步骤[/yellow]"
            )

        if has_chart_step and not found_chart:
            cli_adapter.console.print(
                "[yellow]⚠ 任务要求生成图表，但未检测到有效的图表文件 — 验证失败[/yellow]"
            )
            return False

        return True

    async def _repair_plan_deliverables(
        self,
        plan: Any,
        cli_adapter: Any,
        exec_result: Any,
        history_msgs: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> bool:
        """尝试修复验证失败的交付物

        当 write_file 步骤写入的内容质量不达标（如 placeholder、描述文字代替代码）时，
        收集上下文信息并调用 LLM 重新生成正确的文件内容，然后重新执行脚本。

        Returns:
            True 如果修复成功，False 如果修复失败
        """
        import os as _os

        output_dir = _os.path.join(self.workspace.root, "output") if self.workspace else "output"

        # 收集 write_file 步骤及其写入的文件
        write_steps: list[tuple[Any, str]] = []  # (step, file_path)
        for step in plan.steps:
            args = step.args or {}
            file_path = ""
            if step.action.lower() == "write_file" and args.get("path"):
                file_path = args["path"]
            elif step.action.lower() == "call_tool" and args.get("tool_name") == "write_file":
                params = args.get("parameters", {})
                if isinstance(params, dict):
                    file_path = params.get("path", "")
                if not file_path:
                    file_path = args.get("path", "")
            if file_path:
                write_steps.append((step, file_path))

        if not write_steps:
            return False

        # 检查哪些文件内容不达标
        failed_files: list[tuple[str, str]] = []  # (file_path, error_reason)
        for step, file_path in write_steps:
            full_path = _os.path.join(self.workspace.root, file_path) if self.workspace and not _os.path.isabs(file_path) else file_path
            if not _os.path.exists(full_path):
                failed_files.append((file_path, "文件不存在"))
                continue

            try:
                with open(full_path, encoding="utf-8") as f:
                    content = f.read()
                content_stripped = content.lstrip("# \n\r\t")

                # 检查常见问题
                if len(content) < 30:
                    failed_files.append((file_path, f"内容过短（仅{len(content)}字符）：'{content[:50]}'"))
                elif content_stripped.startswith(("placeholder", "根据", "# 根据")):
                    # 检查是否包含实际代码特征
                    has_code = bool(re.search(
                        r'(?:^|\n)\s*(?:from\s+\w+\s+import|import\s+\w+|def\s+\w+\s*\(|class\s+\w+|\w+\s*=\s*)',
                        content,
                    ))
                    if not has_code:
                        failed_files.append((file_path, "内容为描述/占位文字而非可执行代码"))
            except Exception as e:
                failed_files.append((file_path, f"读取失败：{e}"))

        if not failed_files:
            # 没有文件内容问题，检查是否有 execute_file 步骤需要重新执行
            return False

        # 收集计划上下文：目标、搜索结果、skill 文档
        plan_goal = getattr(plan, 'goal', '') or ''
        context_parts: list[str] = [f"任务目标：{plan_goal}"]

        # 优先使用对话历史作为上下文（比 exec_result 更可靠）
        if history_msgs:
            for msg in history_msgs[-10:]:  # 最近10条消息
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "tool" and content:
                    if len(content) > 8000:
                        content = content[:8000] + "\n...(内容过长已截断)"
                    context_parts.append(f"工具返回结果：\n{content}")
                elif role == "assistant" and content and len(content) > 20:
                    if len(content) > 800:
                        content = content[:800] + "\n...(内容过长已截断)"
                    context_parts.append(f"助手输出：\n{content}")

        # 补充 exec_result 中的成功步骤结果
        for sr in exec_result.step_results:
            if sr.success and sr.output and len(sr.output) > 50:
                output = sr.output
                if len(output) > 8000:
                    output = output[:8000] + "\n...(内容过长已截断)"
                step_desc = getattr(sr, 'step_id', '')
                context_parts.append(f"步骤 {step_desc} 的输出：\n{output}")

        context_summary = "\n\n".join(context_parts)

        # 收集结果步骤中的 execute_file 步骤（需要重新执行）
        execute_steps: list[Any] = []
        for step in plan.steps:
            args = step.args or {}
            if step.action.lower() == "execute_file" and args.get("path"):
                execute_steps.append(step)
            elif step.action.lower() == "call_tool" and args.get("tool_name") == "execute_file":
                params = args.get("parameters", {})
                if isinstance(params, dict) and params.get("path"):
                    execute_steps.append(step)
                elif args.get("path"):
                    execute_steps.append(step)

        # 对每个失败的文件，生成修复 prompt 并调用 LLM
        from long.llm.base import LLMMessage

        for file_path, error_reason in failed_files:
            cli_adapter.console.print(
                    f"  🔧 修复文件: {file_path} ({error_reason})"
                )

            # 构建修复上下文：找出该文件依赖的前置步骤（搜索结果、skill 文档）
            file_step = None
            for s in write_steps:
                if s[1] == file_path:
                    file_step = s[0]
                    break

            deps: list[str] = []
            if file_step:
                for dep_id in (file_step.depends_on or []):
                    for sr in exec_result.step_results:
                        if getattr(sr, 'step_id', '') == dep_id and sr.success and sr.output:
                            output = sr.output
                            if len(output) > 8000:
                                output = output[:8000] + "\n...(内容过长已截断)"
                            deps.append(f"前置步骤 {dep_id} 的输出：\n{output}")

            dep_context = "\n\n".join(deps) if deps else context_summary

            repair_prompt = (
                f"你之前生成的计划中，write_file 步骤写入了文件 {file_path}，但内容验证失败：{error_reason}\n\n"
                f"以下是任务的上下文信息：\n\n{dep_context}\n\n"
                f"请根据以上上下文，重新生成文件 {file_path} 的完整内容。"
                f"要求：\n"
                f"1. 内容必须是完整可执行的代码（.py 文件）或完整的文档内容（.md 文件）\n"
                f"2. 如果是 .py 文件，必须包含 import 语句、实际的业务逻辑和处理\n"
                f"3. 如果是生成 PPT，必须包含完整的 python-pptx 代码"
                f"  包括封面、内容页、保存输出等\n"
                f"4. 文件中的 save/write 路径必须使用 output/ 前缀（如 prs.save('output/xxx.pptx')）\n"
                f"5. 直接输出文件内容，不要加任何解释"
            )

            # 判断任务是否需要图表
            has_chart_requirement = any(
                kw in plan_goal or kw in dep_context
                for kw in ("图表", "chart", "折线图", "柱状图", "饼图", "可视化")
            )
            if has_chart_requirement:
                repair_prompt += (
                    "\n6. ⚠️ 任务要求生成图表！必须使用 matplotlib 生成图表并保存为 PNG 文件"
                    "（如 plt.savefig('output/xxx_chart.png')），然后在 Word/PPT 文档中插入该图片。"
                    "不能只生成表格，必须有 matplotlib 图表！\n"
                    "matplotlib 中文字体配置代码（必须包含）：\n"
                    "```python\n"
                    "import matplotlib; matplotlib.use('Agg')\n"
                    "import matplotlib.pyplot as plt\n"
                    "from matplotlib.font_manager import FontProperties\n"
                    "import os as _os\n"
                    "_font_path = _os.path.join(_os.getcwd(), 'font', 'simhei.ttf')\n"
                    "if _os.path.exists(_font_path):\n"
                    "    _fp = FontProperties(fname=_font_path)\n"
                    "    plt.rcParams['font.family'] = _fp.get_name()\n"
                    "plt.rcParams['axes.unicode_minus'] = False\n"
                    "```\n"
                )

            messages: list[LLMMessage] = [
                LLMMessage(role="user", content=repair_prompt),
            ]

            try:
                cli_adapter.console.print("  🤔 正在生成修复内容...")
                response = await self.llm.chat(messages, purpose="repair")
                self._llm_call_total += 1

                repaired_content = response.content.strip()
                # 去掉可能的 markdown 代码块包裹
                if repaired_content.startswith("```"):
                    lines = repaired_content.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    repaired_content = "\n".join(lines)

                # 写入修复后的内容
                full_path = _os.path.join(self.workspace.root, file_path) if self.workspace and not _os.path.isabs(file_path) else file_path
                _os.makedirs(_os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(repaired_content)

                cli_adapter.console.print(
                    f"  ✅ 已重新写入: {file_path} ({len(repaired_content)} 字符)"
                )
            except Exception as e:
                logger.warning("修复文件 %s 失败: %s", file_path, e)
                return False

        # 重新执行 execute_file 步骤
        for step in execute_steps:
            args = step.args or {}
            script_path = ""
            if step.action.lower() == "execute_file":
                script_path = args.get("path", "")
            else:
                params = args.get("parameters", {})
                if isinstance(params, dict):
                    script_path = params.get("path", "")
                if not script_path:
                    script_path = args.get("path", "")

            if not script_path:
                continue

            cli_adapter.console.print(f"  🔧 重新执行脚本: {script_path}")
            try:
                result = await self._execute_tool("execute_file", {"path": script_path})
                cli_adapter.console.print(
                    f"  → {result[:200]}{'...' if len(result) > 200 else ''}"
                )
            except Exception as e:
                logger.warning("重新执行 %s 失败: %s", script_path, e)

        # 再次验证
        return await self._verify_plan_deliverables(plan, cli_adapter, exec_result)

    async def _try_plan_execution(
        self,
        cli_adapter: Any,
        history_msgs: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> bool:
        """PlanIR 结构化计划：所有请求先生成计划，再逐步骤执行

        流程：
        1. 调用 LLM 生成 PlanIR（结构化动作序列）
        2. 编译时验证（状态机路径检查 + 类型检查 + 安全检查）
        3. 逐步骤执行（每步前运行时验证，每步后状态更新）
        4. 终态验证（LTL 规则检查）

        Returns:
            True 如果计划成功执行，False 如果降级
        """
        if self.plan_executor is None:
            return False

        user_msgs = [m for m in history_msgs if m.get("role") == "user"]
        if not user_msgs:
            return False

        user_message = user_msgs[-1].get("content", "")

        import time as _time
        _plan_start = _time.monotonic()

        # 1. 生成 PlanIR
        with cli_adapter.console.status("[bold cyan]📋 正在生成执行计划...[/bold cyan]", spinner="dots"):
            plan = None
            for _attempt in range(2):
                try:
                    plan = await asyncio.wait_for(
                        self.plan_executor.generate_plan(
                            user_message=user_message,
                            history_msgs=history_msgs,
                            available_tools=tools,
                        ),
                        timeout=180,
                    )
                except asyncio.TimeoutError:
                    self._record_llm_timeout()
                    if _attempt < 1:
                        cli_adapter.console.print("[dim]计划生成超时，重试...[/dim]")
                    else:
                        return False
                except Exception:
                    if _attempt < 1:
                        cli_adapter.console.print("[dim]计划生成失败，重试...[/dim]")
                    else:
                        return False

        if plan is None:
            return False

        _plan_elapsed = _time.monotonic() - _plan_start

        # 2. 编译时验证
        validation = self.plan_executor.constraint_validator.validate_plan(plan)
        if not validation.valid:
            logger.warning("PlanIR 编译时验证失败: %s", validation.errors)
            return False

        # 单步计划直接输出
        if len(plan.steps) <= 1:
            step = plan.steps[0]
            content = step.args.get("content", "")
            if content:
                self.active_session.add_message("assistant", content)
                cli_adapter.console.print(content)
                return True
            # 没有内容时降级到普通聊天模式，让 LLM 直接回复
            return False

        # 3. 执行 PlanIR
        cli_adapter.console.print(
            f"[bold blue]📋 执行计划: {plan.goal}[/bold blue] "
            f"[dim]({len(plan.steps)} 步, 生成耗时 {_plan_elapsed:.1f}s)[/dim]"
        )

        async def tool_executor(tool_name: str, arguments: dict[str, Any]) -> str:
            return await self._execute_tool(tool_name, arguments)

        exec_result = await self.plan_executor.execute_plan(
            plan=plan,
            cli_adapter=cli_adapter,
            tool_executor=tool_executor,
            history_msgs=history_msgs,
        )

        # 4. 处理结果
        if exec_result.success and exec_result.output_text:
            self.active_session.add_message("assistant", exec_result.output_text)
            cli_adapter.console.print()

            if self.memory is not None:
                try:
                    await self.memory.store(
                        f"assistant: {exec_result.output_text}",
                        memory_type=MemoryType.EPISODIC,
                        importance=0.5,
                    )
                except Exception:
                    pass
            self._save_session()
            self._schedule_auto_eval()
            return True

        if exec_result.success and not exec_result.output_text:
            self.active_session.add_message("assistant", "任务已完成。")
            self._save_session()
            self._schedule_auto_eval()
            return True

        if not exec_result.success:
            if exec_result.errors:
                for err in exec_result.errors:
                    cli_adapter.console.print(f"[yellow]  ⚠ {err}[/yellow]")
            return False

        return True

        user_msgs = [m for m in history_msgs if m.get("role") == "user"]
        if not user_msgs:
            return False

        user_message = user_msgs[-1].get("content", "")

        complexity = self.plan_executor.classifier.classify(user_message, tools)

        complexity_style = {
            "simple": "[green]简单[/green]",
            "moderate": "[yellow]中等[/yellow]",
            "complex": "[red]复杂[/red]",
        }
        complexity_label = complexity_style.get(complexity.level.value, complexity.level.value)

        logger.info(
            "任务复杂度: %s (score=%.1f, reasons=%s)",
            complexity.level.value,
            complexity.score,
            complexity.reasons,
        )

        if not complexity.needs_planning:
            cli_adapter.console.print(
                f"[dim]任务复杂度: {complexity_label}，使用直接工具调用模式[/dim]"
            )
            return False

        cli_adapter.console.print(
            f"[dim]任务复杂度: {complexity_label}，生成结构化执行计划...[/dim]"
        )

        with cli_adapter.console.status("[bold cyan]📋 正在生成执行计划...[/bold cyan]", spinner="dots"):
            _PLAN_MAX_RETRIES = 2
            plan = None
            for _plan_attempt in range(_PLAN_MAX_RETRIES):
                try:
                    plan = await asyncio.wait_for(
                        self.plan_executor.generate_plan(
                            user_message=user_message,
                            history_msgs=history_msgs,
                            available_tools=tools,
                            memory_controller=self.memory,
                        ),
                        timeout=300,
                    )
                    # 记录计划生成中的 LLM 调用次数（generate_plan 内部可能多次调用 LLM）
                    self._llm_call_total += 1
                    if plan is not None:
                        break
                except asyncio.TimeoutError:
                    self._record_llm_timeout()
                    if _plan_attempt < _PLAN_MAX_RETRIES - 1:
                        cli_adapter.console.print(f"[dim]计划生成超时，重试 {_plan_attempt + 2}/{_PLAN_MAX_RETRIES}...[/dim]")
                    else:
                        cli_adapter.console.print("[dim]计划生成超时，降级为直接工具调用模式[/dim]")
                        return False
                except Exception:
                    self._llm_call_total += 1
                    if _plan_attempt < _PLAN_MAX_RETRIES - 1:
                        cli_adapter.console.print(f"[dim]计划生成失败，重试 {_plan_attempt + 2}/{_PLAN_MAX_RETRIES}...[/dim]")
                    else:
                        cli_adapter.console.print("[dim]计划生成失败，降级为直接工具调用模式[/dim]")
                        return False

        if plan is None:
            cli_adapter.console.print("[dim]计划生成失败，降级为直接工具调用模式[/dim]")
            return False

        if len(plan.steps) <= 1:
            # 单步计划检查：如果单步包含有意义的代码生成动作，仍执行计划
            from long.ir.plan_ir import ActionType

            _CODE_GEN_ACTIONS = {
                ActionType.WRITE_FILE,
                ActionType.CALL_TOOL,
                ActionType.EXECUTE_FILE,
                ActionType.CALL_SKILL,
            }
            has_code_gen = any(
                step.action in _CODE_GEN_ACTIONS for step in plan.steps
            )
            if not has_code_gen:
                cli_adapter.console.print("[dim]计划仅含单步且无代码生成，使用直接工具调用模式[/dim]")
                return False

        async def tool_executor(tool_name: str, arguments: dict[str, Any]) -> str:
            return await self._execute_tool(tool_name, arguments)

        exec_result = await self.plan_executor.execute_plan(
            plan=plan,
            cli_adapter=cli_adapter,
            tool_executor=tool_executor,
            history_msgs=history_msgs,
        )

        # 占位参数被拒绝时，重试计划生成（带提示）
        if exec_result.has_placeholder_params:
            cli_adapter.console.print("[yellow]正在重新生成计划（要求包含实际代码内容）...[/yellow]")
            # 在 history_msgs 中添加重试提示
            retry_hint = (
                "上一次生成的计划中 write_file 的 content 参数包含占位描述（如'根据搜索结果编写代码'），"
                "而不是实际的可执行代码。请重新生成计划，确保 write_file 的 content 参数"
                "包含完整的、可执行的 Python 代码，不要写描述性文字。"
            )
            retry_msgs = history_msgs + [{"role": "user", "content": retry_hint}]
            try:
                retry_plan = await asyncio.wait_for(
                    self.plan_executor.generate_plan(
                        user_message=user_message,
                        history_msgs=retry_msgs,
                        available_tools=tools,
                        memory_controller=self.memory,
                    ),
                    timeout=300,
                )
                if retry_plan is not None:
                    from long.ir.plan_ir import ActionType

                    _CODE_GEN_ACTIONS = {
                        ActionType.WRITE_FILE,
                        ActionType.CALL_TOOL,
                        ActionType.EXECUTE_FILE,
                        ActionType.CALL_SKILL,
                    }
                    has_code_gen = any(
                        step.action in _CODE_GEN_ACTIONS for step in retry_plan.steps
                    )
                    if len(retry_plan.steps) > 1 or has_code_gen:
                        plan = retry_plan
                        exec_result = await self.plan_executor.execute_plan(
                            plan=plan,
                            cli_adapter=cli_adapter,
                            tool_executor=tool_executor,
                            history_msgs=history_msgs,
                        )
                        if exec_result.success:
                            cli_adapter.console.print("[green]✅ 重新生成的计划执行成功[/green]")
            except Exception:
                pass

        # 注意：不再单独打印 exec_result.output_text（通常是简短摘要），
        # 统一由 _print_generated_files 读取完整报告文件内容输出

        if exec_result.success:
            verified = await self._verify_plan_deliverables(plan, cli_adapter, exec_result)

            if not verified:
                # 先尝试自动修复失败的交付物
                cli_adapter.console.print(
                    "[bold yellow]⚠️ 计划完成但交付物验证失败，尝试自动修复...[/bold yellow]"
                )
                repaired = await self._repair_plan_deliverables(plan, cli_adapter, exec_result, history_msgs, tools)
                if repaired:
                    cli_adapter.console.print("[dim]  ✅ 自动修复成功[/dim]")
                else:
                    cli_adapter.console.print(
                        "[bold yellow]  自动修复失败，降级到认知运行时[/bold yellow]"
                    )
                    return False

            # 读取生成的报告/图表文件内容并输出到前端
            report_content = await self._print_generated_files(plan, cli_adapter)

            # 将报告内容作为最终输出保存
            if report_content:
                self.active_session.add_message("assistant", report_content)
                if self.memory is not None:
                    try:
                        await self.memory.add_message("assistant", report_content)
                    except Exception:
                        pass
                if self.memory is not None:
                    try:
                        await self.memory.store(
                            f"assistant: {report_content[:500]}",
                            memory_type=MemoryType.EPISODIC,
                            importance=0.5,
                        )
                    except Exception:
                        pass
                self._save_session()
                self._schedule_auto_eval()
                return True

        if exec_result.success and exec_result.output_text:
            if self.memory is not None:
                try:
                    await self.memory.store(
                        f"assistant: {exec_result.output_text}",
                        memory_type=MemoryType.EPISODIC,
                        importance=0.5,
                    )
                except Exception:
                    pass
            self._save_session()
            self._schedule_auto_eval()
            return True

        if exec_result.success and not exec_result.output_text:
            from long.llm.base import LLMMessage

            cli_adapter.console.print()
            gen_status = cli_adapter.console.status(
                "[bold green]✨ 正在生成回复...[/bold green]", spinner="bouncingBar"
            )
            gen_status.start()
            try:
                response_parts: list[str] = []
                first_token = True
                async for token in self.llm.stream_chat(
                    [LLMMessage(role=m["role"], content=m.get("content", ""), tool_calls=m.get("tool_calls"), tool_call_id=m.get("tool_call_id")) for m in history_msgs],
                    purpose="chat",
                ):
                    if first_token:
                        gen_status.stop()
                        first_token = False
                    response_parts.append(token)
                    cli_adapter.console.print(token, end="", highlight=False)
                response_text = "".join(response_parts)
            finally:
                gen_status.stop()

            cli_adapter.console.print()
            cli_adapter.console.print()

            self.active_session.add_message("assistant", response_text)

            if self.memory is not None:
                try:
                    await self.memory.add_message("assistant", response_text)
                except Exception:
                    pass

            if self.memory is not None:
                try:
                    await self.memory.store(
                        f"assistant: {response_text}",
                        memory_type=MemoryType.EPISODIC,
                        importance=0.5,
                    )
                except Exception:
                    pass
            self._save_session()
            self._schedule_auto_eval()
            return True

        if not exec_result.success:
            if exec_result.errors:
                for err in exec_result.errors:
                    cli_adapter.console.print(f"[yellow]  ⚠ {err}[/yellow]")
            cli_adapter.console.print("[yellow]计划执行未完成，降级到认知运行时模式...[/yellow]")
            return False

        return True

    async def _fallback_tool_call_loop(
        self, cli_adapter: Any, history_msgs: list[dict[str, str]], tools: list[dict[str, Any]]
    ) -> None:
        """降级工具调用循环 — 轻量级约束检查

        降级模式与计划模式的约束策略不同：
        - 计划模式：严格约束（状态机 + LTL + Schema + 白名单 + 预算）
        - 降级模式：轻量级约束（仅安全检查 + 预算），不检查状态机转移

        原因：降级模式下 LLM 自由决定工具调用顺序，状态机的严格转移规则
        （如 HAS_DATA 下只能 reason/summarize）会阻止合理的多步工具调用。
        """
        from long.llm.base import LLMMessage
        from long.ir.plan_ir import ActionType
        from long.cognitive.compression import SemanticCompressor

        _compressor = SemanticCompressor()

        MAX_TOOL_ROUNDS = 8
        MAX_SEARCH_ROUNDS = 2  # 严格限制搜索次数

        # 跟踪搜索调用频率，防止无限搜索
        search_call_count: int = 0
        search_queries_used: list[str] = []
        last_round_had_search: bool = False  # ReAct: 上一轮是否搜索过

        tool_source_map: dict[str, str] = {}
        mcp_server_map: dict[str, str] = {}
        for tool_entry in tools:
            func = tool_entry.get("function", {})
            name = func.get("name", "")
            source = tool_entry.get("_source", "local")
            tool_source_map[name] = source
            mcp_server = tool_entry.get("_mcp_server")
            if mcp_server:
                mcp_server_map[name] = mcp_server

        budget_remaining = MAX_TOOL_ROUNDS

        # 保存原始用户消息，用于后续 needs_tool_required 检查
        # （重试时追加的 user 消息会覆盖倒序查找的结果）
        _original_user_msg = ""
        for m in reversed(history_msgs):
            if m.get("role") == "user" and m.get("content"):
                _original_user_msg = m["content"]
                break

        for _round in range(MAX_TOOL_ROUNDS):
            # ── 展示层: 轮次分隔 ──
            round_display = _round + 1
            cli_adapter.console.print()
            cli_adapter.console.print(
                f"━━━ [bold blue]Round {round_display}[/bold blue] "
                f"[dim](最多{MAX_TOOL_ROUNDS}轮, 搜索:{search_call_count}, 预算:{MAX_TOOL_ROUNDS - budget_remaining}/{MAX_TOOL_ROUNDS})[/dim] ━━━"
            )

            if (
                self.dialog_compressor is not None
                and _round > 0
                and self.dialog_compressor.should_compress(history_msgs, tool_rounds=_round)
            ):
                history_msgs = await self.dialog_compressor.compress(
                    self.llm, history_msgs, tool_rounds=_round,
                )

            llm_messages = []
            for m in history_msgs:
                msg = LLMMessage(
                    role=m["role"],
                    content=m.get("content", ""),
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                )
                llm_messages.append(msg)

            with cli_adapter.console.status("[bold cyan]⏳ 正在思考...[/bold cyan]", spinner="dots"):
                trace = self.tracer.get_trace(current_trace_id()) if current_trace_id() else None
                span_ctx = None
                if trace is not None:
                    span_ctx = trace.span("llm.chat_with_tools", attributes={"round": _round + 1})
                    span_ctx.__enter__()

                import time as _time
                _ROUND_TIMEOUT = 150.0
                round_deadline = _time.monotonic() + _ROUND_TIMEOUT

                cli_adapter.console.print("[dim]  🤔 LLM 思考中...[/dim]", end="\r")
                _think_start = _time.monotonic()
                try:
                    response = await asyncio.wait_for(
                        self.llm.chat_with_tools(
                            llm_messages, self._clean_tools_for_api(tools),
                            purpose="chat", deadline=round_deadline,
                        ),
                        timeout=_ROUND_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    self._record_llm_timeout()
                    if span_ctx is not None:
                        from long.observability.tracing import SpanStatus
                        span_ctx._span.finish(SpanStatus.TIMEOUT)
                        span_ctx.__exit__(None, None, None)

                    # 超时时尝试用已有工具结果生成简单回复
                    tool_results_in_history = [
                        m for m in history_msgs if m["role"] == "tool"
                    ]
                    if tool_results_in_history and _round > 0:
                        logger.warning(
                            "LLM 超时 (第%d轮)，尝试用已有 %d 条工具结果生成回复",
                            _round + 1, len(tool_results_in_history),
                        )
                        fallback_msg = (
                            "抱歉，LLM 服务响应超时。以下是我已获取的信息摘要：\n\n"
                        )
                        for tr in tool_results_in_history:
                            content = tr.get("content", "")
                            if content and not content.startswith("❌"):
                                preview = content[:500].strip()
                                fallback_msg += f"```\n{preview}\n```\n\n"
                        fallback_msg += "\n请基于以上信息继续操作，或稍后重试。"
                        cli_adapter.console.print(fallback_msg)
                        if self.active_session is not None:
                            self.active_session.add_message("assistant", fallback_msg)
                            if self.memory is not None:
                                try:
                                    await self.memory.add_message("assistant", fallback_msg)
                                except Exception:
                                    pass
                            self._save_session()
                        return
                    raise
                except Exception:
                    if span_ctx is not None:
                        span_ctx.__exit__(None, None, None)
                    raise

                if span_ctx is not None:
                    span_ctx.__exit__(None, None, None)

            _think_elapsed = _time.monotonic() - _think_start
            cli_adapter.console.print(f"[dim]  🤔 LLM 思考完成 ({_think_elapsed:.1f}s)[/dim]")

            if response.tool_calls:
                _PARALLEL_SAFE_TOOLS = frozenset({
                    "read_file", "list_files", "get_current_time",
                    "read_skill_md",
                })

                parallel_calls: list[tuple[int, dict[str, Any]]] = []
                serial_calls: list[tuple[int, dict[str, Any]]] = []
                all_tool_calls: list[dict[str, Any]] = []
                intercepted_ids: set[str] = set()
                intercept_reasons: dict[str, str] = {}
                _redirected_ids: set[str] = set()

                # 检查并拦截搜索调用
                for idx, tc in enumerate(response.tool_calls):
                    tool_name = tc["name"]
                    arguments = tc["arguments"]

                    tc_def = {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tool_name, "arguments": json.dumps(arguments)},
                    }
                    all_tool_calls.append(tc_def)

                    # 跟踪并限制搜索调用
                    if tool_name == "tavily_search":
                        query = arguments.get("query", "")

                        _WEATHER_KEYWORDS = (
                            "天气", "天气预报", "气温", "温度", "湿度", "风力",
                            "下雨", "下雨吗", "降雨", "降水", "紫外线",
                            "weather", "forecast", "temperature",
                        )
                        _query_lower = query.lower()
                        _is_weather_query = any(kw in _query_lower for kw in _WEATHER_KEYWORDS)
                        if _is_weather_query:
                            _city_patterns = [
                                r'([\u4e00-\u9fff]{2,4})(?:的|市|地区)?(?:天气|气温|温度|湿度|风力|降雨|降水|天气预报|实时天气)',
                                r'(?:天气|气温|温度|湿度|风力|降雨|降水|天气预报|实时天气)([\u4e00-\u9fff]{2,4})',
                            ]
                            _city_name = ""
                            for _pat in _city_patterns:
                                _m = re.search(_pat, query)
                                if _m:
                                    _city_name = _m.group(1).strip()
                                    break
                            if not _city_name:
                                _city_name = re.sub(
                                    r'(天气|天气预报|气温|温度|湿度|风力|降雨|降水|实时|今日|今天|明天|本周|一周|未来|的|如何|怎么样|查询|搜索|\d+年?\d*月?\d*日?|[a-zA-Z]+)',
                                    '', query,
                                ).strip()
                            _city_name = re.sub(
                                r'^(今日|明天|后天|大后天|本周|下周|上周|未来|一周|实时|当前|现在|附近|周边)',
                                '', _city_name,
                            ).strip()
                            _city_name = re.sub(
                                r'(市|地区|的|省|县|区|实时|今日|今天|明天|本周|一周|未来|当前|现在|附近|周边)$',
                                '', _city_name,
                            ).strip()
                            _non_city_prefixes = ("今日", "明天", "后天", "本周", "下周", "上周", "未来", "一周", "实时", "当前", "现在")
                            while _city_name and any(_city_name.startswith(p) for p in _non_city_prefixes):
                                for p in _non_city_prefixes:
                                    if _city_name.startswith(p):
                                        _city_name = _city_name[len(p):]
                                        break
                            if _city_name:
                                logger.info(
                                    "天气查询拦截: tavily_search → query_weather, city=%s, query=%s",
                                    _city_name, query,
                                )
                                tc["name"] = "query_weather"
                                tc["arguments"] = {"city": _city_name}
                                tool_name = "query_weather"
                                arguments = {"city": _city_name}
                                tc_def["function"]["name"] = "query_weather"
                                tc_def["function"]["arguments"] = json.dumps(arguments)
                                _redirected_ids.add(tc["id"])
                                cli_adapter.console.print(
                                    f"[bold yellow]🔧 Round{_round+1}:[/bold yellow] "
                                    f"[bold cyan]query_weather(city={_city_name!r})[/bold cyan] "
                                    f"[dim](自动从 tavily_search 重定向)[/dim]"
                                )
                            else:
                                intercept_reasons[tc["id"]] = (
                                    "[天气查询拦截] 天气查询请使用 query_weather 工具，"
                                    "参数 city 传入城市名（如'杭州'）。"
                                    "不要使用 tavily_search 查询天气。"
                                )
                                intercepted_ids.add(tc["id"])
                                continue

                        if tool_name == "tavily_search" and search_call_count >= MAX_SEARCH_ROUNDS:
                            logger.warning(
                                "搜索次数已达上限 (%d/%d)，拦截搜索: %s",
                                search_call_count, MAX_SEARCH_ROUNDS, query,
                            )
                            intercept_reasons[tc["id"]] = (
                                f"[搜索限制] 已达到最大搜索次数 ({MAX_SEARCH_ROUNDS})，"
                                f"请基于已有信息生成回复，不得继续搜索。"
                            )
                            intercepted_ids.add(tc["id"])
                            continue

                        # ReAct 约束：上一轮刚搜索过，这一轮不能直接再搜索
                        # 必须先分析搜索结果（通过非搜索操作或文本回复）
                        if tool_name == "tavily_search" and last_round_had_search and search_call_count > 0:
                            logger.debug(
                                "ReAct 约束：连续搜索被拦截: %s (上一轮已搜索，需先分析结果)",
                                query,
                            )
                            intercept_reasons[tc["id"]] = (
                                f"[ReAct 约束] 你刚搜索过，请先分析已有搜索结果再决定是否需要再次搜索。"
                                f"如果结果足够，请直接使用；如果确实不足，下一轮再搜索。"
                            )
                            intercepted_ids.add(tc["id"])
                            continue

                        # 检查是否重复搜索相似内容
                        is_duplicate = False
                        if tool_name == "tavily_search":
                            for prev_query in search_queries_used:
                                if query in prev_query or prev_query in query:
                                    logger.debug(
                                        "检测到重复搜索，拦截: %s (已搜索过: %s)",
                                        query, prev_query,
                                    )
                                    intercept_reasons[tc["id"]] = (
                                        f"[搜索限制] 你已搜索过类似内容（{prev_query}），"
                                        f"请基于已有信息生成回复，不得重复搜索。"
                                    )
                                    intercepted_ids.add(tc["id"])
                                    is_duplicate = True
                                    break

                        if tool_name == "tavily_search" and not is_duplicate and tc["id"] not in intercepted_ids:
                            search_call_count += 1
                            search_queries_used.append(query)
                            logger.info(
                                "允许搜索 #%d: %s", search_call_count, query,
                            )

                    # 拦截重复的天气查询
                    if tool_name == "query_weather":
                        _weather_city = arguments.get("city", "")
                        _weather_cities = set(c.strip() for c in _weather_city.replace("，", ",").replace("、", ",").split(",") if c.strip())
                        if hasattr(self, '_weather_queried_cities'):
                            _already_queried = _weather_cities & self._weather_queried_cities
                            if _already_queried:
                                logger.info(
                                    "拦截重复天气查询: %s (已查询: %s)",
                                    _weather_city, ', '.join(_already_queried),
                                )
                                intercept_reasons[tc["id"]] = (
                                    f"[天气查询限制] 以下城市已查询过天气: {', '.join(_already_queried)}，"
                                    f"请直接使用已有数据回答，不要重复查询。"
                                )
                                intercepted_ids.add(tc["id"])
                                continue
                        else:
                            self._weather_queried_cities = set()
                        self._weather_queried_cities.update(_weather_cities)

                    if budget_remaining <= 0:
                        cli_adapter.console.print("[yellow]⚠ 预算已耗尽，停止工具调用[/yellow]")
                        intercepted_ids.add(tc["id"])
                        intercept_reasons[tc["id"]] = "[预算耗尽] 已达到最大工具调用轮次"
                        continue

                    if tool_name in ("delete_file",) and not self._check_dangerous_tool(tool_name, arguments):
                        intercepted_ids.add(tc["id"])
                        intercept_reasons[tc["id"]] = f"[安全拦截] {tool_name} 操作需要用户确认"
                        continue

                    if (tool_name not in ("tavily_search",) or tc["id"] not in intercepted_ids) and tc["id"] not in _redirected_ids:
                        # 格式化参数展示（截断超长内容，如文件内容）
                        display_args: dict[str, Any] = {}
                        for k, v in arguments.items():
                            if isinstance(v, str) and len(v) > 60:
                                display_args[k] = v[:60] + "..."
                            else:
                                display_args[k] = v
                        param_str = ", ".join(f"{k}={v!r}" for k, v in display_args.items())
                        cli_adapter.console.print(
                            f"[bold yellow]🔧 Round{_round+1}:[/bold yellow] "
                            f"[bold cyan]{tool_name}({param_str})[/bold cyan]"
                        )

                    if tool_name in _PARALLEL_SAFE_TOOLS and budget_remaining > 0:
                        parallel_calls.append((idx, tc))
                    else:
                        serial_calls.append((idx, tc))

                history_msgs.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": all_tool_calls,
                })

                # ReAct: 判断本轮是否有搜索（未被拦截的）
                current_round_has_search = any(
                    tc["name"] == "tavily_search" and tc["id"] not in intercepted_ids
                    for tc in response.tool_calls
                )

                for tc_id in intercepted_ids:
                    cli_adapter.console.print(f"  [bold yellow]🛡️ 拦截:[/bold yellow] [dim]{intercept_reasons[tc_id][:100]}[/dim]")
                    history_msgs.append({
                        "role": "tool",
                        "content": intercept_reasons[tc_id],
                        "tool_call_id": tc_id,
                    })

                if len(parallel_calls) > 1:
                    active_parallel = [(idx, tc) for idx, tc in parallel_calls if tc["id"] not in intercepted_ids]
                    if active_parallel:
                        async def _exec_one(tc: dict[str, Any]) -> tuple[str, str]:
                            return tc["id"], await self._execute_tool(tc["name"], tc["arguments"])

                        exec_start = _time.monotonic()
                        with cli_adapter.console.status("[bold yellow]🔧 并行执行工具...[/bold yellow]", spinner="line"):
                            results = await asyncio.gather(
                                *[_exec_one(tc) for _, tc in active_parallel],
                                return_exceptions=True,
                            )
                        exec_elapsed = _time.monotonic() - exec_start
                        for result in results:
                            if isinstance(result, Exception):
                                logger.warning("并行工具执行异常: %s", result)
                                continue
                            tc_id, tool_result = result
                            result_preview = tool_result[:60].replace("\n", " ")
                            cli_adapter.console.print(f"  [bold green]→[/bold green] [dim]{result_preview}...[/dim] [dim]({exec_elapsed:.1f}s)[/dim]")
                            is_error = tool_result.startswith((
                                "未知工具:", "工具执行失败:", "工具异常:", "MCP工具异常:",
                                "Skill 执行错误:", "Skill 工具", "沙箱未初始化",
                                "TAVILY_API_KEY 未配置", "搜索执行失败", "搜索超时",
                                "Tavily 搜索脚本未找到",
                            )) or any(
                                tool_result.startswith(f"{prefix} 错误:")
                                for prefix in ("list_files", "read_file", "write_file", "delete_file")
                            ) or any(
                                tool_result.startswith(f"{prefix} 失败:")
                                for prefix in ("执行文件", "读取 SKILL.md")
                            ) or tool_result.startswith("代码内容为空") or tool_result.startswith("文件路径不能为空") or tool_result.startswith("不支持执行")
                            budget_remaining -= 1
                            history_msgs.append({
                                "role": "tool",
                                "content": tool_result,
                                "tool_call_id": tc_id,
                            })
                else:
                    for _, tc in parallel_calls:
                        if tc["id"] in intercepted_ids:
                            continue
                        exec_start = _time.monotonic()
                        with cli_adapter.console.status(f"[bold yellow]🔧 执行 {tc['name']}...[/bold yellow]", spinner="line"):
                            tool_result = await self._execute_tool(tc["name"], tc["arguments"])
                        exec_elapsed = _time.monotonic() - exec_start

                        # 语义压缩工具结果（先硬截断保底，再语义压缩）
                        _MAX_TOOL_RESULT_LEN = 8000
                        if len(tool_result) > _MAX_TOOL_RESULT_LEN:
                            tool_result = tool_result[:_MAX_TOOL_RESULT_LEN] + "\n...(结果已截断)"
                        tool_result = _compressor.compress(tc["name"], tool_result)

                        is_error = tool_result.startswith((
                            "未知工具:", "工具执行失败:", "工具异常:", "MCP工具异常:",
                            "Skill 执行错误:", "Skill 工具", "沙箱未初始化",
                            "TAVILY_API_KEY 未配置", "搜索执行失败", "搜索超时",
                            "Tavily 搜索脚本未找到",
                        )) or any(
                            tool_result.startswith(f"{prefix} 错误:")
                            for prefix in ("list_files", "read_file", "write_file", "delete_file")
                        ) or any(
                            tool_result.startswith(f"{prefix} 失败:")
                            for prefix in ("执行文件", "读取 SKILL.md")
                        ) or tool_result.startswith("代码内容为空") or tool_result.startswith("文件路径不能为空") or tool_result.startswith("不支持执行")
                        if is_error:
                            cli_adapter.console.print(f"  [bold red]❌[/bold red] [dim]{tool_result[:100]}[/dim] [dim]({exec_elapsed:.1f}s)[/dim]")
                        else:
                            result_preview = tool_result[:80].replace("\n", " ")
                            cli_adapter.console.print(f"  [bold green]→[/bold green] [dim]{result_preview}...[/dim] [dim]({exec_elapsed:.1f}s)[/dim]")
                        budget_remaining -= 1
                        history_msgs.append({
                            "role": "tool",
                            "content": tool_result,
                            "tool_call_id": tc["id"],
                        })

                for _, tc in serial_calls:
                    if tc["id"] in intercepted_ids:
                        continue
                    tool_name = tc["name"]
                    arguments = tc["arguments"]

                    if budget_remaining <= 0:
                        history_msgs.append({
                            "role": "tool",
                            "content": "[预算耗尽] 已达到最大工具调用轮次",
                            "tool_call_id": tc["id"],
                        })
                        continue

                    if tool_name in ("delete_file",) and not self._check_dangerous_tool(tool_name, arguments):
                        history_msgs.append({
                            "role": "tool",
                            "content": f"[安全拦截] {tool_name} 操作需要用户确认",
                            "tool_call_id": tc["id"],
                        })
                        continue

                    exec_start = _time.monotonic()
                    with cli_adapter.console.status(f"[bold yellow]🔧 执行 {tool_name}...[/bold yellow]", spinner="line"):
                        tool_result = await self._execute_tool(tool_name, arguments)
                    exec_elapsed = _time.monotonic() - exec_start

                    # 语义压缩工具结果（先硬截断保底，再语义压缩）
                    _MAX_TOOL_RESULT_LEN = 8000
                    if len(tool_result) > _MAX_TOOL_RESULT_LEN:
                        tool_result = tool_result[:_MAX_TOOL_RESULT_LEN] + "\n...(结果已截断)"
                    tool_result = _compressor.compress(tool_name, tool_result)

                    is_error = tool_result.startswith((
                        "未知工具:", "工具执行失败:", "工具异常:", "MCP工具异常:",
                        "Skill 执行错误:", "Skill 工具", "沙箱未初始化",
                        "TAVILY_API_KEY 未配置", "搜索执行失败", "搜索超时",
                        "Tavily 搜索脚本未找到",
                    )) or any(
                        tool_result.startswith(f"{prefix} 错误:")
                        for prefix in ("list_files", "read_file", "write_file", "delete_file")
                    ) or any(
                        tool_result.startswith(f"{prefix} 失败:")
                        for prefix in ("执行文件", "读取 SKILL.md")
                    ) or tool_result.startswith("代码内容为空") or tool_result.startswith("文件路径不能为空") or tool_result.startswith("不支持执行")
                    if is_error:
                        cli_adapter.console.print(f"  [bold red]❌[/bold red] [dim]{tool_result[:100]}[/dim] [dim]({exec_elapsed:.1f}s)[/dim]")
                    else:
                        result_preview = tool_result[:80].replace("\n", " ")
                        result_len_info = f" ({len(tool_result)} 字符)" if len(tool_result) > 80 else ""
                        cli_adapter.console.print(f"  [bold green]→[/bold green] [dim]{result_preview}{'...' if len(tool_result) > 80 else ''}[/dim][dim]{result_len_info} ({exec_elapsed:.1f}s)[/dim]")
                    budget_remaining -= 1

                    history_msgs.append({
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tc["id"],
                    })

                # ReAct: 更新搜索状态
                # 只有执行了代码/写文件操作才算"分析了搜索结果"
                _CODE_TOOLS = frozenset({"write_file", "execute_code", "execute_file"})
                has_code_action = any(
                    tc["name"] in _CODE_TOOLS and tc["id"] not in intercepted_ids
                    for tc in response.tool_calls
                )
                if has_code_action:
                    last_round_had_search = False
                elif current_round_has_search:
                    last_round_had_search = True
                # else: 保持 last_round_had_search 不变
            else:
                cli_adapter.console.print()

                response_text = response.content or ""

                self._record_llm_stats(response)
                self._check_output_safety(response_text)

                _CODE_TASK_PATTERNS = (
                    "排序", "算法", "写代码", "实现", "编程", "函数", "程序",
                    "生成图表", "画图", "数据分析", "可视化", "折线图", "柱状图",
                    "快速排序", "归并排序", "冒泡排序", "桶排序", "树排序",
                    "二叉树", "链表", "哈希表", "栈", "队列",
                )
                # 这类任务必须调用工具获取数据，不能纯文本回答
                _TOOL_REQUIRED_PATTERNS = (
                    "天气", "气温", "温度", "下雨", "下雪", "weather",
                    "汇率", "股价", "股票", "基金", "比特币",
                    "新闻", "热点", "赛事", "比分",
                )
                _FABRICATED_RESULT_PATTERNS = (
                    "测试结果", "运行结果", "执行结果", "排序结果", "输出结果",
                    "程序输出", "运行输出", "测试通过", "测试成功",
                    "Output:", "Result:", "Test passed",
                )

                user_msg = _original_user_msg

                needs_code = any(p in user_msg for p in _CODE_TASK_PATTERNS)
                needs_tool_required = any(p in user_msg for p in _TOOL_REQUIRED_PATTERNS)
                has_code_tools = any(
                    m.get("role") == "tool"
                    and any(
                        kw in (m.get("content", "")[:200])
                        for kw in ("✅", "成功", "文件已保存", "写入成功", "执行完成")
                    )
                    for m in history_msgs
                )
                has_code_exec = any(
                    m.get("role") == "tool"
                    and any(
                        kw in (m.get("content", "")[:200])
                        for kw in ("执行完成", "执行成功", "execute_code", "execute_file")
                    )
                    for m in history_msgs
                )
                has_any_tool = any(m.get("role") == "tool" for m in history_msgs)

                # 需要工具获取数据的任务，LLM 没有调用任何工具
                if needs_tool_required and not has_any_tool and _round < MAX_TOOL_ROUNDS - 1:
                    logger.info("检测到需要工具的任务但 LLM 未调用任何工具，强制重试")
                    history_msgs.append({"role": "assistant", "content": response_text})
                    history_msgs.append({
                        "role": "user",
                        "content": (
                            "这个任务需要调用工具获取实时数据，你不能直接用文本回答。"
                            "请立即调用以下工具之一获取数据：\n"
                            "- execute_file: 执行已有脚本（如 skills/ 目录下的脚本），参数 path 和 args\n"
                            "- tavily_search: 搜索网络信息，参数 query\n"
                            "- read_skill_md: 读取 skill 文档，了解如何使用某个 skill\n"
                            "不要放弃，不要编造数据。必须调用工具！"
                        ),
                    })
                    continue

                if needs_code and not has_code_tools and _round < MAX_TOOL_ROUNDS - 1:
                    if not has_any_tool:
                        logger.info("检测到需要代码的任务但 LLM 未调用工具，强制重试")
                    else:
                        logger.info("检测到需要代码的任务但 LLM 未调用代码工具，强制重试")
                    history_msgs.append({"role": "assistant", "content": response_text})
                    history_msgs.append({
                        "role": "user",
                        "content": (
                            "这个任务需要写代码并执行，"
                            "使用 write_file 写入代码文件，然后用 execute_file 执行该文件（传入 path 参数，如 path='output/xxx.py'）。"
                            "不要只在文本中描述结果，必须实际执行代码。"
                            "不要调用 list_files 等无关工具。"
                        ),
                    })
                    continue

                if needs_code and has_code_tools and not has_code_exec and _round < MAX_TOOL_ROUNDS - 1:
                    logger.info("检测到需要代码的任务：代码已写入但未执行，强制重试")
                    history_msgs.append({"role": "assistant", "content": response_text})
                    history_msgs.append({
                        "role": "user",
                        "content": (
                            "你已经用 write_file 写入了代码文件，但还没有执行它。"
                            "请立即调用 execute_file 工具执行该文件（传入 path 参数）。"
                            "不要在文本中编造运行结果。"
                        ),
                    })
                    continue

                fabricated = (
                    not has_code_exec
                    and response_text
                    and any(p in response_text for p in _FABRICATED_RESULT_PATTERNS)
                    and "```" in response_text
                    and ("python" in response_text.lower() or "def " in response_text or "import " in response_text)
                )
                if fabricated:
                    logger.info("检测到 LLM 编造了不存在的执行结果，强制重试")
                    history_msgs.append({"role": "assistant", "content": response_text})
                    history_msgs.append({
                        "role": "user",
                        "content": (
                            "⚠️ 你刚才的回复包含了编造的测试/执行结果。"
                            "你并没有调用 execute_code/execute_file 工具，所以不可能有真实的执行结果。"
                            "请使用 write_file 写入代码文件，然后用 execute_file 执行该文件（传入 path 参数），"
                            "展示真实的工具返回结果。禁止编造执行结果。"
                        ),
                    })
                    continue

                if response_text:
                    cli_adapter.console.print(response_text)
                else:
                    gen_status = cli_adapter.console.status(
                        "[bold green]✨ 正在生成回复...[/bold green]", spinner="bouncingBar"
                    )
                    gen_status.start()
                    try:
                        response_parts: list[str] = []
                        first_token = True
                        async for token in self.llm.stream_chat(
                            [LLMMessage(role=m["role"], content=m.get("content", ""), tool_calls=m.get("tool_calls"), tool_call_id=m.get("tool_call_id")) for m in history_msgs],
                            purpose="chat",
                        ):
                            if first_token:
                                gen_status.stop()
                                first_token = False
                            response_parts.append(token)
                            cli_adapter.console.print(token, end="", highlight=False)
                        response_text = "".join(response_parts)
                    except Exception as e:
                        gen_status.stop()
                        logger.warning("stream_chat 失败: %s", e)
                    finally:
                        gen_status.stop()

                    # 如果 stream_chat 也没有生成内容，用工具结果生成摘要
                    if not response_text:
                        tool_results = [m for m in history_msgs if m["role"] == "tool"]
                        if tool_results:
                            response_text = "任务已完成。以下是执行结果摘要：\n\n"
                            for tr in tool_results:
                                content = tr.get("content", "")
                                if content and not content.startswith("❌"):
                                    preview = content[:300].strip()
                                    response_text += f"```\n{preview}\n```\n\n"
                            cli_adapter.console.print(response_text)

                cli_adapter.console.print()
                cli_adapter.console.print()

                self.active_session.add_message("assistant", response_text)

                if self.memory is not None:
                    try:
                        await self.memory.add_message("assistant", response_text)
                    except Exception:
                        pass

                if self.memory is not None:
                    try:
                        await self.memory.store(
                            f"assistant: {response_text}",
                            memory_type=MemoryType.EPISODIC,
                            importance=0.5,
                        )
                    except Exception:
                        pass
                self._save_session()
                self._schedule_auto_eval()
                return

        response_text = response.content or ""
        self._record_llm_stats(response)
        self._check_output_safety(response_text)
        self.active_session.add_message("assistant", response_text)
        if self.memory is not None:
            try:
                await self.memory.add_message("assistant", response_text)
            except Exception:
                pass
        cli_adapter.console.print()
        cli_adapter.console.print(response_text)
        cli_adapter.console.print()
        self._save_session()
        self._schedule_auto_eval()

    def _check_dangerous_tool(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """检查危险工具是否允许执行（降级模式下的安全检查）

        Returns:
            True 允许执行，False 拦截
        """
        high_risk_tools = frozenset({"delete_file"})
        if tool_name in high_risk_tools:
            if self._security_mode == "service":
                return False
        return True

    def _build_virtual_step_args(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_source_map: dict[str, str],
        mcp_server_map: dict[str, str],
    ) -> tuple[str, dict[str, Any]]:
        """根据工具来源构建正确的 ActionType 和参数

        Returns:
            (action_type, step_args) 元组
        """
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
    def _map_tool_to_action(tool_name: str) -> str:
        """将工具名映射到 ActionType（仅用于无 _source 信息时的降级）"""
        from long.ir.plan_ir import ActionType

        tool_action_map = {
            "list_files": ActionType.SEARCH.value,
            "read_file": ActionType.SEARCH.value,
            "read_skill_md": ActionType.SEARCH.value,
            "write_file": ActionType.CALL_TOOL.value,
            "delete_file": ActionType.CALL_TOOL.value,
            "execute_code": ActionType.CALL_TOOL.value,
            "execute_file": ActionType.CALL_TOOL.value,
        }

        if tool_name in tool_action_map:
            return tool_action_map[tool_name]

        if "_" in tool_name:
            parts = tool_name.split("_", 1)
            if len(parts) == 2:
                return ActionType.CALL_MCP.value

        return ActionType.CALL_TOOL.value

    async def _chat_stream(self, cli_adapter: Any, history_msgs: list[dict[str, str]]) -> None:
        """流式聊天（无工具）"""
        from long.llm.base import LLMMessage

        messages = [LLMMessage(role=m["role"], content=m["content"], tool_calls=m.get("tool_calls"), tool_call_id=m.get("tool_call_id")) for m in history_msgs]

        gen_status = cli_adapter.console.status(
            "[bold green]✨ 正在生成回复...[/bold green]", spinner="bouncingBar"
        )
        gen_status.start()
        try:
            full: list[str] = []
            first_token = True
            async for token in self.llm.stream_chat(messages, purpose="chat"):
                if first_token:
                    gen_status.stop()
                    first_token = False
                full.append(token)
                cli_adapter.console.print(token, end="", highlight=False)
        finally:
            if first_token:
                gen_status.stop()

        if full:
            response_text = "".join(full)
            self.active_session.add_message("assistant", response_text)
            if self.memory is not None:
                try:
                    await self.memory.add_message("assistant", response_text)
                except Exception:
                    pass
            cli_adapter.console.print()
            cli_adapter.console.print()

            if self.memory is not None:
                try:
                    await self.memory.store(
                        f"assistant: {response_text}",
                        memory_type=MemoryType.EPISODIC,
                        importance=0.5,
                    )
                except Exception:
                    pass
            self._save_session()
            self._schedule_auto_eval()
        else:
            cli_adapter.console.print("[dim](模型返回了空响应)[/dim]")


def load_dotenv(env_path: str | Path | None = None) -> None:
    """加载 .env 文件到环境变量（不覆盖已存在的环境变量）"""
    if env_path is None:
        env_path = Path.cwd() / ".env"
    else:
        env_path = Path(env_path)

    if not env_path.exists():
        logger.debug(".env 文件不存在: %s", env_path)
        return

    count = 0
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            os.environ.setdefault(key, value)
            count += 1

    logger.info("已加载 .env 文件: %s (%d 个变量)", env_path, count)


def main() -> None:
    """CLI 主入口"""
    parser = argparse.ArgumentParser(description="可控AI智能系统 (Long)")
    parser.add_argument(
        "--mode",
        choices=["cli", "webui"],
        default="cli",
        help="交互模式 (默认: cli)",
    )
    parser.add_argument(
        "--workspace",
        default="./workspace",
        help="工作区根目录 (默认: ./workspace)",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="配置文件目录 (默认: CWD/configs，不存在则回退到包 configs/)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="WebUI 监听地址 (默认: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="WebUI 监听端口 (默认: 8000)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)",
    )

    args = parser.parse_args()

    # 配置目录：优先使用显式指定的路径；否则回退到 CWD/configs；都不存在则用包内 configs/
    _package_configs = Path(__file__).resolve().parent.parent.parent / "configs"
    if args.config_dir is None:
        args.config_dir = "configs" if Path("configs").exists() else str(_package_configs)

    log_level = getattr(logging, args.log_level)

    log_dir = Path(args.workspace) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "long.log"

    setup_structured_logging(
        log_file=str(log_file),
        level=log_level,
        max_bytes=50 * 1024 * 1024,
        backup_count=5,
        redact=True,
    )

    # 搜索 .env 文件：优先当前目录，再回退到包根目录
    _env_path = Path(".env")
    if not _env_path.exists():
        _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(env_path=str(_env_path) if _env_path.exists() else None)

    system = LongSystem(
        config_dir=args.config_dir,
        workspace_root=args.workspace,
    )

    try:
        if args.mode == "cli":
            asyncio.run(system.run_cli())
        elif args.mode == "webui":
            asyncio.run(system.run_webui(host=args.host, port=args.port))
    except KeyboardInterrupt:
        pass
    finally:
        system.shutdown()


if __name__ == "__main__":
    main()
