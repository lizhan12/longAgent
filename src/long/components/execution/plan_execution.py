from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from long.llm.base import LLMMessage
from long.memory.base import MemoryType

logger = logging.getLogger(__name__)


class PlanExecution:

    def __init__(
        self,
        *,
        llm: Any,
        plan_executor: Any,
        constraint_validator: Any,
        state_machine: Any,
        ir_parser: Any,
        type_checker: Any,
        ltl_validator: Any,
        tool_manager: Any,
        tracer: Any,
        configs: dict[str, Any],
        memory: Any = None,
        active_session_getter: Callable[[], Any] | None = None,
        session_manager_getter: Callable[[], Any] | None = None,
        memory_bridge_getter: Callable[[], Any] | None = None,
    ) -> None:
        self.llm = llm
        self.plan_executor = plan_executor
        self.constraint_validator = constraint_validator
        self.state_machine = state_machine
        self.ir_parser = ir_parser
        self.type_checker = type_checker
        self.ltl_validator = ltl_validator
        self.tool_manager = tool_manager
        self.tracer = tracer
        self.configs = configs
        self.memory = memory
        self._active_session_getter = active_session_getter
        self._session_manager_getter = session_manager_getter
        self._memory_bridge_getter = memory_bridge_getter

    @property
    def active_session(self) -> Any:
        getter = self._active_session_getter
        return getter() if getter is not None else None

    def _save_session(self) -> None:
        getter = self._session_manager_getter
        if getter is not None:
            sm = getter()
            if sm is not None:
                sm.save_session()

    def _schedule_auto_eval(self) -> None:
        getter = self._memory_bridge_getter
        if getter is not None:
            mb = getter()
            if mb is not None:
                mb.schedule_auto_eval()

    def _record_llm_timeout(self) -> None:
        getter = self._memory_bridge_getter
        if getter is not None:
            mb = getter()
            if mb is not None:
                mb.record_llm_timeout()

    def classify_complexity(self, user_message: str, tools: list[dict[str, Any]]) -> Any:
        return self.plan_executor.classifier.classify(user_message, tools)

    async def generate_plan(
        self,
        user_message: str,
        history_msgs: list[dict[str, str]],
        tools: list[dict[str, Any]],
        cli_adapter: Any,
    ) -> Any:
        _PLAN_MAX_RETRIES = 2
        plan = None
        for _plan_attempt in range(_PLAN_MAX_RETRIES):
            try:
                plan = await asyncio.wait_for(
                    self.plan_executor.generate_plan(
                        user_message=user_message,
                        history_msgs=history_msgs,
                        available_tools=tools,
                    ),
                    timeout=180,
                )
                if plan is not None:
                    break
            except asyncio.TimeoutError:
                self._record_llm_timeout()
                if _plan_attempt < _PLAN_MAX_RETRIES - 1:
                    cli_adapter.console.print(f"[dim]计划生成超时，重试 {_plan_attempt + 2}/{_PLAN_MAX_RETRIES}...[/dim]")
                else:
                    cli_adapter.console.print("[dim]计划生成超时，降级为直接工具调用模式[/dim]")
                    return None
            except Exception:
                if _plan_attempt < _PLAN_MAX_RETRIES - 1:
                    cli_adapter.console.print(f"[dim]计划生成失败，重试 {_plan_attempt + 2}/{_PLAN_MAX_RETRIES}...[/dim]")
                else:
                    cli_adapter.console.print("[dim]计划生成失败，降级为直接工具调用模式[/dim]")
                    return None
        return plan

    async def execute_plan(
        self,
        plan: Any,
        cli_adapter: Any,
        history_msgs: list[dict[str, str]],
    ) -> Any:
        async def tool_executor(tool_name: str, arguments: dict[str, Any]) -> str:
            return await self.tool_manager.execute_tool(tool_name, arguments)

        return await self.plan_executor.execute_plan(
            plan=plan,
            cli_adapter=cli_adapter,
            tool_executor=tool_executor,
            history_msgs=history_msgs,
        )

    async def handle_plan_result(
        self,
        exec_result: Any,
        cli_adapter: Any,
        history_msgs: list[dict[str, str]],
    ) -> bool:
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
            cli_adapter.console.print("[dim]计划执行未完成，尝试直接工具调用...[/dim]")
            return False

        return True

    async def run(
        self,
        cli_adapter: Any,
        history_msgs: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> bool:
        if self.plan_executor is None:
            return False

        user_msgs = [m for m in history_msgs if m.get("role") == "user"]
        if not user_msgs:
            return False

        user_message = user_msgs[-1].get("content", "")

        complexity = self.classify_complexity(user_message, tools)

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
            plan = await self.generate_plan(user_message, history_msgs, tools, cli_adapter)

        if plan is None:
            cli_adapter.console.print("[dim]计划生成失败，降级为直接工具调用模式[/dim]")
            return False

        if len(plan.steps) <= 1:
            cli_adapter.console.print("[dim]计划仅含单步，使用直接工具调用模式[/dim]")
            return False

        exec_result = await self.execute_plan(plan, cli_adapter, history_msgs)

        return await self.handle_plan_result(exec_result, cli_adapter, history_msgs)
