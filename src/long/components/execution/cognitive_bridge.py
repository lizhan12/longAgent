from __future__ import annotations

import logging
from typing import Any, Callable

from long.llm.base import LLMMessage

logger = logging.getLogger(__name__)


class CognitiveBridge:

    def __init__(
        self,
        *,
        llm: Any,
        memory: Any,
        configs: dict[str, Any],
        tool_manager: Any = None,
        active_session_getter: Callable[[], Any] | None = None,
        session_manager_getter: Callable[[], Any] | None = None,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.configs = configs
        self.tool_manager = tool_manager
        self._active_session_getter = active_session_getter
        self._session_manager_getter = session_manager_getter

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

    @staticmethod
    def to_llm_messages(messages: list[dict[str, Any]]) -> list:
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

    async def run(
        self,
        cli_adapter: Any,
        history_msgs: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> bool:
        try:
            from long.cognitive.runtime import (
                CognitiveRuntime, CognitiveContext,
            )
        except ImportError:
            logger.warning("Cognitive Runtime 不可用，降级到 Fallback 模式")
            return False

        async def llm_chat_fn(messages, purpose="chat"):
            llm_msgs = self.to_llm_messages(messages)
            return await self.llm.chat(llm_msgs, purpose=purpose)

        async def llm_chat_with_tools_fn(messages, tools_list, purpose="chat", **kwargs):
            llm_msgs = self.to_llm_messages(messages)
            return await self.llm.chat_with_tools(llm_msgs, tools_list, purpose=purpose, **kwargs)

        async def tool_execute_fn(tool_name, arguments):
            return await self.tool_manager.execute_tool(tool_name, arguments)

        async def output_fn(text):
            if text:
                cli_adapter.console.print(text)
                if self.active_session is not None:
                    self.active_session.add_message("assistant", text)
                    self._save_session()

        tool_capability_registry = None
        try:
            from long.capabilities.tool_capability import ToolCapabilityRegistry
            tool_capability_registry = ToolCapabilityRegistry()
        except ImportError:
            pass

        runtime = CognitiveRuntime(
            llm_chat_fn=llm_chat_fn,
            llm_chat_with_tools_fn=llm_chat_with_tools_fn,
            tool_execute_fn=tool_execute_fn,
            output_fn=output_fn,
            memory_controller=self.memory,
            tool_capability_registry=tool_capability_registry,
        )

        context = CognitiveContext(
            user_message=history_msgs[-1].get("content", "") if history_msgs else "",
            messages=list(history_msgs),
            max_rounds=8,
        )

        try:
            result_context = await runtime.run(
                context, extra={"_tools": self.tool_manager.clean_tools_for_api(tools)}
            )
            return result_context.is_complete
        except Exception as e:
            logger.warning("Cognitive Runtime 执行失败: %s，降级到 Fallback", e)
            return False
