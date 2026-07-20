"""Worker Agent

专业执行 Agent — 携带极简工具集，独立 Think-Act-Observe 循环。
使用 FAST tier 模型，职责单一、Prompt 极短。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from long.llm.base import LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class WorkerResult:
    """Worker 执行结果"""

    output: str = ""
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    tokens_used: int = 0
    elapsed_ms: float = 0.0
    success: bool = False
    error: str = ""


class WorkerAgent:
    """专业执行 Agent — 携带极简工具集，独立循环

    使用 FAST tier 模型，职责单一、Prompt 极短。
    独立的 Think → Act → Observe 循环：

    1. Think: LLM 决策（只看到自己有限工具集中的工具）
    2. Act: 执行工具调用
    3. Observe: 记录结果并进入下一轮
    """

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, Any],
        llm_chat_fn: Any,
        llm_chat_with_tools_fn: Any,
        model: str = "",
        max_rounds: int = 5,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ) -> None:
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_handlers = tool_handlers
        self._chat_fn = llm_chat_fn
        self._chat_with_tools_fn = llm_chat_with_tools_fn
        self.model = model
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def execute(self, instruction: str, context: dict[str, Any] | None = None) -> WorkerResult:
        """独立执行 Think → Act → Observe 循环

        Args:
            instruction: 子任务指令
            context: 可选上下文信息

        Returns:
            WorkerResult: 执行结果
        """
        t_start = time.monotonic()
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=self.system_prompt),
            LLMMessage(role="user", content=instruction),
        ]
        tool_history: list[dict[str, Any]] = []
        total_tokens = 0

        for round_idx in range(self.max_rounds):
            try:
                response = await asyncio.wait_for(
                    self._chat_with_tools_fn(
                        messages,
                        self.tools,
                        purpose="chat",
                        model=self.model or "",
                        max_tokens=self.max_tokens,
                        temperature=0.3,
                    ),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                return WorkerResult(
                    output=f"任务超时 ({self.timeout}s, round={round_idx + 1})",
                    tool_history=tool_history,
                    rounds=round_idx + 1,
                    tokens_used=total_tokens,
                    elapsed_ms=(time.monotonic() - t_start) * 1000,
                    success=False,
                    error="timeout",
                )
            except Exception as exc:
                return WorkerResult(
                    output=f"执行异常: {exc}",
                    tool_history=tool_history,
                    rounds=round_idx + 1,
                    tokens_used=total_tokens,
                    elapsed_ms=(time.monotonic() - t_start) * 1000,
                    success=False,
                    error=str(exc),
                )

            if response.usage:
                total_tokens += response.usage.total_tokens

            messages.append(LLMMessage(
                role="assistant",
                content=response.content or None,
                tool_calls=response.tool_calls,
            ))

            if response.tool_calls:
                tool_results: list[dict[str, Any]] = []
                for tc in response.tool_calls:
                    tool_name = tc.get("function", {}).get("name", "")
                    tool_args_str = tc.get("function", {}).get("arguments", "{}")
                    try:
                        import json
                        tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                    except json.JSONDecodeError:
                        tool_args = {}

                    handler = self.tool_handlers.get(tool_name)
                    if handler:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                result = await handler(**tool_args)
                            else:
                                result = handler(**tool_args)
                        except Exception as exc:
                            result = f"工具执行错误: {exc}"
                    else:
                        result = f"工具 '{tool_name}' 不可用"

                    tool_results.append({
                        "tool_call_id": tc.get("id", ""),
                        "name": tool_name,
                        "args": tool_args,
                        "result": str(result),
                    })
                    tool_history.append(tool_results[-1])

                    messages.append(LLMMessage(
                        role="tool",
                        content=str(result),
                        tool_call_id=tc.get("id", ""),
                    ))

                has_final_text = bool(response.content and response.content.strip())
                if not has_final_text and not response.tool_calls:
                    break
                if has_final_text and not self.tools:
                    elapsed = (time.monotonic() - t_start) * 1000
                    return WorkerResult(
                        output=response.content or str(tool_results[-1]["result"]) if tool_results else "",
                        tool_history=tool_history,
                        rounds=round_idx + 1,
                        tokens_used=total_tokens,
                        elapsed_ms=elapsed,
                        success=True,
                    )
            else:
                elapsed = (time.monotonic() - t_start) * 1000
                return WorkerResult(
                    output=response.content or "",
                    tool_history=tool_history,
                    rounds=round_idx + 1,
                    tokens_used=total_tokens,
                    elapsed_ms=elapsed,
                    success=True,
                )

        elapsed = (time.monotonic() - t_start) * 1000
        return WorkerResult(
            output="达到最大轮次限制",
            tool_history=tool_history,
            rounds=self.max_rounds,
            tokens_used=total_tokens,
            elapsed_ms=elapsed,
            success=False,
            error="max_rounds",
        )