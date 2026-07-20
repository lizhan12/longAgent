"""LLM 中间件管道

提供可组合的 I/O 中间件机制：
- pre_process: 模型输入前预处理（如 PII 过滤、动态 Prompt 组装）
- post_process: 模型输出后检查（如 PII 脱敏、安全审计、格式校验）

设计灵感来自 Web 框架的 Middleware Pipeline 模式。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable

from .base import LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class Middleware(ABC):
    """LLM I/O 中间件基类"""

    @abstractmethod
    async def pre_process(self, messages: list[LLMMessage]) -> list[LLMMessage]:
        """输入前处理

        Args:
            messages: 原始消息列表

        Returns:
            处理后的消息列表
        """

    @abstractmethod
    async def post_process(self, response: LLMResponse) -> LLMResponse:
        """输出后处理

        Args:
            response: LLM 原始响应

        Returns:
            处理后的响应
        """


class MiddlewarePipeline:
    """LLM 请求/响应处理中间件管道

    使用方式:
        pipeline = MiddlewarePipeline()
        pipeline.add(PIIFilterMiddleware())
        pipeline.add(SafetyFilterMiddleware())

        response = await pipeline.wrap(
            call_fn=lambda msgs: llm.chat(msgs),
            messages=messages,
        )
    """

    def __init__(self) -> None:
        self._middlewares: list[Middleware] = []

    def add(self, middleware: Middleware) -> MiddlewarePipeline:
        self._middlewares.append(middleware)
        return self

    async def pre_process(self, messages: list[LLMMessage]) -> list[LLMMessage]:
        for mw in self._middlewares:
            try:
                messages = await mw.pre_process(messages)
            except Exception:
                logger.exception("中间件 pre_process 失败: %s", type(mw).__name__)
        return messages

    async def post_process(self, response: LLMResponse) -> LLMResponse:
        for mw in reversed(self._middlewares):
            try:
                response = await mw.post_process(response)
            except Exception:
                logger.exception("中间件 post_process 失败: %s", type(mw).__name__)
        return response

    async def wrap(
        self,
        call_fn: Callable[[list[LLMMessage]], Awaitable[LLMResponse]],
        messages: list[LLMMessage],
    ) -> LLMResponse:
        messages = await self.pre_process(messages)
        response = await call_fn(messages)
        response = await self.post_process(response)
        return response


class PIIFilterMiddleware(Middleware):
    """PII 过滤中间件

    输入侧: 检测消息中是否包含 PII，记录警告
    输出侧: 将 PII 实例替换为 [REDACTED]
    """

    def __init__(self) -> None:
        from long.harness.output_guard import OutputGuard, OutputGuardConfig

        self._guard = OutputGuard(OutputGuardConfig(block_on_pii=True))

    async def pre_process(self, messages: list[LLMMessage]) -> list[LLMMessage]:
        for m in messages:
            if m.content and m.role == "user":
                result = self._guard.check(m.content)
                if not result.passed:
                    logger.warning("PII 检测: 用户输入包含敏感信息 (%s)", [p.type for p in result.pii_matches])
        return messages

    async def post_process(self, response: LLMResponse) -> LLMResponse:
        if response.content:
            result = self._guard.check(response.content)
            if result.pii_matches:
                content = response.content
                for match in reversed(result.pii_matches):
                    content = content[:match.start] + "[REDACTED]" + content[match.end:]
                response.content = content
        return response


class SafetyFilterMiddleware(Middleware):
    """安全过滤中间件

    输入侧: 检测 Prompt Injection
    输出侧: 检测危险代码模式
    """

    def __init__(self) -> None:
        pass

    async def pre_process(self, messages: list[LLMMessage]) -> list[LLMMessage]:
        from long.cognitive.modes.safety_boundary import SafetyBoundary

        for m in messages:
            if m.content and m.role == "user":
                safe, reason = SafetyBoundary.check_prompt_injection(m.content)
                if not safe:
                    logger.warning("Prompt Injection 检测: %s", reason)
        return messages

    async def post_process(self, response: LLMResponse) -> LLMResponse:
        if response.content:
            safe, reason = self._check_output_safety(response.content)
            if not safe:
                logger.warning("输出安全检测: %s", reason)
        return response

    @staticmethod
    def _check_output_safety(content: str) -> tuple[bool, str]:
        from long.cognitive.modes.safety_boundary import SafetyBoundary

        safe, reason = SafetyBoundary.check_code_safety(content)
        if not safe:
            return False, reason
        return True, ""


_DEFAULT_MIDDLEWARE_CLASSES: list[type[Middleware]] = [
    PIIFilterMiddleware,
    SafetyFilterMiddleware,
]


def create_default_pipeline() -> MiddlewarePipeline:
    pipeline = MiddlewarePipeline()
    for cls in _DEFAULT_MIDDLEWARE_CLASSES:
        pipeline.add(cls())
    return pipeline