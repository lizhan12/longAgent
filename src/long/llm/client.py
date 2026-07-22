"""LLMClient — 基于 OpenAI SDK 的统一模型调用入口

集成应用层重试、Fallback 模型降级、异常分类和全链路追踪。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from long.errors import (
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    classify_openai_error,
)
from long.llm.base import (
    FallbackConfig,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMUsage,
    ModelProvider,
    RetryConfig,
)
from long.llm.middleware import MiddlewarePipeline, create_default_pipeline
from long.observability.tracing import current_trace, current_trace_id, current_span_id
from long.retry import LLM_RETRY_POLICY, RetryPolicy, retry
from long.cache import LLMResponseCache

logger = logging.getLogger(__name__)


class LLMClient:
    """统一的 LLM 调用客户端

    基于 OpenAI SDK，兼容 OpenAI / OpenRouter / Anthropic / 自定义端点。
    内置预算控制、应用层重试、Fallback 模型降级。
    """

    def __init__(self, config: LLMConfig | dict[str, Any] | None = None) -> None:
        if isinstance(config, dict):
            self._config = LLMConfig(**config)
        elif config is None:
            self._config = LLMConfig()
        else:
            self._config = config

        self._daily_tokens_used: int = 0
        self._daily_tokens_date: str = ""  # 用于自动重置日预算的日期标记
        self._task_tokens_used: int = 0
        self._request_count: int = 0

        self._retry_policy = RetryPolicy(
            max_attempts=self._config.retry.max_retries,
            base_delay=self._config.retry.base_delay,
            max_delay=self._config.retry.max_delay,
            backoff_factor=self._config.retry.backoff_factor,
        )

        self._fallback_models: list[str] = []
        if self._config.fallback.enabled and self._config.fallback.chain:
            self._fallback_models = list(self._config.fallback.chain)

        self._response_cache = LLMResponseCache(
            ttl=float(self._config.cache.ttl),
            max_size=self._config.cache.max_size,
        )
        self._cached_api_key: str | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._middleware_pipeline: MiddlewarePipeline | None = None

        api_key = self._cached_api_key
        if api_key is None:
            api_key = self._cached_api_key or self._config.resolve_api_key()
            self._cached_api_key = api_key
        base_url = self._config.resolve_base_url()

        import httpx

        http_client_kwargs: dict[str, Any] = {
            "http2": False,
            "timeout": httpx.Timeout(
                connect=float(self._config.timeout.connect),
                read=float(self._config.timeout.read),
                write=float(self._config.timeout.write),
                pool=30.0,
            ),
            "limits": httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=60.0,
            ),
        }
        proxy_url = self._resolve_proxy()
        if proxy_url:
            http_client_kwargs["proxy"] = proxy_url
            logger.info("使用代理: %s", proxy_url)

        self._http_client = httpx.AsyncClient(**http_client_kwargs)

        self._client = AsyncOpenAI(
            api_key=api_key or "sk-placeholder",
            base_url=base_url or None,
            max_retries=0,
            http_client=self._http_client,
        )

        logger.info(
            "LLM 客户端初始化: provider=%s, model=%s, base_url=%s, fallback=%s",
            self._config.provider.value,
            self._config.model,
            base_url or "(默认)",
            self._fallback_models,
        )

    @property
    def config(self) -> LLMConfig:
        return self._config

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def enable_middleware(self, pipeline: MiddlewarePipeline | None = None) -> MiddlewarePipeline:
        if pipeline is None:
            pipeline = create_default_pipeline()
        self._middleware_pipeline = pipeline
        return pipeline

    async def aclose(self) -> None:
        """关闭底层 httpx.AsyncClient 连接池"""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    def _resolve_proxy(self) -> str:
        proxy = self._config.proxy.https_proxy or self._config.proxy.http_proxy
        if proxy:
            proxy = self._resolve_env_var(proxy)
            if proxy:
                return proxy
        return ""

    def _resolve_env_var(self, value: str) -> str:
        if not value:
            return ""
        if value.startswith("${") and value.endswith("}"):
            inner = value[2:-1]
            if ":" in inner:
                var_name, default = inner.split(":", 1)
                import os
                return os.environ.get(var_name, default)
            else:
                import os
                return os.environ.get(inner, "")
        return value

    async def chat(
        self,
        messages: list[LLMMessage],
        purpose: str = "chat",
        *,
        deadline: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """发送聊天请求（含应用层重试 + Fallback + 缓存）"""
        model_config = self._config.get_model_config(purpose)
        model = kwargs.get("model", model_config.model)
        temperature = kwargs.get("temperature", model_config.temperature)

        cached = self._response_cache.get(messages, model, temperature)
        if cached is not None:
            return cached

        response = await self._call_with_retry_and_fallback(
            self._chat_impl, messages, purpose, deadline=deadline, **kwargs
        )

        self._response_cache.put(messages, response, model, temperature)
        return response

    async def _chat_impl(
        self,
        messages: list[LLMMessage],
        purpose: str = "chat",
        **kwargs: Any,
    ) -> LLMResponse:
        """chat 的实际实现（流式）"""
        model_config = self._config.get_model_config(purpose)
        model = kwargs.pop("model", model_config.model)
        logger.debug("_chat_impl: purpose=%s model=%s model_config.model=%s self._config.model=%s", purpose, model, model_config.model, self._config.model)
        temperature = kwargs.pop("temperature", model_config.temperature)
        max_tokens = kwargs.pop("max_tokens", model_config.max_tokens)
        top_p = kwargs.pop("top_p", model_config.top_p)

        self._check_budget(max_tokens)

        api_key = self._cached_api_key or self._config.resolve_api_key()
        if not api_key:
            raise LLMError(
                f"未配置 API 密钥。请设置环境变量或配置 llm.api_key。"
                f"当前提供商: {self._config.provider.value}",
                model=model,
                provider=self._config.provider.value,
                trace_id=current_trace_id(),
                span_id=current_span_id(),
            )

        if self._middleware_pipeline:
            messages = await self._middleware_pipeline.pre_process(messages)

        sdk_messages = self._build_sdk_messages(messages)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "LLM chat request: model=%s, msgs=%d, purpose=%s",
                model, len(sdk_messages), purpose,
            )

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": sdk_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True,
        }

        response_format = kwargs.get("response_format") or model_config.response_format
        if response_format:
            create_kwargs["response_format"] = response_format

        logger.debug(
            "LLM chat request (stream): model=%s, messages=%d",
            model,
            len(sdk_messages),
        )

        import time as _time
        start_time = _time.monotonic()

        stream_timeout = float(self._config.timeout.read)

        try:
            stream = await asyncio.wait_for(
                self._client.chat.completions.create(**create_kwargs),
                timeout=stream_timeout,
            )
        except asyncio.TimeoutError:
            raise LLMTimeoutError(
                f"流式请求建立超时 ({stream_timeout}s)",
                model=model,
                timeout=stream_timeout,
                trace_id=current_trace_id(),
                span_id=current_span_id(),
            )
        except Exception as exc:
            elapsed = _time.monotonic() - start_time
            logger.warning(
                "LLM API 调用失败: model=%s, elapsed=%.1fs, error=%s: %s",
                model, elapsed, type(exc).__name__, str(exc)[:200],
            )
            raise classify_openai_error(exc) from exc

        content_parts: list[str] = []
        finish_reason = ""
        usage = None
        _FIRST_CHUNK_TIMEOUT = 120.0
        _CHUNK_IDLE_TIMEOUT = 60.0
        first_chunk_received = False
        loop_start = _time.monotonic()

        try:
            aiter = stream.__aiter__()
            while True:
                current_timeout = _FIRST_CHUNK_TIMEOUT if not first_chunk_received else _CHUNK_IDLE_TIMEOUT
                try:
                    chunk = await asyncio.wait_for(aiter.__anext__(), timeout=current_timeout)
                except StopAsyncIteration:
                    break

                if not first_chunk_received:
                    first_chunk_received = True
                    elapsed = _time.monotonic() - loop_start
                    logger.debug(
                        "_chat_impl first chunk: model=%s, latency=%.1fs",
                        model, elapsed,
                    )
                if chunk.usage:
                    usage = chunk.usage
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta is not None:
                        if delta.content:
                            content_parts.append(delta.content)
                        elif getattr(delta, "reasoning", None):
                            # 兼容 SenseNova 等 API：内容放在 reasoning 而非 content 字段
                            content_parts.append(getattr(delta, "reasoning", ""))
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
        except asyncio.TimeoutError:
            elapsed = _time.monotonic() - loop_start
            detail = "首 token 超时" if not first_chunk_received else "流式读取空闲超时"
            raise LLMTimeoutError(
                f"LLM 流式响应超时: {detail} ({elapsed:.0f}s, 空闲阈值 {current_timeout:.0f}s)",
                model=model,
                timeout=current_timeout,
                trace_id=current_trace_id(),
                span_id=current_span_id(),
            )
        except LLMTimeoutError:
            raise
        except Exception as exc:
            raise classify_openai_error(exc) from exc

        response = LLMResponse(
            content="".join(content_parts),
            model=model,
            usage=LLMUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            finish_reason=finish_reason,
        )

        self._track_usage(response.usage)

        # 记录 OpenAI 兼容 API 的缓存 token 使用（DashScope/Qwen 等服务端自动缓存）
        if usage:
            cached_tokens = None
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details:
                cached_tokens = getattr(prompt_details, "cached_tokens", None)
            if cached_tokens:
                logger.info(
                    "API prompt caching: cached_tokens=%d, prompt_tokens=%d, model=%s",
                    cached_tokens,
                    usage.prompt_tokens,
                    model,
                )

        trace = current_trace()
        if trace and trace._current_span:
            trace._current_span.set_attribute("model", response.model)
            trace._current_span.set_attribute("prompt_tokens", response.usage.prompt_tokens)
            trace._current_span.set_attribute("completion_tokens", response.usage.completion_tokens)

        if self._middleware_pipeline:
            response = await self._middleware_pipeline.post_process(response)

        return response

    async def chat_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        purpose: str = "chat",
        *,
        deadline: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """带工具定义的聊天请求（含应用层重试 + Fallback + 缓存）"""
        model_config = self._config.get_model_config(purpose)
        model = kwargs.get("model", model_config.model)
        temperature = kwargs.get("temperature", model_config.temperature)
        tools_hash = self._response_cache._make_tool_cache_key(tools)

        cached = self._response_cache.get(messages, model, temperature, tools_hash=tools_hash)
        if cached is not None:
            return cached

        response = await self._call_with_retry_and_fallback(
            self._chat_with_tools_impl, messages, tools, purpose, deadline=deadline, **kwargs
        )

        self._response_cache.put(messages, response, model, temperature, tools_hash=tools_hash)
        return response

    async def _chat_with_tools_impl(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        purpose: str = "chat",
        **kwargs: Any,
    ) -> LLMResponse:
        """chat_with_tools 的实际实现（流式，支持 tool_calls）"""
        model_config = self._config.get_model_config(purpose)
        model = kwargs.pop("model", model_config.model)
        temperature = kwargs.pop("temperature", model_config.temperature)
        max_tokens = kwargs.pop("max_tokens", model_config.max_tokens)
        top_p = kwargs.pop("top_p", model_config.top_p)

        self._check_budget(max_tokens)

        api_key = self._cached_api_key or self._config.resolve_api_key()
        if not api_key:
            raise LLMError(
                "未配置 API 密钥",
                model=model,
                provider=self._config.provider.value,
                trace_id=current_trace_id(),
                span_id=current_span_id(),
            )

        if self._middleware_pipeline:
            messages = await self._middleware_pipeline.pre_process(messages)

        sdk_messages = self._build_sdk_messages(messages)

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": sdk_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "tools": tools,
            "tool_choice": "auto",
            "stream": True,
        }

        response_format = kwargs.get("response_format") or model_config.response_format
        if response_format:
            create_kwargs["response_format"] = response_format

        logger.debug(
            "LLM chat_with_tools (stream): model=%s, messages=%d, tools=%d",
            model, len(sdk_messages), len(tools),
        )

        import time as _time
        start_time = _time.monotonic()

        stream_timeout = float(self._config.timeout.read)

        try:
            stream = await asyncio.wait_for(
                self._client.chat.completions.create(**create_kwargs),
                timeout=stream_timeout,
            )
        except asyncio.TimeoutError:
            raise LLMTimeoutError(
                f"流式请求建立超时 ({stream_timeout}s)",
                model=model,
                timeout=stream_timeout,
                trace_id=current_trace_id(),
                span_id=current_span_id(),
            )
        except Exception as exc:
            elapsed = _time.monotonic() - start_time
            logger.debug(
                "LLM API 调用失败: model=%s, elapsed=%.1fs, error=%s: %s",
                model, elapsed, type(exc).__name__, str(exc)[:200],
            )
            raise classify_openai_error(exc) from exc

        content_parts: list[str] = []
        tool_call_chunks: dict[int, dict[str, Any]] = {}
        finish_reason = ""
        usage = None
        first_chunk_received = False
        loop_start = _time.monotonic()
        _FIRST_CHUNK_TIMEOUT = 120.0
        _CHUNK_IDLE_TIMEOUT = 60.0

        try:
            aiter = stream.__aiter__()
            while True:
                current_timeout = _FIRST_CHUNK_TIMEOUT if not first_chunk_received else _CHUNK_IDLE_TIMEOUT
                try:
                    chunk = await asyncio.wait_for(aiter.__anext__(), timeout=current_timeout)
                except StopAsyncIteration:
                    break

                if not first_chunk_received:
                    first_chunk_received = True
                    elapsed = _time.monotonic() - loop_start
                    logger.debug(
                        "_chat_with_tools_impl first chunk: model=%s, latency=%.1fs",
                        model, elapsed,
                    )
                if chunk.usage:
                    usage = chunk.usage
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta is not None:
                        if delta.content:
                            content_parts.append(delta.content)
                    if delta and delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_call_chunks:
                                tool_call_chunks[idx] = {
                                    "id": "",
                                    "name": "",
                                    "arguments_parts": [],
                                }
                            entry = tool_call_chunks[idx]
                            if tc_delta.id:
                                entry["id"] = tc_delta.id
                            if tc_delta.function and tc_delta.function.name:
                                entry["name"] += tc_delta.function.name
                            if tc_delta.function and tc_delta.function.arguments:
                                entry["arguments_parts"].append(tc_delta.function.arguments)
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
        except asyncio.TimeoutError:
            elapsed = _time.monotonic() - loop_start
            detail = "首 token 超时" if not first_chunk_received else "流式读取空闲超时"
            raise LLMTimeoutError(
                f"LLM 流式响应超时: {detail} ({elapsed:.0f}s, 空闲阈值 {current_timeout:.0f}s)",
                model=model,
                timeout=current_timeout,
                trace_id=current_trace_id(),
                span_id=current_span_id(),
            )
        except LLMTimeoutError:
            raise
        except Exception as exc:
            raise classify_openai_error(exc) from exc

        import json

        tool_calls: list[dict[str, Any]] = []
        for idx in sorted(tool_call_chunks.keys()):
            entry = tool_call_chunks[idx]
            arguments_str = "".join(entry["arguments_parts"])
            try:
                arguments = json.loads(arguments_str) if arguments_str else {}
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append({
                "id": entry["id"],
                "name": entry["name"],
                "arguments": arguments,
            })

        response = LLMResponse(
            content="".join(content_parts),
            model=model,
            usage=LLMUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )

        self._track_usage(response.usage)

        # 记录 OpenAI 兼容 API 的缓存 token 使用（DashScope/Qwen 等服务端自动缓存）
        if usage:
            cached_tokens = None
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details:
                cached_tokens = getattr(prompt_details, "cached_tokens", None)
            if cached_tokens:
                logger.info(
                    "API prompt caching (tools): cached_tokens=%d, prompt_tokens=%d, model=%s",
                    cached_tokens,
                    usage.prompt_tokens,
                    model,
                )

        trace = current_trace()
        if trace and trace._current_span:
            trace._current_span.set_attribute("model", response.model)
            trace._current_span.set_attribute("prompt_tokens", response.usage.prompt_tokens)
            trace._current_span.set_attribute("completion_tokens", response.usage.completion_tokens)
            trace._current_span.set_attribute("tool_calls_count", len(tool_calls))

        if self._middleware_pipeline:
            response = await self._middleware_pipeline.post_process(response)

        return response

    async def stream_chat(
        self,
        messages: list[LLMMessage],
        purpose: str = "chat",
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """流式聊天请求 — 逐 token 返回内容（含超时控制）"""
        model_config = self._config.get_model_config(purpose)

        model = kwargs.pop("model", model_config.model)
        temperature = kwargs.pop("temperature", model_config.temperature)
        max_tokens = kwargs.pop("max_tokens", model_config.max_tokens)
        top_p = kwargs.pop("top_p", model_config.top_p)

        self._check_budget(max_tokens)

        api_key = self._cached_api_key or self._config.resolve_api_key()
        if not api_key:
            raise LLMError(
                f"未配置 API 密钥。请设置环境变量或配置 llm.api_key。"
                f"当前提供商: {self._config.provider.value}",
                model=model,
                provider=self._config.provider.value,
            )

        if self._middleware_pipeline:
            messages = await self._middleware_pipeline.pre_process(messages)

        sdk_messages = self._build_sdk_messages(messages)

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": sdk_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True,
        }

        logger.debug(
            "LLM stream_chat request: model=%s, messages=%d",
            model,
            len(sdk_messages),
        )

        stream_timeout = float(self._config.timeout.read)

        try:
            stream = await asyncio.wait_for(
                self._client.chat.completions.create(**create_kwargs),
                timeout=stream_timeout,
            )
        except asyncio.TimeoutError:
            raise LLMTimeoutError(
                f"流式请求建立超时 ({stream_timeout}s)",
                model=model,
                timeout=stream_timeout,
                trace_id=current_trace_id(),
                span_id=current_span_id(),
            )
        except Exception as exc:
            raise classify_openai_error(exc) from exc

        collected_content: list[str] = []
        completion_tokens = 0
        _FIRST_TOKEN_TIMEOUT = 60.0
        _CHUNK_IDLE_TIMEOUT = 60.0
        first_token_received = False

        try:
            aiter = stream.__aiter__()
            while True:
                current_timeout = _FIRST_TOKEN_TIMEOUT if not first_token_received else _CHUNK_IDLE_TIMEOUT
                try:
                    chunk = await asyncio.wait_for(aiter.__anext__(), timeout=current_timeout)
                except StopAsyncIteration:
                    break

                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta is not None:
                        token = delta.content or getattr(delta, "reasoning", "")
                        if token:
                            collected_content.append(token)
                            completion_tokens += 1
                            first_token_received = True
                            yield token
        except asyncio.TimeoutError:
            if not first_token_received:
                raise LLMTimeoutError(
                    f"流式首 token 超时 ({_FIRST_TOKEN_TIMEOUT}s)",
                    model=model,
                    timeout=_FIRST_TOKEN_TIMEOUT,
                    trace_id=current_trace_id(),
                    span_id=current_span_id(),
                )
            logger.warning(
                "stream_chat 空闲超时: model=%s, tokens=%d, idle_threshold=%.0fs",
                model, completion_tokens, current_timeout,
            )
        except LLMTimeoutError:
            raise
        except Exception as exc:
            raise classify_openai_error(exc) from exc

        estimated_usage = LLMUsage(
            prompt_tokens=0,
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens,
        )
        self._track_usage(estimated_usage)

        logger.debug(
            "LLM stream_chat complete: model=%s, tokens=%d",
            model,
            completion_tokens,
        )

    async def _call_with_retry_and_fallback(
        self,
        fn: Any,
        *args: Any,
        deadline: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """带重试和 Fallback 的调用包装

        1. 先用主模型 + 应用层重试
        2. 主模型重试耗尽后，依次尝试 Fallback 模型链
        3. 所有模型都失败，抛出最后一个异常

        优化：如果主模型因超时/连接错误失败，且 Fallback 模型在同一 API 端点上，
        则跳过 Fallback（同一端点大概率也会超时），避免双倍等待时间。
        """
        import time as _time

        last_exc: Exception | None = None

        try:
            return await retry(fn, *args, policy=self._retry_policy, deadline=deadline, **kwargs)
        except LLMError as exc:
            last_exc = exc
            if not self._fallback_models:
                raise
            if not exc.is_retryable():
                logger.info(
                    "主模型不可重试错误 (%s)，跳过 Fallback",
                    type(exc).__name__,
                )
                raise

        for fallback_model in self._fallback_models:
            if deadline is not None:
                remaining = deadline - _time.monotonic()
                if remaining <= 10.0:
                    logger.warning("剩余时间不足(%.1fs)，跳过 Fallback", remaining)
                    if last_exc is not None:
                        raise last_exc
                    raise LLMError("所有模型调用均失败（时间不足）")

            logger.info(
                "主模型失败，尝试 Fallback 模型: %s",
                fallback_model,
            )
            try:
                return await retry(
                    fn, *args,
                    policy=self._retry_policy,
                    deadline=deadline,
                    model=fallback_model,
                    **kwargs,
                )
            except LLMError as exc:
                last_exc = exc
                logger.debug(
                    "Fallback 模型 %s 也失败: %s",
                    fallback_model,
                    type(exc).__name__,
                )
                continue

        if last_exc is not None:
            raise last_exc
        raise LLMError("所有模型调用均失败")

    def _build_sdk_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        import json as _json
        sdk_messages: list[dict[str, Any]] = []
        for m in messages:
            msg_dict: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                # 将 LLMMessage 格式的 tool_calls 转为 OpenAI SDK 格式
                sdk_tool_calls = []
                for tc in m.tool_calls:
                    if "type" in tc and tc.get("type") == "function" and "function" in tc:
                        # 已经是 SDK 格式，直接使用
                        sdk_tool_calls.append(tc)
                    else:
                        # 转换: {"id": "...", "name": "...", "arguments": {...}} →
                        #        {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
                        tc_id = tc.get("id", "")
                        tc_name = tc.get("name", "")
                        tc_args = tc.get("arguments", {})
                        if isinstance(tc_args, dict):
                            tc_args = _json.dumps(tc_args, ensure_ascii=False)
                        elif not isinstance(tc_args, str):
                            tc_args = str(tc_args)
                        sdk_tool_calls.append({
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": tc_name,
                                "arguments": tc_args,
                            },
                        })
                msg_dict["tool_calls"] = sdk_tool_calls
            if m.tool_call_id:
                msg_dict["tool_call_id"] = m.tool_call_id
            if m.name:
                msg_dict["name"] = m.name
            sdk_messages.append(msg_dict)
        return sdk_messages

    def _parse_completion(self, completion: Any, default_model: str) -> LLMResponse:
        choice = completion.choices[0]
        usage = completion.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=completion.model or default_model,
            usage=LLMUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            finish_reason=choice.finish_reason or "",
        )

    def _check_budget(self, max_tokens: int) -> None:
        budget = self._config.budget
        if self._task_tokens_used + max_tokens > budget.max_tokens_per_task:
            from long.errors import LLMBudgetExceededError
            raise LLMBudgetExceededError(
                f"超出单任务预算: 已用 {self._task_tokens_used}, "
                f"请求 {max_tokens}, 上限 {budget.max_tokens_per_task}",
                budget_type="task",
                used=self._task_tokens_used,
                limit=budget.max_tokens_per_task,
                trace_id=current_trace_id(),
                span_id=current_span_id(),
            )
        if self._daily_tokens_used + max_tokens > budget.daily_token_limit:
            from long.errors import LLMBudgetExceededError
            raise LLMBudgetExceededError(
                f"超出日预算: 已用 {self._daily_tokens_used}, "
                f"请求 {max_tokens}, 上限 {budget.daily_token_limit}",
                budget_type="daily",
                used=self._daily_tokens_used,
                limit=budget.daily_token_limit,
                trace_id=current_trace_id(),
                span_id=current_span_id(),
            )

    def _track_usage(self, usage: LLMUsage) -> None:
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._daily_tokens_date:
            self._daily_tokens_used = 0
            self._daily_tokens_date = today
        self._daily_tokens_used += usage.total_tokens
        self._task_tokens_used += usage.total_tokens
        self._request_count += 1

    def reset_task_budget(self) -> None:
        self._task_tokens_used = 0

    def reset_daily_budget(self) -> None:
        self._daily_tokens_used = 0

    @property
    def daily_tokens_used(self) -> int:
        return self._daily_tokens_used

    @property
    def task_tokens_used(self) -> int:
        return self._task_tokens_used

    @property
    def request_count(self) -> int:
        return self._request_count

    def get_judge_fn(self, purpose: str = "judge") -> Any:
        async def judge_fn(prompt: str) -> str:
            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.chat(messages, purpose=purpose)
            return response.content
        return judge_fn

    def get_repair_fn(self, purpose: str = "repair") -> Any:
        async def repair_fn(prompt: str) -> str:
            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.chat(messages, purpose=purpose)
            return response.content
        return repair_fn
