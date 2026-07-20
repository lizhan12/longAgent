"""统一异常层次结构

所有 Long 系统异常的基类和分类体系。
每个异常携带结构化上下文，支持 Trace 关联和错误分类。

异常层次:
    LongError
    ├── LLMError
    │   ├── LLMRateLimitError      (429 - 可重试)
    │   ├── LLMServerError         (5xx - 可重试)
    │   ├── LLMTimeoutError        (超时 - 可重试)
    │   ├── LLMConnectionError     (网络 - 可重试)
    │   ├── LLMBudgetExceededError (预算 - 不可重试)
    │   └── LLMInvalidRequestError (4xx - 不可重试)
    ├── ToolError
    │   ├── ToolExecutionError     (执行失败)
    │   ├── ToolTimeoutError       (超时 - 可重试)
    │   └── ToolNotFoundError      (不存在 - 不可重试)
    ├── MemoryError
    │   ├── MemoryStorageError     (存储失败)
    │   └── MemoryRetrievalError   (检索失败)
    └── PlanError
        ├── PlanGenerationError    (生成失败)
        └── PlanValidationError    (验证失败)
"""

from __future__ import annotations

from typing import Any


class LongError(Exception):
    """所有 Long 系统异常的基类"""

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        trace_id: str = "",
        span_id: str = "",
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.context = context or {}
        self.trace_id = trace_id
        self.span_id = span_id
        self.cause = cause

    def is_retryable(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": type(self).__name__,
            "message": str(self),
            "retryable": self.is_retryable(),
            "context": self.context,
        }
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.cause:
            result["cause"] = f"{type(self.cause).__name__}: {self.cause}"
        return result


class LLMError(LongError):
    """LLM 调用相关异常"""

    def __init__(
        self,
        message: str,
        *,
        model: str = "",
        provider: str = "",
        status_code: int | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if model:
            context["model"] = model
        if provider:
            context["provider"] = provider
        if status_code is not None:
            context["status_code"] = status_code
        super().__init__(message, context=context, **kwargs)
        self.model = model
        self.provider = provider
        self.status_code = status_code


class LLMRateLimitError(LLMError):
    """429 速率限制 - 可重试"""

    def __init__(
        self,
        message: str = "API 速率限制",
        *,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if retry_after is not None:
            context["retry_after"] = retry_after
        super().__init__(message, context=context, **kwargs)
        self.retry_after = retry_after

    def is_retryable(self) -> bool:
        return True


class LLMServerError(LLMError):
    """5xx 服务端错误 - 可重试"""

    def __init__(
        self,
        message: str = "服务端错误",
        *,
        status_code: int = 500,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, status_code=status_code, **kwargs)

    def is_retryable(self) -> bool:
        return True


class LLMTimeoutError(LLMError):
    """请求超时 - 可重试"""

    def __init__(
        self,
        message: str = "LLM 请求超时",
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if timeout is not None:
            context["timeout"] = timeout
        super().__init__(message, context=context, **kwargs)

    def is_retryable(self) -> bool:
        return True


class LLMConnectionError(LLMError):
    """网络连接错误 - 可重试"""

    def __init__(
        self,
        message: str = "LLM 连接失败",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)

    def is_retryable(self) -> bool:
        return True


class LLMBudgetExceededError(LLMError):
    """预算超限 - 不可重试"""

    def __init__(
        self,
        message: str = "LLM 预算超限",
        *,
        budget_type: str = "",
        used: int = 0,
        limit: int = 0,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if budget_type:
            context["budget_type"] = budget_type
        context["used"] = used
        context["limit"] = limit
        super().__init__(message, context=context, **kwargs)

    def is_retryable(self) -> bool:
        return False


class LLMInvalidRequestError(LLMError):
    """4xx 请求错误 - 不可重试"""

    def __init__(
        self,
        message: str = "无效请求",
        *,
        status_code: int = 400,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, status_code=status_code, **kwargs)

    def is_retryable(self) -> bool:
        return False


class ToolError(LongError):
    """工具调用相关异常"""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if tool_name:
            context["tool_name"] = tool_name
        super().__init__(message, context=context, **kwargs)
        self.tool_name = tool_name


class ToolExecutionError(ToolError):
    """工具执行失败"""

    def __init__(
        self,
        message: str = "工具执行失败",
        *,
        tool_name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, tool_name=tool_name, **kwargs)

    def is_retryable(self) -> bool:
        return self.context.get("retryable", False)


class ToolTimeoutError(ToolError):
    """工具执行超时 - 可重试"""

    def __init__(
        self,
        message: str = "工具执行超时",
        *,
        tool_name: str = "",
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if timeout is not None:
            context["timeout"] = timeout
        super().__init__(message, tool_name=tool_name, context=context, **kwargs)

    def is_retryable(self) -> bool:
        return True


class ToolNotFoundError(ToolError):
    """工具不存在 - 不可重试"""

    def __init__(
        self,
        message: str = "工具未找到",
        *,
        tool_name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, tool_name=tool_name, **kwargs)

    def is_retryable(self) -> bool:
        return False


class MemoryError(LongError):
    """记忆系统异常"""

    def __init__(
        self,
        message: str,
        *,
        memory_type: str = "",
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if memory_type:
            context["memory_type"] = memory_type
        super().__init__(message, context=context, **kwargs)


class MemoryStorageError(MemoryError):
    """记忆存储失败"""

    def __init__(
        self,
        message: str = "记忆存储失败",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)


class MemoryRetrievalError(MemoryError):
    """记忆检索失败"""

    def __init__(
        self,
        message: str = "记忆检索失败",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)


class PlanError(LongError):
    """计划相关异常"""

    def __init__(
        self,
        message: str,
        *,
        plan_id: str = "",
        step_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if plan_id:
            context["plan_id"] = plan_id
        if step_index is not None:
            context["step_index"] = step_index
        super().__init__(message, context=context, **kwargs)


class PlanGenerationError(PlanError):
    """计划生成失败"""

    def __init__(
        self,
        message: str = "计划生成失败",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)


class PlanValidationError(PlanError):
    """计划验证失败"""

    def __init__(
        self,
        message: str = "计划验证失败",
        *,
        violations: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if violations:
            context["violations"] = violations
        super().__init__(message, context=context, **kwargs)


def classify_openai_error(exc: Exception) -> LongError:
    """将 OpenAI SDK 异常分类为 Long 异常体系

    Args:
        exc: OpenAI SDK 抛出的异常

    Returns:
        对应的 Long 异常
    """
    from long.observability.tracing import current_trace_id, current_span_id

    trace_id = current_trace_id()
    span_id = current_span_id()
    common = {"trace_id": trace_id, "span_id": span_id, "cause": exc}

    exc_type = type(exc).__name__
    exc_message = str(exc)

    if "RateLimitError" in exc_type or exc_message.startswith("Rate limit"):
        retry_after = None
        if hasattr(exc, "headers") and exc.headers:
            ra = exc.headers.get("retry-after")
            if ra:
                try:
                    retry_after = float(ra)
                except (ValueError, TypeError):
                    pass
        return LLMRateLimitError(
            exc_message,
            retry_after=retry_after,
            **common,
        )

    if "APITimeoutError" in exc_type or "Timeout" in exc_type:
        return LLMTimeoutError(exc_message, **common)

    if "APIConnectionError" in exc_type or "ConnectionError" in exc_type:
        return LLMConnectionError(exc_message, **common)

    if any(kw in exc_type for kw in (
        "RemoteProtocolError", "ReadError", "ConnectError",
        "ConnectTimeout", "ReadTimeout", "WriteTimeout",
        "PoolTimeout", "ConnectionReset", "ConnectionAborted",
        "RemoteDisconnected",
    )):
        return LLMConnectionError(exc_message, **common)

    if "BadRequestError" in exc_type or "InvalidRequestError" in exc_type or "BadRequest" in exc_type:
        status_code = getattr(exc, "status_code", 400)
        return LLMInvalidRequestError(
            exc_message,
            status_code=status_code,
            **common,
        )

    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        if 400 <= status_code < 500:
            return LLMInvalidRequestError(
                exc_message,
                status_code=status_code,
                **common,
            )
        if 500 <= status_code < 600:
            return LLMServerError(
                exc_message,
                status_code=status_code,
                **common,
            )

    if "InternalServerError" in exc_type or "APIStatusError" in exc_type:
        status_code = getattr(exc, "status_code", 500)
        if 500 <= status_code < 600:
            return LLMServerError(
                exc_message,
                status_code=status_code,
                **common,
            )
        if 400 <= status_code < 500:
            return LLMInvalidRequestError(
                exc_message,
                status_code=status_code,
                **common,
            )

    if "budget" in exc_message.lower() or "预算" in exc_message:
        return LLMBudgetExceededError(exc_message, **common)

    return LLMError(exc_message, **common)
