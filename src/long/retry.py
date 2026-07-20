"""统一重试框架

提供可配置的重试策略，支持指数退避、随机抖动、错误分类。
与 Trace 体系集成，每次重试记录 Span Event。

用法:
    # 使用默认策略重试
    response = await retry(llm.chat, messages)

    # 自定义策略
    policy = RetryPolicy(max_attempts=5, base_delay=2.0)
    response = await retry(llm.chat, messages, policy=policy)

    # 装饰器模式
    @retryable(policy=RetryPolicy(max_attempts=3))
    async def call_api():
        ...
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from long.errors import LongError, LLMRateLimitError, LLMTimeoutError

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter_range: float = 0.1
    retryable_types: tuple[type[Exception], ...] = ()
    max_timeout_retries: int = 1

    def compute_delay(self, attempt: int, exc: Exception | None = None) -> float:
        if isinstance(exc, LLMRateLimitError) and exc.retry_after is not None:
            return exc.retry_after

        delay = self.base_delay * (self.backoff_factor ** attempt)
        jitter = delay * self.jitter_range * random.random()
        delay = delay + jitter
        return min(delay, self.max_delay)

    def should_retry(self, exc: Exception, consecutive_timeouts: int = 0) -> bool:
        if isinstance(exc, LLMTimeoutError):
            return consecutive_timeouts <= self.max_timeout_retries

        if self.retryable_types:
            return isinstance(exc, self.retryable_types)

        if isinstance(exc, LongError):
            return exc.is_retryable()

        exc_name = type(exc).__name__
        retryable_names = {
            "RateLimitError", "APITimeoutError", "APIConnectionError",
            "InternalServerError", "TimeoutError", "ConnectionError",
            "ConnectionResetError", "ConnectionAbortedError",
        }
        return exc_name in retryable_names


DEFAULT_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0,
    jitter_range=0.1,
)

LLM_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay=2.0,
    max_delay=120.0,
    backoff_factor=2.0,
    jitter_range=0.15,
)

TOOL_RETRY_POLICY = RetryPolicy(
    max_attempts=2,
    base_delay=0.5,
    max_delay=10.0,
    backoff_factor=2.0,
    jitter_range=0.1,
)


@dataclass
class RetryStats:
    function_name: str = ""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_retries: int = 0
    last_error: str = ""
    last_error_time: float = 0.0
    retry_history: list[dict[str, Any]] = field(default_factory=list)

    def record_success(self, attempts: int) -> None:
        self.total_calls += 1
        self.successful_calls += 1
        if attempts > 1:
            self.total_retries += attempts - 1

    def record_failure(self, exc: Exception, attempts: int) -> None:
        self.total_calls += 1
        self.failed_calls += 1
        self.total_retries += attempts - 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        self.last_error_time = time.time()

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def avg_retries(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_retries / self.total_calls


class RetryRegistry:
    def __init__(self) -> None:
        self._stats: dict[str, RetryStats] = {}

    def get_stats(self, name: str) -> RetryStats:
        if name not in self._stats:
            self._stats[name] = RetryStats(function_name=name)
        return self._stats[name]

    def get_all_stats(self) -> dict[str, RetryStats]:
        return dict(self._stats)

    def get_summary(self) -> dict[str, Any]:
        total_calls = sum(s.total_calls for s in self._stats.values())
        total_retries = sum(s.total_retries for s in self._stats.values())
        total_failures = sum(s.failed_calls for s in self._stats.values())
        return {
            "total_calls": total_calls,
            "total_retries": total_retries,
            "total_failures": total_failures,
            "by_function": {
                name: {
                    "calls": s.total_calls,
                    "success_rate": s.success_rate,
                    "avg_retries": s.avg_retries,
                    "last_error": s.last_error,
                }
                for name, s in self._stats.items()
            },
        }


retry_registry = RetryRegistry()


async def retry(
    fn: Callable[..., Any],
    *args: Any,
    policy: RetryPolicy | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    deadline: float | None = None,
    **kwargs: Any,
) -> Any:
    """使用重试策略调用异步函数

    Args:
        fn: 要调用的异步函数
        *args: 函数位置参数
        policy: 重试策略，默认 DEFAULT_RETRY_POLICY
        on_retry: 重试回调 (attempt, exception, delay)
        deadline: 绝对截止时间（time.monotonic 值），超时则不再重试
        **kwargs: 函数关键字参数

    Returns:
        函数返回值

    Raises:
        最后一次尝试的异常
    """
    if policy is None:
        policy = DEFAULT_RETRY_POLICY

    fn_name = getattr(fn, "__qualname__", getattr(fn, "__name__", str(fn)))
    stats = retry_registry.get_stats(fn_name)

    last_exc: Exception | None = None
    consecutive_timeouts = 0

    for attempt in range(policy.max_attempts):
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "[%s] 已超过截止时间，停止重试 (attempt %d/%d)",
                    fn_name, attempt + 1, policy.max_attempts,
                )
                if last_exc is not None:
                    stats.record_failure(last_exc, attempt)
                    raise last_exc
                stats.record_failure(RuntimeError("deadline exceeded"), attempt)
                raise LLMTimeoutError(
                    f"重试截止时间已到，停止重试 ({fn_name})",
                    timeout=0.0,
                )

        try:
            result = await fn(*args, **kwargs)
            stats.record_success(attempt + 1)
            return result
        except Exception as exc:
            last_exc = exc

            if isinstance(exc, LLMTimeoutError):
                consecutive_timeouts += 1
            else:
                consecutive_timeouts = 0

            if not policy.should_retry(exc, consecutive_timeouts=consecutive_timeouts):
                logger.debug(
                    "[%s] 不可重试错误: %s (attempt %d/%d)",
                    fn_name, type(exc).__name__, attempt + 1, policy.max_attempts,
                )
                stats.record_failure(exc, attempt + 1)
                raise

            if attempt < policy.max_attempts - 1:
                delay = policy.compute_delay(attempt, exc)

                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= delay + 5.0:
                        logger.warning(
                            "[%s] 剩余时间不足(%.1fs)，停止重试 (attempt %d/%d)",
                            fn_name, remaining, attempt + 1, policy.max_attempts,
                        )
                        stats.record_failure(exc, attempt + 1)
                        raise

                logger.info(
                    "[%s] 可重试错误: %s (attempt %d/%d, delay=%.1fs)",
                    fn_name, type(exc).__name__, attempt + 1, policy.max_attempts, delay,
                )

                stats.retry_history.append({
                    "attempt": attempt + 1,
                    "error": f"{type(exc).__name__}: {exc}",
                    "delay": delay,
                    "timestamp": time.time(),
                })

                _record_retry_span_event(attempt + 1, exc, delay)

                if on_retry:
                    on_retry(attempt + 1, exc, delay)

                await asyncio.sleep(delay)
            else:
                logger.debug(
                    "[%s] 重试耗尽: %s (attempt %d/%d)",
                    fn_name, type(exc).__name__, attempt + 1, policy.max_attempts,
                )
                stats.record_failure(exc, attempt + 1)
                raise

    if last_exc is not None:
        raise last_exc

    raise RuntimeError("retry loop exited without result or exception")


def retryable(
    policy: RetryPolicy | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> Callable[..., Any]:
    """异步函数重试装饰器

    Args:
        policy: 重试策略
        on_retry: 重试回调

    Returns:
        装饰后的函数
    """
    _policy = policy

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await retry(fn, *args, policy=_policy, on_retry=on_retry, **kwargs)
        return wrapper
    return decorator


def _record_retry_span_event(attempt: int, exc: Exception, delay: float) -> None:
    try:
        from long.observability.tracing import current_trace
        trace = current_trace()
        if trace and trace._current_span:
            trace._current_span.add_event(
                "retry",
                {
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:200],
                    "delay": delay,
                },
            )
    except ImportError:
        pass
