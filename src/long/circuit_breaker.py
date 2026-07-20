"""熔断器注册表

管理系统中所有的熔断器实例，提供状态查询。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitStats:
    name: str = ""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_time: float = 0.0
    last_failure_reason: str = ""
    last_state_change: float = 0.0
    opened_at: float = 0.0

    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_failures / self.total_calls


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 30.0
    half_open_max_calls: int = 1


class CircuitBreaker:
    """熔断器实例

    状态机: CLOSED → OPEN → HALF_OPEN → CLOSED
    - CLOSED: 正常状态，允许请求通过
    - OPEN: 熔断状态，拒绝请求；超时后转为 HALF_OPEN
    - HALF_OPEN: 半开状态，允许有限请求通过以探测服务是否恢复
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._last_failure_time: float = 0.0
        self._last_failure_reason: str = ""
        self._last_state_change: float = time.monotonic()
        self._opened_at: float = 0.0
        self._half_open_calls: int = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """获取当前状态，OPEN 超时后自动转为 HALF_OPEN"""
        with self._lock:
            self._check_timeout()
            return self._state

    def can_execute(self) -> bool:
        """判断是否允许执行请求

        - CLOSED: 允许
        - OPEN: 不允许（超时后转为 HALF_OPEN 后允许）
        - HALF_OPEN: 允许但受 half_open_max_calls 限制
        """
        with self._lock:
            self._check_timeout()
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                return False
            # HALF_OPEN 状态
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    def record_success(self) -> None:
        """记录成功调用

        - CLOSED: 重置失败计数
        - HALF_OPEN: 递增成功计数，达到 success_threshold 后转为 CLOSED
        """
        with self._lock:
            self._total_calls += 1
            self._total_successes += 1

            if self._state == CircuitState.CLOSED:
                # 成功调用重置连续失败计数
                self._failure_count = 0
            elif self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def record_failure(self, reason: str = "") -> None:
        """记录失败调用

        - CLOSED: 递增失败计数，达到 failure_threshold 后转为 OPEN
        - HALF_OPEN: 任何失败都立即转为 OPEN
        - OPEN: 仅更新统计
        """
        with self._lock:
            self._total_calls += 1
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            self._last_failure_reason = reason

            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.HALF_OPEN:
                # 半开状态下任何失败都回到熔断
                self._transition_to(CircuitState.OPEN)

    def reset(self) -> None:
        """重置为 CLOSED 状态"""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)

    def force_open(self, reason: str = "") -> None:
        """强制转为 OPEN 状态"""
        with self._lock:
            self._last_failure_reason = reason
            self._transition_to(CircuitState.OPEN)

    def get_stats(self) -> CircuitStats:
        """获取当前熔断器统计信息"""
        with self._lock:
            self._check_timeout()
            return CircuitStats(
                name=self.name,
                state=self._state,
                failure_count=self._failure_count,
                success_count=self._success_count,
                total_calls=self._total_calls,
                total_failures=self._total_failures,
                total_successes=self._total_successes,
                last_failure_time=self._last_failure_time,
                last_failure_reason=self._last_failure_reason,
                last_state_change=self._last_state_change,
                opened_at=self._opened_at,
            )

    def _check_timeout(self) -> None:
        """检查 OPEN 状态是否超时，超时则转为 HALF_OPEN"""
        if self._state == CircuitState.OPEN and self._opened_at > 0:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.config.timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """状态转换，重置相关计数器"""
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self._last_state_change = time.monotonic()

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._opened_at = 0.0
        elif new_state == CircuitState.OPEN:
            self._success_count = 0
            self._half_open_calls = 0
            self._opened_at = time.monotonic()
        elif new_state == CircuitState.HALF_OPEN:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0


class CircuitBreakerRegistry:
    """熔断器注册表

    管理所有命名熔断器实例，提供全局状态查询和操作。
    """

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self, name: str, config: CircuitBreakerConfig | None = None
    ) -> CircuitBreaker:
        """获取已有熔断器或创建新实例"""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """按名称获取熔断器，不存在返回 None"""
        with self._lock:
            return self._breakers.get(name)

    def get_all_stats(self) -> dict[str, CircuitStats]:
        """获取所有熔断器的统计信息"""
        with self._lock:
            breakers = list(self._breakers.values())
        return {b.name: b.get_stats() for b in breakers}

    def get_open_breakers(self) -> list[str]:
        """获取所有处于 OPEN 状态的熔断器名称"""
        with self._lock:
            breakers = list(self._breakers.values())
        return [b.name for b in breakers if b.state == CircuitState.OPEN]

    def record_success(self, name: str) -> None:
        """记录指定熔断器的成功调用"""
        breaker = self.get(name)
        if breaker:
            breaker.record_success()

    def record_failure(self, name: str, reason: str = "") -> None:
        """记录指定熔断器的失败调用"""
        breaker = self.get(name)
        if breaker:
            breaker.record_failure(reason)

    def can_execute(self, name: str) -> bool:
        """判断指定熔断器是否允许执行"""
        breaker = self.get(name)
        if breaker:
            return breaker.can_execute()
        # 不存在的熔断器默认允许执行
        return True


circuit_registry = CircuitBreakerRegistry()
