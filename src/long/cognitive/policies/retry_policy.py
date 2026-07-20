from __future__ import annotations
import statistics
from typing import Any
from .base import PolicyDecision, RetryRecord, AdaptivePolicy

class StatisticalRetryPolicy:
    def __init__(self, initial_max: int = 2, min_max: int = 1, max_max: int = 4, warmup: int = 10):
        self._initial_max = initial_max
        self._min_max = min_max
        self._max_max = max_max
        self._warmup = warmup
        self._history: list[RetryRecord] = []

    def decide(self, context: dict[str, Any] | None = None) -> PolicyDecision:
        if len(self._history) < self._warmup:
            return PolicyDecision(max_retries=self._initial_max)

        recent = self._history[-30:]
        success_rate = sum(1 for r in recent if r.retry_succeeded) / len(recent)

        if success_rate < 0.2:
            recommended = max(self._min_max, self._initial_max - 1)
        elif success_rate > 0.6:
            recommended = min(self._max_max, self._initial_max + 1)
        else:
            recommended = self._initial_max

        return PolicyDecision(
            max_retries=recommended,
            metadata={"retry_success_rate": success_rate, "sample_size": len(recent)},
        )

    def update(self, outcome: dict[str, Any]) -> None:
        self._history.append(RetryRecord(
            error_type=outcome.get("error_type", "unknown"),
            retry_succeeded=outcome.get("retry_succeeded", False),
            tool_name=outcome.get("tool_name", ""),
            attempt_number=outcome.get("attempt_number", 1),
        ))

    def get_stats(self) -> dict[str, Any]:
        if not self._history:
            return {"policy_type": "statistical_retry", "history_size": 0, "current_max": self._initial_max}
        success_rate = sum(1 for r in self._history if r.retry_succeeded) / len(self._history)
        return {
            "policy_type": "statistical_retry",
            "history_size": len(self._history),
            "retry_success_rate": success_rate,
            "current_max": self.decide().max_retries,
        }
