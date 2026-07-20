from __future__ import annotations
import statistics
from typing import Any
from .base import PolicyDecision, SearchRecord, AdaptivePolicy

class StatisticalSearchPolicy:
    def __init__(self, initial_max: int = 3, min_max: int = 2, max_max: int = 6, warmup: int = 5):
        self._initial_max = initial_max
        self._min_max = min_max
        self._max_max = max_max
        self._warmup = warmup
        self._history: list[SearchRecord] = []

    def decide(self, context: dict[str, Any] | None = None) -> PolicyDecision:
        if len(self._history) < self._warmup:
            return PolicyDecision(max_searches=self._initial_max)

        recent = self._history[-20:]
        successes = [r for r in recent if r.completed]
        if successes:
            avg = statistics.mean(r.search_count for r in successes)
            recommended = max(self._min_max, min(self._max_max, int(avg * 1.3)))
        else:
            recommended = self._initial_max

        return PolicyDecision(
            max_searches=recommended,
            metadata={"avg_successful_searches": avg if successes else 0, "sample_size": len(successes)},
        )

    def update(self, outcome: dict[str, Any]) -> None:
        self._history.append(SearchRecord(
            search_count=outcome.get("search_count", 0),
            completed=outcome.get("completed", False),
            query_type=outcome.get("query_type", "general"),
            had_sufficient_results=outcome.get("had_sufficient_results", True),
        ))

    def get_stats(self) -> dict[str, Any]:
        if not self._history:
            return {"policy_type": "statistical_search", "history_size": 0, "current_max": self._initial_max}
        successes = [r for r in self._history if r.completed]
        return {
            "policy_type": "statistical_search",
            "history_size": len(self._history),
            "success_rate": len(successes) / len(self._history),
            "current_max": self.decide().max_searches,
            "avg_searches": statistics.mean(r.search_count for r in successes) if successes else 0,
        }
