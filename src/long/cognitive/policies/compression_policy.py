from __future__ import annotations
from typing import Any
from .base import PolicyDecision, AdaptivePolicy

class AdaptiveCompressionPolicy:
    def __init__(
        self,
        search_threshold: int = 800,
        code_threshold: int = 1200,
        general_threshold: int = 1000,
        context_window: int = 128000,
    ):
        self._base_search = search_threshold
        self._base_code = code_threshold
        self._base_general = general_threshold
        self._context_window = context_window

    def decide(self, context: dict[str, Any] | None = None) -> PolicyDecision:
        ctx = context or {}
        used_ratio = ctx.get("context_used_ratio", 0.0)

        if used_ratio > 0.8:
            factor = 0.5
        elif used_ratio > 0.6:
            factor = 0.7
        elif used_ratio > 0.4:
            factor = 0.85
        else:
            factor = 1.0

        return PolicyDecision(
            compression_threshold_search=max(200, int(self._base_search * factor)),
            compression_threshold_code=max(300, int(self._base_code * factor)),
            compression_threshold_general=max(250, int(self._base_general * factor)),
            metadata={"context_used_ratio": used_ratio, "compression_factor": factor},
        )

    def update(self, outcome: dict[str, Any]) -> None:
        pass

    def get_stats(self) -> dict[str, Any]:
        return {
            "policy_type": "adaptive_compression",
            "base_thresholds": {
                "search": self._base_search,
                "code": self._base_code,
                "general": self._base_general,
            },
        }
