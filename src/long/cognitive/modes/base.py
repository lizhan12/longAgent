from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExecutionMode(str, Enum):
    CONTROLLED = "controlled"
    BALANCED = "balanced"
    EXPLORATORY = "exploratory"


@dataclass
class ModeConfig:
    max_search_rounds: int = 4
    strict_react: bool = False
    allow_self_correction: bool = True
    confidence_threshold: float = 0.7
    allow_strategy_change: bool = True
    max_rounds: int = 8


MODE_PRESETS: dict[ExecutionMode, ModeConfig] = {
    ExecutionMode.CONTROLLED: ModeConfig(
        max_search_rounds=2,
        strict_react=True,
        allow_self_correction=False,
        confidence_threshold=0.9,
        allow_strategy_change=False,
        max_rounds=8,
    ),
    ExecutionMode.BALANCED: ModeConfig(
        max_search_rounds=4,
        strict_react=False,
        allow_self_correction=True,
        confidence_threshold=0.7,
        allow_strategy_change=True,
        max_rounds=10,
    ),
    ExecutionMode.EXPLORATORY: ModeConfig(
        max_search_rounds=6,
        strict_react=False,
        allow_self_correction=True,
        confidence_threshold=0.5,
        allow_strategy_change=True,
        max_rounds=12,
    ),
}


def get_mode_config(mode: ExecutionMode | str, overrides: dict[str, Any] | None = None) -> ModeConfig:
    if isinstance(mode, str):
        mode = ExecutionMode(mode)
    config = ModeConfig(**MODE_PRESETS[mode].__dict__)
    if overrides:
        for k, v in overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)
    return config
