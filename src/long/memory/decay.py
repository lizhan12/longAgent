"""记忆衰减模块

定义记忆强度衰减公式和相关工具函数。

衰减公式: strength(t) = initial × exp(-λ × hours)
- short_term: λ=0.1（10小时衰减到 ~37%）
- long_term: λ=0.005（200小时衰减到 ~37%）
- procedural: λ=0.001（1000小时衰减到 ~37%）
"""

from __future__ import annotations

import math
import time
from enum import Enum
from typing import Any


class DecayRate(str, Enum):
    """衰减率预设"""

    SHORT_TERM = "short_term"      # λ=0.1，快速衰减
    LONG_TERM = "long_term"        # λ=0.005，缓慢衰减
    PROCEDURAL = "procedural"      # λ=0.001，几乎不衰减


DECAY_LAMBDA: dict[DecayRate, float] = {
    DecayRate.SHORT_TERM: 0.1,
    DecayRate.LONG_TERM: 0.005,
    DecayRate.PROCEDURAL: 0.001,
}


def compute_strength(
    initial_strength: float,
    created_at: float,
    decay_rate: DecayRate = DecayRate.LONG_TERM,
    custom_lambda: float | None = None,
    now: float | None = None,
) -> float:
    """计算记忆强度

    Args:
        initial_strength: 初始强度 [0, 1]
        created_at: 创建时间戳（秒）
        decay_rate: 衰减率预设
        custom_lambda: 自定义 λ 值（覆盖预设）
        now: 当前时间戳，默认 time.time()

    Returns:
        当前强度 [0, 1]
    """
    if initial_strength <= 0:
        return 0.0

    age_hours = ((now or time.time()) - created_at) / 3600.0
    if age_hours <= 0:
        return initial_strength

    lam = custom_lambda if custom_lambda is not None else DECAY_LAMBDA[decay_rate]

    # strength(t) = initial × exp(-λ × hours)
    strength = initial_strength * math.exp(-lam * age_hours)

    return max(0.0, min(1.0, strength))


def apply_decay_to_item(
    item: Any,
    decay_rate: DecayRate = DecayRate.LONG_TERM,
    custom_lambda: float | None = None,
) -> float:
    """对记忆项应用衰减，更新其 strength 字段

    Args:
        item: 具有 strength, created_at 属性的对象
        decay_rate: 衰减率预设
        custom_lambda: 自定义 λ 值

    Returns:
        衰减后的强度
    """
    new_strength = compute_strength(
        initial_strength=item.strength,
        created_at=item.created_at,
        decay_rate=decay_rate,
        custom_lambda=custom_lambda,
    )
    item.strength = new_strength
    return new_strength


def estimate_half_life(decay_rate: DecayRate, custom_lambda: float | None = None) -> float:
    """估算半衰期（小时）

    记忆强度衰减到50%所需的时间。
    t_½ = ln(2) / λ
    """
    lam = custom_lambda if custom_lambda is not None else DECAY_LAMBDA[decay_rate]
    if lam <= 0:
        return float("inf")
    return math.log(2) / lam


def get_retention_rate(
    age_hours: float,
    decay_rate: DecayRate = DecayRate.LONG_TERM,
    custom_lambda: float | None = None,
) -> float:
    """计算保留率（给定年龄的强度比率）

    Args:
        age_hours: 年龄（小时）
        decay_rate: 衰减率预设
        custom_lambda: 自定义 λ 值

    Returns:
        保留率 [0, 1]
    """
    if age_hours <= 0:
        return 1.0
    lam = custom_lambda if custom_lambda is not None else DECAY_LAMBDA[decay_rate]
    return math.exp(-lam * age_hours)