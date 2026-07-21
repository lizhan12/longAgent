"""记忆检索策略

定义多种检索策略，用于从记忆存储中检索最相关的记忆项。

策略:
- recency: 按时间倒序（最近优先）
- relevance: 按文本相关性（匹配度优先）
- importance_weighted: 按重要性 × 强度的加权分数
- frequency: 按检索频率（常用优先）
- hybrid: 综合评分（默认，结合多种因素）
"""

from __future__ import annotations

import time
from typing import Any

from .base import MemoryItem, MemoryQuery


def score_recency(item: MemoryItem, now: float | None = None) -> float:
    """新鲜度评分

    最近创建的项得分高。
    score = 1 / (1 + age_hours/24)
    """
    age_hours = ((now or time.time()) - item.created_at) / 3600.0
    return 1.0 / (1.0 + age_hours / 24.0)


def score_relevance(item: MemoryItem, query: str) -> float:
    """文本相关性评分

    Args:
        item: 记忆项
        query: 查询文本

    Returns:
        [0, 1] 的相关性分数
    """
    if not query or query == "*":
        return 0.5

    query_lower = query.lower()
    content_lower = item.content.lower()

    if query_lower in content_lower:
        return 1.0

    query_words = set(query_lower.split())
    content_words = set(content_lower.split())
    overlap = query_words & content_words

    if overlap:
        return len(overlap) / max(len(query_words), 1)

    return 0.0


def score_importance(item: MemoryItem) -> float:
    """重要性加权评分

    score = importance × strength
    """
    return item.importance * item.strength


def score_frequency(item: MemoryItem) -> float:
    """频率评分

    检索次数越多，分数越高（但衰减，避免热门偏见）。
    score = 1 - 1 / (1 + retrieval_count)
    """
    return 1.0 - 1.0 / (1.0 + item.retrieval_count)


def score_hybrid(
    item: MemoryItem,
    query: str,
    now: float | None = None,
) -> float:
    """综合评分（默认策略）

    结合相关性、重要性、新鲜度和频率。

    score = 0.4 × relevance + 0.3 × importance + 0.2 × recency + 0.1 × frequency
    """
    return (
        0.4 * score_relevance(item, query)
        + 0.3 * score_importance(item)
        + 0.2 * score_recency(item, now)
        + 0.1 * score_frequency(item)
    )


STRATEGY_REGISTRY: dict[str, Any] = {
    "recency": score_recency,
    "relevance": score_relevance,
    "importance": score_importance,
    "frequency": score_frequency,
    "hybrid": score_hybrid,
}


def rank_items(
    items: list[MemoryItem],
    query: MemoryQuery,
) -> list[MemoryItem]:
    """根据查询策略对记忆项排序

    Args:
        items: 候选记忆项列表
        query: 查询条件（含 strategy 字段）

    Returns:
        排序后的记忆项列表（按分数降序），不超过 query.limit 条
    """
    if not items:
        return []

    strategy = query.strategy or "hybrid"
    scorer = STRATEGY_REGISTRY.get(strategy, score_hybrid)

    now = time.time()
    scored: list[tuple[float, MemoryItem]] = []

    for item in items:
        if strategy in ("recency", "hybrid"):
            score = scorer(item, query.query, now)  # type: ignore[call-arg]
        elif strategy == "relevance":
            score = score_relevance(item, query.query)
        elif strategy == "importance":
            score = score_importance(item)
        elif strategy == "frequency":
            score = score_frequency(item)
        else:
            score = score_hybrid(item, query.query, now)

        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[: query.limit]]


def filter_items(
    items: list[MemoryItem],
    query: MemoryQuery,
) -> list[MemoryItem]:
    """根据查询条件过滤记忆项

    Args:
        items: 候选记忆项列表
        query: 过滤条件（min_importance, min_strength, tags, memory_type）

    Returns:
        过滤后的记忆项列表
    """
    results = []
    for item in items:
        if query.min_importance > 0 and item.importance < query.min_importance:
            continue
        if query.min_strength > 0 and item.strength < query.min_strength:
            continue
        if query.memory_type and item.memory_type != query.memory_type:
            continue
        if query.tags and not any(t in item.tags for t in query.tags):
            continue
        results.append(item)
    return results