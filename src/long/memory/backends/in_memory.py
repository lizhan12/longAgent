"""内存后端

使用 RingBuffer + Dict 实现的内存存储后端。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

from ..base import MemoryItem, MemoryQuery, MemoryStore


class InMemoryBackend(MemoryStore):
    """内存后端

    使用 OrderedDict 实现的内存存储，支持容量限制（类似 RingBuffer）。
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._items: OrderedDict[str, MemoryItem] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    async def store(self, item: MemoryItem) -> str:
        with self._lock:
            if len(self._items) >= self._max_size:
                self._items.popitem(last=False)

            if item.created_at == 0.0:
                item = item.model_copy(update={"created_at": time.time()})

            self._items[item.id] = item
            self._items.move_to_end(item.id)
            return item.id

    async def recall(self, item_id: str) -> MemoryItem | None:
        with self._lock:
            return self._items.get(item_id)

    async def forget(self, item_id: str) -> bool:
        with self._lock:
            if item_id in self._items:
                del self._items[item_id]
                return True
            return False

    async def search(self, query: MemoryQuery) -> list[MemoryItem]:
        with self._lock:
            items = list(self._items.values())

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

            score = self._compute_relevance(item, query.query)
            if score > 0:
                results.append((score, item))

        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results[: query.limit]]

    async def count(self) -> int:
        with self._lock:
            return len(self._items)

    def _compute_relevance(self, item: MemoryItem, query: str) -> float:
        """计算相关性分数（简单文本匹配）"""
        query_lower = query.lower()

        if query_lower == "*" or query_lower == "":
            return 1.0

        content_lower = item.content.lower()

        if query_lower in content_lower:
            return 1.0

        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        overlap = query_words & content_words

        if overlap:
            return len(overlap) / max(len(query_words), 1)

        return 0.0
