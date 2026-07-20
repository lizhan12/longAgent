"""VectorRAG - 长期向量检索桶

合并原 SemanticMemory + EpisodicMemory + ProceduralMemory，
统一到向量检索后端，按 collection 分区。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .backends.in_memory import InMemoryBackend
from .base import MemoryItem, MemoryQuery, MemoryType

logger = logging.getLogger(__name__)

COLLECTIONS = {"semantic", "episodic", "procedural"}


class VectorRAG:
    """长期向量检索桶

    合并原 Semantic + Episodic + Procedural 的功能，
    使用 ChromaDB 向量检索（优先）或 InMemory 回退，
    按 collection 分区存储。

    Attributes:
        persist_dir: 持久化目录
    """

    def __init__(self, persist_dir: str | Path | None = None) -> None:
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._use_chroma = False
        self._chroma_backends: dict[str, Any] = {}
        self._memory_backends: dict[str, InMemoryBackend] = {}
        self._initialized = False

    def _ensure_backends(self) -> None:
        if self._initialized:
            return

        try:
            from .backends.vector import ChromaDBBackend

            for collection in COLLECTIONS:
                chroma_dir = None
                if self._persist_dir:
                    chroma_dir = self._persist_dir / "chroma"
                    chroma_dir.mkdir(parents=True, exist_ok=True)

                backend = ChromaDBBackend(
                    persist_dir=chroma_dir,
                    collection_name=f"long_{collection}",
                )
                backend._ensure_client()
                if not hasattr(backend, "_fallback"):
                    self._chroma_backends[collection] = backend
                    self._use_chroma = True
                else:
                    self._memory_backends[collection] = backend._fallback
        except Exception:
            logger.debug("ChromaDB 不可用，使用 InMemory 后端")

        for collection in COLLECTIONS:
            if collection not in self._memory_backends and collection not in self._chroma_backends:
                self._memory_backends[collection] = InMemoryBackend(max_size=10000)

        self._initialized = True

    def _get_backend(self, collection: str) -> InMemoryBackend | Any:
        self._ensure_backends()
        if self._use_chroma and collection in self._chroma_backends:
            return self._chroma_backends[collection]
        return self._memory_backends.get(collection, InMemoryBackend())

    async def store(
        self,
        collection: str,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
        **kwargs: object,
    ) -> str:
        """存储到指定 collection"""
        if collection not in COLLECTIONS:
            collection = "semantic"

        type_map = {
            "semantic": MemoryType.SEMANTIC,
            "episodic": MemoryType.EPISODIC,
            "procedural": MemoryType.PROCEDURAL,
        }

        item = MemoryItem(
            content=content,
            memory_type=type_map.get(collection, MemoryType.SEMANTIC),
            importance=importance,
            created_at=time.time(),
            tags=tags or [],
            **kwargs,  # type: ignore[arg-type]
        )

        backend = self._get_backend(collection)
        return await backend.store(item)

    async def search(
        self,
        query: str,
        collections: list[str] | None = None,
        limit: int = 5,
    ) -> list[MemoryItem]:
        """搜索指定 collections"""
        self._ensure_backends()

        target_collections = collections or list(COLLECTIONS)
        all_items: list[MemoryItem] = []

        import asyncio

        search_tasks = []
        for col in target_collections:
            if col in COLLECTIONS:
                backend = self._get_backend(col)
                mq = MemoryQuery(
                    query=query,
                    memory_type=None,
                    limit=limit,
                    strategy="relevance",
                )
                search_tasks.append(backend.search(mq))

        if search_tasks:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    all_items.extend(result)

        all_items.sort(key=lambda x: x.importance, reverse=True)
        return all_items[:limit]

    async def search_all(self, query: str, limit: int = 5) -> list[MemoryItem]:
        """搜索所有 collections"""
        return await self.search(query, collections=None, limit=limit)

    async def recall(self, collection: str, item_id: str) -> MemoryItem | None:
        """根据 ID 回忆"""
        backend = self._get_backend(collection)
        return await backend.recall(item_id)

    async def forget(self, collection: str, item_id: str) -> bool:
        """删除记忆"""
        backend = self._get_backend(collection)
        return await backend.forget(item_id)

    async def count(self, collection: str | None = None) -> int:
        """获取记忆项数量"""
        self._ensure_backends()

        if collection:
            backend = self._get_backend(collection)
            return await backend.count()

        total = 0
        for col in COLLECTIONS:
            backend = self._get_backend(col)
            total += await backend.count()
        return total
