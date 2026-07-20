"""向量存储后端

使用 ChromaDB 实现的向量存储。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..base import MemoryItem, MemoryQuery, MemoryStore, MemoryType

logger = logging.getLogger(__name__)


class ChromaDBBackend(MemoryStore):
    """ChromaDB 向量存储后端

    使用 ChromaDB 进行向量相似度检索。

    Attributes:
        collection_name: 集合名称
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = "long_memory",
    ) -> None:
        self._persist_dir = str(persist_dir) if persist_dir else None
        self._collection_name = collection_name
        self._client = None
        self._collection = None

    def _ensure_client(self) -> None:
        """确保 ChromaDB 客户端已初始化"""
        if self._client is not None:
            return

        try:
            import chromadb

            if self._persist_dir:
                self._client = chromadb.PersistentClient(path=self._persist_dir)
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            logger.debug("chromadb not installed, using in-memory fallback")
            from .in_memory import InMemoryBackend

            self._fallback = InMemoryBackend()

    async def store(self, item: MemoryItem) -> str:
        self._ensure_client()

        if hasattr(self, "_fallback"):
            return await self._fallback.store(item)

        import time

        if item.created_at == 0.0:
            item = item.model_copy(update={"created_at": time.time()})

        metadata = {
            "importance": item.importance,
            "created_at": item.created_at,
            "strength": item.strength,
            "memory_type": item.memory_type.value,
        }

        self._collection.upsert(
            ids=[item.id],
            documents=[item.content],
            metadatas=[metadata],
        )

        return item.id

    async def recall(self, item_id: str) -> MemoryItem | None:
        self._ensure_client()

        if hasattr(self, "_fallback"):
            return await self._fallback.recall(item_id)

        result = self._collection.get(ids=[item_id])
        if not result["ids"]:
            return None

        return self._result_to_items(result)[0]

    async def forget(self, item_id: str) -> bool:
        self._ensure_client()

        if hasattr(self, "_fallback"):
            return await self._fallback.forget(item_id)

        try:
            self._collection.delete(ids=[item_id])
            return True
        except Exception:
            return False

    async def search(self, query: MemoryQuery) -> list[MemoryItem]:
        self._ensure_client()

        if hasattr(self, "_fallback"):
            return await self._fallback.search(query)

        where_filter: dict[str, Any] = {}
        if query.memory_type:
            where_filter["memory_type"] = query.memory_type.value
        if query.min_importance > 0:
            where_filter["importance"] = {"$gte": query.min_importance}

        result = self._collection.query(
            query_texts=[query.query],
            n_results=min(query.limit, 100),
            where=where_filter or None,
        )

        if not result["ids"] or not result["ids"][0]:
            return []

        return self._query_result_to_items(result)[:query.limit]

    async def count(self) -> int:
        self._ensure_client()

        if hasattr(self, "_fallback"):
            return await self._fallback.count()

        return self._collection.count()

    def _result_to_items(self, result: dict) -> list[MemoryItem]:
        items = []
        for i, doc_id in enumerate(result["ids"]):
            metadata = result["metadatas"][i] if result["metadatas"] else {}
            items.append(
                MemoryItem(
                    id=doc_id,
                    content=result["documents"][i] if result["documents"] else "",
                    metadata=metadata,
                    importance=metadata.get("importance", 0.5),
                    created_at=metadata.get("created_at", 0.0),
                    strength=metadata.get("strength", 1.0),
                    memory_type=MemoryType(metadata.get("memory_type", "short_term")),
                )
            )
        return items

    def _query_result_to_items(self, result: dict) -> list[MemoryItem]:
        items = []
        if not result["ids"] or not result["ids"][0]:
            return items

        for i, doc_id in enumerate(result["ids"][0]):
            metadata = result["metadatas"][0][i] if result["metadatas"] else {}
            items.append(
                MemoryItem(
                    id=doc_id,
                    content=result["documents"][0][i] if result["documents"] else "",
                    metadata=metadata,
                    importance=metadata.get("importance", 0.5),
                    created_at=metadata.get("created_at", 0.0),
                    strength=metadata.get("strength", 1.0),
                    memory_type=MemoryType(metadata.get("memory_type", "short_term")),
                )
            )
        return items

