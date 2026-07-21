"""SQLite 存储后端

使用 SQLite 实现记忆持久化，支持结构化存储、重要性加权查询和标记检索。
适用于 EpisodicMemory（情景记忆）和 ProceduralMemory（过程记忆）。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from ..base import MemoryItem, MemoryQuery, MemoryStore

logger = logging.getLogger(__name__)

# SQLite 连接池（线程级缓存）
_local_connections: threading.local = threading.local()


def _get_connection(db_path: str) -> sqlite3.Connection:
    """获取线程级 SQLite 连接"""
    if not hasattr(_local_connections, db_path):
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.row_factory = sqlite3.Row
        setattr(_local_connections, db_path, conn)
    return getattr(_local_connections, db_path)


class SQLiteBackend(MemoryStore):
    """SQLite 存储后端

    支持 CRUD、重要性加权查询、标签过滤、分页和按强度衰减清理。

    Attributes:
        db_path: SQLite 数据库文件路径
        table_name: 表名
        max_size: 最大记忆项数（超出时删除最旧的）
    """

    def __init__(
        self,
        db_path: str | Path,
        table_name: str = "memories",
        max_size: int = 10000,
    ) -> None:
        self._db_path = str(Path(db_path).resolve())
        self._table_name = table_name
        self._max_size = max_size
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库表"""
        conn = _get_connection(self._db_path)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{{}}',
                importance REAL DEFAULT 0.5,
                created_at REAL DEFAULT 0.0,
                strength REAL DEFAULT 1.0,
                memory_type TEXT DEFAULT 'short_term',
                tags TEXT DEFAULT '[]',
                retrieval_count INTEGER DEFAULT 0
            )
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_strength
            ON {self._table_name}(strength)
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_importance
            ON {self._table_name}(importance)
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_created
            ON {self._table_name}(created_at)
        """)
        conn.commit()

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        """将 SQLite 行转换为 MemoryItem"""
        return MemoryItem(
            id=row["id"],
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            importance=row["importance"],
            created_at=row["created_at"],
            strength=row["strength"],
            memory_type=row["memory_type"],  # type: ignore[arg-type]
            tags=json.loads(row["tags"]),
            retrieval_count=row["retrieval_count"],
        )

    async def store(self, item: MemoryItem) -> str:
        """存储记忆项（插入或更新）"""
        created_at = item.created_at if item.created_at > 0 else time.time()
        with self._lock:
            conn = _get_connection(self._db_path)

            # 容量检查：超出时删除最旧的
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM {self._table_name}"
            )
            count = cursor.fetchone()[0]
            if count >= self._max_size:
                conn.execute(
                    f"DELETE FROM {self._table_name} "
                    f"WHERE id IN ("
                    f"  SELECT id FROM {self._table_name} "
                    f"  ORDER BY created_at ASC LIMIT {count - self._max_size + 1}"
                    f")"
                )

            conn.execute(
                f"""
                INSERT OR REPLACE INTO {self._table_name}
                (id, content, metadata, importance, created_at, strength, memory_type, tags, retrieval_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.content,
                    json.dumps(item.metadata, ensure_ascii=False),
                    item.importance,
                    created_at,
                    item.strength,
                    item.memory_type.value,
                    json.dumps(item.tags),
                    item.retrieval_count,
                ),
            )
            conn.commit()
        return item.id

    async def recall(self, item_id: str) -> MemoryItem | None:
        """根据 ID 回忆记忆项"""
        conn = _get_connection(self._db_path)
        cursor = conn.execute(
            f"SELECT * FROM {self._table_name} WHERE id = ?",
            (item_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        # 增加检索计数
        conn.execute(
            f"UPDATE {self._table_name} SET retrieval_count = retrieval_count + 1 WHERE id = ?",
            (item_id,),
        )
        conn.commit()

        return self._row_to_item(row)

    async def forget(self, item_id: str) -> bool:
        """删除记忆项"""
        with self._lock:
            conn = _get_connection(self._db_path)
            cursor = conn.execute(
                f"DELETE FROM {self._table_name} WHERE id = ?",
                (item_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    async def search(self, query: MemoryQuery) -> list[MemoryItem]:
        """搜索记忆项（支持重要性/强度过滤、标签过滤、分页）"""
        conn = _get_connection(self._db_path)

        conditions: list[str] = ["1=1"]
        params: list[Any] = []

        if query.min_importance > 0:
            conditions.append("importance >= ?")
            params.append(query.min_importance)

        if query.min_strength > 0:
            conditions.append("strength >= ?")
            params.append(query.min_strength)

        if query.tags:
            for tag in query.tags:
                conditions.append("tags LIKE ?")
                params.append(f"%{tag}%")

        where_clause = " AND ".join(conditions)
        cursor = conn.execute(
            f"SELECT * FROM {self._table_name} WHERE {where_clause} ORDER BY importance DESC, strength DESC",
            params,
        )
        rows = cursor.fetchall()
        items = [self._row_to_item(row) for row in rows]

        # 按相关性排序
        if query.query and query.query != "*":
            scored = [(self._compute_relevance(item, query.query), item) for item in items]
            scored.sort(key=lambda x: x[0], reverse=True)
            items = [item for score, item in scored if score > 0]

        return items[: query.limit]

    async def search_by_tags(
        self,
        tags: list[str],
        limit: int = 10,
    ) -> list[MemoryItem]:
        """按标签搜索"""
        conn = _get_connection(self._db_path)
        placeholders = ",".join("?" for _ in tags)
        cursor = conn.execute(
            f"SELECT * FROM {self._table_name} WHERE tags LIKE ? "
            f"ORDER BY importance DESC, strength DESC LIMIT ?",
            (f"%{tags[0]}%", limit),
        )
        return [self._row_to_item(row) for row in cursor.fetchall()]

    async def count(self) -> int:
        """获取记忆项数量"""
        conn = _get_connection(self._db_path)
        cursor = conn.execute(f"SELECT COUNT(*) FROM {self._table_name}")
        return cursor.fetchone()[0]

    async def count_by_type(self, memory_type: str) -> int:
        """按类型计数"""
        conn = _get_connection(self._db_path)
        cursor = conn.execute(
            f"SELECT COUNT(*) FROM {self._table_name} WHERE memory_type = ?",
            (memory_type,),
        )
        return cursor.fetchone()[0]

    async def get_all(self, limit: int = 100) -> list[MemoryItem]:
        """获取所有记忆项（按重要性降序）"""
        conn = _get_connection(self._db_path)
        cursor = conn.execute(
            f"SELECT * FROM {self._table_name} ORDER BY importance DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_item(row) for row in cursor.fetchall()]

    async def cleanup_by_strength(self, min_strength: float = 0.05) -> int:
        """删除强度低于阈值的记忆项

        Returns:
            删除的数量
        """
        with self._lock:
            conn = _get_connection(self._db_path)
            cursor = conn.execute(
                f"DELETE FROM {self._table_name} WHERE strength < ?",
                (min_strength,),
            )
            conn.commit()
            deleted = cursor.rowcount
            if deleted > 0:
                logger.info("SQLiteBackend 清理: 删除了 %d 条低强度记忆", deleted)
            return deleted

    async def vacuum(self) -> None:
        """回收数据库空间"""
        with self._lock:
            conn = _get_connection(self._db_path)
            conn.execute("VACUUM")
            conn.commit()

    def _compute_relevance(self, item: MemoryItem, query: str) -> float:
        """计算文本相关性分数"""
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