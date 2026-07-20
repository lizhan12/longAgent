"""ToolResultCache — 工具结果缓存

缓存工具执行结果，避免重复调用相同的外部 API。
设计遵循专家建议的三层缓存模型：Token Cache → Tool Cache → Reasoning Cache。

核心特性:
    - 基于 SHA256(tool_name + args) 的精确匹配缓存
    - 分层 TTL（不同工具类型不同过期时间）
    - 时间语义：created_at / expires_at / freshness
    - 缓存摘要（digest），避免重复 summarize 消耗 token
    - 容量限制 + LRU 淘汰

TTL 策略:
    - 搜索引擎 (tavily_search): 5 分钟
    - 文件读取 (read_file/list_files): 30 秒
    - 天气/实时数据: 5 分钟
    - 代码执行: 不缓存（副作用不可重复）
    - 默认: 60 秒

用法:
    cache = ToolResultCache()
    result = cache.get("tavily_search", {"query": "AI 进展"})
    if result is None:
        result = await execute_tool(...)
        cache.put("tavily_search", {"query": "AI 进展"}, result)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CachedToolResult:
    """缓存的工具结果条目"""

    tool_name: str
    args_hash: str
    raw_result: str
    digest: str = ""
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    freshness: str = "fresh"  # fresh / stale / expired
    hit_count: int = 1

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def is_stale(self) -> bool:
        """接近过期但仍可用"""
        remaining = self.expires_at - time.time()
        ttl = self.expires_at - self.created_at
        return ttl > 0 and 0 < remaining < ttl * 0.2

    @staticmethod
    def build_digest(raw: str, max_len: int = 200) -> str:
        """从原始结果提取摘要，避免重复 summarize 消耗 token"""
        raw_stripped = raw.strip()
        if len(raw_stripped) <= max_len:
            return raw_stripped
        # 取前 100 和后 50 字符拼接
        return raw_stripped[:100] + " ... " + raw_stripped[-50:]


class ToolResultCache:
    """工具结果缓存

    设计要点:
        - Key: SHA256(tool_name + 排序后的 args JSON)
        - Value: CachedToolResult (含摘要和时间元信息)
        - TTL: 按工具类型分层
        - 容量: LRU 淘汰策略
    """

    # TTL 配置（秒）
    _DEFAULT_TTL = 60
    _TTL_MAP: dict[str, int] = {
        "tavily_search": 300,        # 搜索: 5 分钟
        "read_file": 30,             # 文件: 30 秒
        "list_files": 30,
        "get_current_time": 10,      # 时间: 10 秒
        "read_skill_md": 300,        # Skill: 5 分钟（不常变）
    }

    # 不应被缓存的工具（副作用不可重复）
    _NO_CACHE_TOOLS = frozenset({
        "execute_code",
        "execute_file",
        "write_file",
        "delete_file",
        "echo",
        "reverse",
    })

    def __init__(
        self,
        max_size: int = 200,
        ttl_overrides: dict[str, int] | None = None,
        persist_path: str | Path | None = None,
    ) -> None:
        self._max_size = max_size
        self._ttl_map = dict(self._TTL_MAP)
        if ttl_overrides:
            self._ttl_map.update(ttl_overrides)

        self._cache: OrderedDict[str, CachedToolResult] = OrderedDict()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path is not None and self._persist_path.exists():
            self._load_from_disk()

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @property
    def hit_rate(self) -> float:
        total = self._stats["hits"] + self._stats["misses"]
        return self._stats["hits"] / max(total, 1)

    @staticmethod
    def make_key(tool_name: str, arguments: dict[str, Any]) -> str:
        """生成缓存键"""
        normalized = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        raw = f"{tool_name}:{normalized}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, tool_name: str, arguments: dict[str, Any]) -> CachedToolResult | None:
        """获取缓存的结果

        Returns:
            CachedToolResult 如果命中且未过期，否则 None
        """
        if tool_name in self._NO_CACHE_TOOLS:
            self._stats["misses"] += 1
            return None

        key = self.make_key(tool_name, arguments)
        entry = self._cache.get(key)

        if entry is None:
            self._stats["misses"] += 1
            logger.debug(
                "ToolResultCache MISS: %s(%s)",
                tool_name,
                json.dumps(arguments, ensure_ascii=False)[:80],
            )
            return None

        if entry.is_expired:
            del self._cache[key]
            self._stats["misses"] += 1
            logger.debug(
                "ToolResultCache EXPIRED: %s(%s)",
                tool_name,
                json.dumps(arguments, ensure_ascii=False)[:80],
            )
            return None

        entry.hit_count += 1
        if entry.is_stale:
            logger.debug(
                "ToolResultCache STALE (using cached): %s(%s)",
                tool_name,
                json.dumps(arguments, ensure_ascii=False)[:80],
            )

        self._stats["hits"] += 1
        self._cache.move_to_end(key)
        logger.debug(
            "ToolResultCache HIT: %s(%s), age=%.1fs",
            tool_name,
            json.dumps(arguments, ensure_ascii=False)[:80],
            time.time() - entry.created_at,
        )
        return entry

    def put(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        ttl: int | None = None,
    ) -> None:
        """存入缓存

        Args:
            tool_name: 工具名称
            arguments: 调用参数
            result: 执行结果
            ttl: 自定义 TTL，默认使用工具类型映射
        """
        if tool_name in self._NO_CACHE_TOOLS:
            return

        key = self.make_key(tool_name, arguments)

        # LRU 淘汰
        if len(self._cache) >= self._max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            self._stats["evictions"] += 1
            logger.debug("ToolResultCache EVICT: %s", evicted_key[:16])

        actual_ttl = ttl if ttl is not None else self._ttl_map.get(tool_name, self._DEFAULT_TTL)
        now = time.time()

        entry = CachedToolResult(
            tool_name=tool_name,
            args_hash=key[:16],
            raw_result=result,
            digest=CachedToolResult.build_digest(result),
            created_at=now,
            expires_at=now + actual_ttl,
            freshness="fresh",
        )

        self._cache[key] = entry

        if self._persist_path is not None:
            self._save_to_disk()

        logger.debug(
            "ToolResultCache PUT: %s(%s), ttl=%ds",
            tool_name,
            json.dumps(arguments, ensure_ascii=False)[:80],
            actual_ttl,
        )

    def invalidate(self, tool_name: str, arguments: dict[str, Any] | None = None) -> None:
        """使缓存失效

        Args:
            tool_name: 工具名称
            arguments: 如果提供，精确失效；否则失效该工具的全部缓存
        """
        if arguments is not None:
            key = self.make_key(tool_name, arguments)
            if key in self._cache:
                del self._cache[key]
                logger.debug("ToolResultCache INVALIDATE: %s(%s)", tool_name, key[:16])
        else:
            to_remove = [
                k for k, v in self._cache.items()
                if v.tool_name == tool_name
            ]
            for k in to_remove:
                del self._cache[k]
            logger.debug(
                "ToolResultCache INVALIDATE ALL: %s (%d entries)", tool_name, len(to_remove),
            )

    def clear(self) -> None:
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get_age_seconds(self, tool_name: str, arguments: dict[str, Any]) -> float | None:
        """获取指定结果的缓存年龄（秒），用于判断新鲜度"""
        key = self.make_key(tool_name, arguments)
        entry = self._cache.get(key)
        if entry is None:
            return None
        return time.time() - entry.created_at

    @property
    def size(self) -> int:
        return len(self._cache)

    def _save_to_disk(self) -> None:
        """持久化到磁盘（JSON 文件）"""
        if self._persist_path is None:
            return
        try:
            data = {
                "stats": dict(self._stats),
                "entries": [
                    {
                        "tool_name": v.tool_name,
                        "args_hash": v.args_hash,
                        "raw_result": v.raw_result,
                        "digest": v.digest,
                        "created_at": v.created_at,
                        "expires_at": v.expires_at,
                        "freshness": v.freshness,
                        "hit_count": v.hit_count,
                    }
                    for v in self._cache.values()
                ],
            }
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception:
            logger.debug("ToolResultCache 持久化失败", exc_info=True)

    def _load_from_disk(self) -> None:
        """从磁盘恢复缓存"""
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text())
            self._stats = data.get("stats", {"hits": 0, "misses": 0, "evictions": 0})
            now = time.time()
            loaded = 0
            expired = 0
            for entry_data in data.get("entries", []):
                if entry_data["expires_at"] <= now:
                    expired += 1
                    continue
                entry = CachedToolResult(
                    tool_name=entry_data["tool_name"],
                    args_hash=entry_data["args_hash"],
                    raw_result=entry_data["raw_result"],
                    digest=entry_data["digest"],
                    created_at=entry_data["created_at"],
                    expires_at=entry_data["expires_at"],
                    freshness=entry_data["freshness"],
                    hit_count=entry_data["hit_count"],
                )
                self._cache[self.make_key(entry.tool_name, {})] = entry  # 简化
                loaded += 1
            logger.info(
                "ToolResultCache 已从磁盘加载: %d 条有效, %d 条已过期",
                loaded, expired,
            )
        except Exception:
            logger.debug("ToolResultCache 加载失败", exc_info=True)