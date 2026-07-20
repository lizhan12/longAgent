"""缓存模块

提供 LLM 响应缓存，避免重复 API 调用。
精确匹配缓存，基于完整对话上下文 + 模型 + 温度 + 工具哈希。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class LLMResponseCache:
    """LLM 响应精确匹配缓存

    缓存键: hash(完整消息上下文 + 模型 + 温度 + tools_hash)
    缓存值: LLMResponse
    TTL: 60 秒
    容量: LRU 100 条
    """

    def __init__(self, ttl: float = 60.0, max_size: int = 100) -> None:
        self._ttl = ttl
        self._max_size = max_size
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    def _make_key(
        self,
        messages: list[Any],
        model: str = "",
        temperature: float = 0.7,
        tools_hash: str = "",
    ) -> str:
        # 对完整消息列表进行哈希，而非仅最后一条用户消息
        msg_parts: list[str] = []
        for m in messages:
            role = getattr(m, "role", m.get("role", "") if isinstance(m, dict) else "")
            content = getattr(m, "content", m.get("content", "") if isinstance(m, dict) else "")
            msg_parts.append(f"{role}:{content}")
        messages_hash = hashlib.sha256("|".join(msg_parts).encode()).hexdigest()[:32]

        raw = f"{messages_hash}|{model}|{temperature}|{tools_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _make_tool_cache_key(self, tools: list[dict[str, Any]]) -> str:
        """根据工具列表生成哈希键，用于区分工具调用与非工具调用请求"""
        if not tools:
            return ""
        tools_str = json.dumps(tools, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(tools_str.encode()).hexdigest()[:16]

    def get(
        self,
        messages: list[Any],
        model: str = "",
        temperature: float = 0.7,
        tools_hash: str = "",
    ) -> Any | None:
        key = self._make_key(messages, model, temperature, tools_hash)
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        ts, value = entry
        if time.time() - ts > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None

        self._cache.move_to_end(key)
        self._hits += 1
        logger.debug("LLM 缓存命中: key=%s", key[:8])
        return value

    def put(
        self,
        messages: list[Any],
        response: Any,
        model: str = "",
        temperature: float = 0.7,
        tools_hash: str = "",
    ) -> None:
        key = self._make_key(messages, model, temperature, tools_hash)
        self._cache[key] = (time.time(), response)
        self._cache.move_to_end(key)

        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def invalidate(self, messages: list[Any], model: str = "", temperature: float = 0.7, tools_hash: str = "") -> None:
        key = self._make_key(messages, model, temperature, tools_hash)
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate * 100:.1f}%",
            "ttl": self._ttl,
        }
