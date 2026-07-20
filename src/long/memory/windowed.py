"""WindowedMemory - 短期对话窗口

合并原 ShortTermMemory + WorkingMemory，
提供消息窗口 + 任务状态 KV 的统一接口。

支持 token 预算动态压缩：当对话 token 数超出上限时，
自动压缩中间消息，保留头尾关键消息。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

from .base import MemoryItem, MemoryType

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文1.5字/token，英文4字/token）"""
    cn_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    en_chars = len(text) - cn_chars
    return int(cn_chars / 1.5 + en_chars / 4) if text else 0


class WindowedMemory:
    """短期对话窗口

    合并原 ShortTerm + Working 的功能：
    - 消息窗口：最近 N 轮对话（RingBuffer）
    - 任务状态 KV：当前任务的临时状态存储
    - Token 预算：超出上限时自动压缩中间消息

    Attributes:
        max_messages: 最大消息数
        max_task_items: 每个任务最大状态项数
        max_tokens: token 预算上限，超出时触发自动压缩
        auto_compress: 是否启用自动压缩
    """

    def __init__(
        self,
        max_messages: int = 128,
        max_task_items: int = 50,
        max_tokens: int = 8000,
        auto_compress: bool = True,
    ) -> None:
        self.max_messages = max_messages
        self.max_task_items = max_task_items
        self.max_tokens = max_tokens
        self.auto_compress = auto_compress
        self._messages: deque[MemoryItem] = deque(maxlen=max_messages)
        self._task_state: dict[str, Any] = {}
        self._task_items: dict[str, MemoryItem] = {}
        self._lock = threading.Lock()
        self._compressor: Any | None = None  # SemanticCompressor，延迟注入

    def set_compressor(self, compressor: Any) -> None:
        """注入语义压缩器（避免循环依赖）"""
        self._compressor = compressor

    def get_token_count(self) -> int:
        """计算当前所有消息的总 token 数"""
        with self._lock:
            return sum(_estimate_tokens(m.content) for m in self._messages)

    async def add_message(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """添加对话消息"""
        item = MemoryItem(
            content=content,
            memory_type=MemoryType.SHORT_TERM,
            created_at=time.time(),
            metadata={"role": role, **(metadata or {})},
        )
        with self._lock:
            self._messages.append(item)

        # 检查 token 预算，超出时自动压缩
        if self.auto_compress and self.get_token_count() > self.max_tokens:
            await self._auto_compress()

        return item.id

    async def _auto_compress(self) -> None:
        """自动压缩中间消息，保留头尾关键消息

        策略：保留前3条（system/初始指令）+ 最后5条（最近上下文），
        对中间部分进行结构化压缩。
        """
        with self._lock:
            msgs = list(self._messages)

        if len(msgs) <= 8:
            return  # 消息太少，不压缩

        head_count = 3
        tail_count = 5
        head = msgs[:head_count]
        tail = msgs[-tail_count:]
        middle = msgs[head_count:-tail_count]

        if not middle:
            return

        # 压缩中间消息
        compressed_parts: list[str] = []
        for item in middle:
            content = item.content
            if len(content) <= 100:
                compressed_parts.append(content)
            else:
                # 结构化提取：首句 + 关键数字 + 尾句
                first_dot = content.find("。")
                first_newline = content.find("\n")
                end = min(
                    idx for idx in [first_dot, first_newline, 80] if idx > 0
                )
                first_sentence = content[:end].strip()

                last_newline = content.rfind("\n")
                last_sentence = content[last_newline:].strip()[:80] if last_newline > 0 else ""

                compressed = first_sentence
                if last_sentence and last_sentence != first_sentence:
                    compressed += " ... " + last_sentence
                compressed_parts.append(compressed)

        summary = "[对话摘要] " + " | ".join(compressed_parts)
        # 截断防止摘要本身过长
        if _estimate_tokens(summary) > self.max_tokens // 4:
            summary = summary[: self.max_tokens // 2]

        # 重建消息队列
        summary_item = MemoryItem(
            content=summary,
            memory_type=MemoryType.SHORT_TERM,
            created_at=time.time(),
            metadata={"role": "system", "compressed": True},
        )

        with self._lock:
            self._messages.clear()
            for item in head:
                self._messages.append(item)
            self._messages.append(summary_item)
            for item in tail:
                self._messages.append(item)

        logger.info(
            "WindowedMemory 自动压缩: %d条中间消息 → 1条摘要 (token: %d→%d)",
            len(middle),
            sum(_estimate_tokens(m.content) for m in middle),
            _estimate_tokens(summary),
        )

    async def get_messages(self, last_n: int | None = None) -> list[dict[str, str]]:
        """获取对话消息列表

        Args:
            last_n: 获取最近 N 条，None 表示全部

        Returns:
            [{"role": "...", "content": "..."}] 格式的消息列表
        """
        with self._lock:
            msgs = list(self._messages)

        if last_n is not None:
            msgs = msgs[-last_n:]

        return [
            {"role": m.metadata.get("role", "user"), "content": m.content}
            for m in msgs
        ]

    async def set_task_state(self, key: str, value: Any) -> None:
        """设置任务状态 KV"""
        with self._lock:
            if len(self._task_state) >= self.max_task_items and key not in self._task_state:
                oldest_key = next(iter(self._task_state))
                del self._task_state[oldest_key]
            self._task_state[key] = value

    async def get_task_state(self, key: str, default: Any = None) -> Any:
        """获取任务状态"""
        with self._lock:
            return self._task_state.get(key, default)

    async def get_all_task_state(self) -> dict[str, Any]:
        """获取全部任务状态"""
        with self._lock:
            return dict(self._task_state)

    async def clear_task_state(self) -> None:
        """清空任务状态"""
        with self._lock:
            self._task_state.clear()

    async def store(self, content: str, **kwargs: object) -> str:
        """兼容旧接口：存储短期记忆"""
        item = MemoryItem(
            content=content,
            memory_type=MemoryType.SHORT_TERM,
            created_at=time.time(),
            **kwargs,  # type: ignore[arg-type]
        )
        with self._lock:
            self._messages.append(item)
        return item.id

    async def recall(self, item_id: str) -> MemoryItem | None:
        """根据 ID 回忆"""
        with self._lock:
            for item in self._messages:
                if item.id == item_id:
                    return item
        return None

    async def forget(self, item_id: str) -> bool:
        """删除记忆"""
        with self._lock:
            for i, item in enumerate(self._messages):
                if item.id == item_id:
                    del self._messages[i]
                    return True
        return False

    async def search(self, query: str, limit: int = 10) -> list[MemoryItem]:
        """搜索短期记忆（简单文本匹配）"""
        with self._lock:
            items = list(self._messages)

        query_lower = query.lower()
        results = []
        for item in reversed(items):
            if not query_lower or query_lower in item.content.lower():
                results.append(item)
            if len(results) >= limit:
                break
        return results

    async def get_recent(self, limit: int = 10) -> list[MemoryItem]:
        """获取最近的记忆"""
        with self._lock:
            items = list(self._messages)
        return list(reversed(items[-limit:]))

    async def count(self) -> int:
        """获取消息数量"""
        with self._lock:
            return len(self._messages)

    async def clear(self) -> None:
        """清空所有消息和状态"""
        with self._lock:
            self._messages.clear()
            self._task_state.clear()
