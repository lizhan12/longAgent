"""记忆控制器

三栖记忆门面：WindowedMemory + VectorRAG + SemanticCompressor。
简化自原五层记忆架构，移除复杂的晋升/衰减逻辑。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .base import MemoryItem, MemoryType
from .compressor import SemanticCompressor, _estimate_tokens
from .vector_rag import VectorRAG
from .windowed import WindowedMemory

logger = logging.getLogger(__name__)

_LOW_VALUE_PATTERNS = [
    "好的", "嗯", "哦", "是的", "对", "行", "ok", "OK", "thanks", "谢谢",
    "你好", "hi", "hello", "嗨", "嘿",
]
_HIGH_VALUE_KEYWORDS = [
    "喜欢", "偏好", "习惯", "默认", "不用", "不要", "以后", "一直",
    "框架", "语言", "技术", "工具", "项目", "架构", "部署", "配置",
    "FastAPI", "React", "Python", "Docker", "Kubernetes", "vLLM",
    "决定", "选择", "方案", "原则", "规范", "风格",
]


def _compute_admission_score(content: str, importance: float) -> float:
    stripped = content.strip()
    if not stripped:
        return 0.0
    if stripped.lower() in [p.lower() for p in _LOW_VALUE_PATTERNS]:
        return 0.1
    if len(stripped) < 5:
        return 0.15
    high_value_hits = sum(1 for kw in _HIGH_VALUE_KEYWORDS if kw.lower() in stripped.lower())
    task_relevance = min(1.0, 0.3 + high_value_hits * 0.25)
    return importance * task_relevance


_TYPE_TO_COLLECTION = {
    MemoryType.SEMANTIC: "semantic",
    MemoryType.EPISODIC: "episodic",
    MemoryType.PROCEDURAL: "procedural",
}


class MemoryController:
    """三栖记忆门面

    WindowedMemory（短期对话窗口）+ VectorRAG（长期向量检索）+ SemanticCompressor（语义压缩）。

    Attributes:
        windowed: 短期对话窗口
        vector_rag: 长期向量检索桶
        compressor: 语义压缩器
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        max_messages: int = 128,
        max_task_items: int = 50,
        compressor_max_tokens: int = 4000,
        llm_client: Any | None = None,
        max_tokens: int = 8000,
        auto_compress: bool = True,
        summarize_threshold: int = 20,
        auto_summarize: bool = True,
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else None
        self._llm_client = llm_client
        self._summarize_threshold = summarize_threshold
        self._auto_summarize = auto_summarize

        self.windowed = WindowedMemory(
            max_messages=max_messages,
            max_task_items=max_task_items,
            max_tokens=max_tokens,
            auto_compress=auto_compress,
        )

        self.vector_rag = VectorRAG(persist_dir=self._data_dir)

        self.compressor = SemanticCompressor(
            max_tokens=compressor_max_tokens,
            llm_client=llm_client,
        )

        # 注入压缩器到 WindowedMemory
        self.windowed.set_compressor(self.compressor)

        self.short_term = self.windowed
        self.working = self.windowed
        self.semantic = _SemanticCompat(self.vector_rag)
        self.episodic = _EpisodicCompat(self.vector_rag)
        self.procedural = _ProceduralCompat(self.vector_rag)

    async def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        importance: float = 0.5,
        **kwargs: Any,
    ) -> str:
        """统一存储接口"""
        admission_score = _compute_admission_score(content, importance)

        if memory_type in (MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL):
            if admission_score < 0.3:
                logger.debug("记忆准入拒绝: score=%.2f", admission_score)
                memory_type = MemoryType.SHORT_TERM
                importance = max(importance, admission_score)
            else:
                await self._detect_conflicts(content, memory_type)

        if memory_type == MemoryType.SHORT_TERM:
            return await self.windowed.store(content, importance=importance, **kwargs)
        elif memory_type == MemoryType.WORKING:
            return await self.windowed.store(content, importance=importance, **kwargs)
        else:
            collection = _TYPE_TO_COLLECTION.get(memory_type, "semantic")
            return await self.vector_rag.store(
                collection, content, importance=importance, **kwargs
            )

    async def _detect_conflicts(self, content: str, memory_type: MemoryType) -> None:
        conflict_keywords = ["不用", "不要", "不喜欢", "改用", "换成", "现在用", "不再"]
        if not any(kw in content for kw in conflict_keywords):
            return
        try:
            collection = _TYPE_TO_COLLECTION.get(memory_type, "semantic")
            search_terms = [w for w in content.split() if len(w) > 2]
            if not search_terms:
                return
            query = " ".join(search_terms[:5])
            existing = await self.vector_rag.search(query, collections=[collection], limit=5)
            for item in existing:
                if item.strength > 0.3:
                    item.strength = 0.3
                    item.metadata["conflict_detected"] = True
                    logger.info("记忆冲突标记: %s... → strength=0.3", item.content[:40])
        except Exception as e:
            logger.debug("冲突检测失败: %s", e)

    async def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = 10,
        strategy: str = "hybrid",
    ) -> list[MemoryItem]:
        """跨类型检索"""
        if memory_type:
            if memory_type in (MemoryType.SHORT_TERM, MemoryType.WORKING):
                return await self.windowed.search(query, limit=limit)
            collection = _TYPE_TO_COLLECTION.get(memory_type, "semantic")
            return await self.vector_rag.search(query, collections=[collection], limit=limit)

        import asyncio

        windowed_results, rag_results = await asyncio.gather(
            self.windowed.search(query, limit=limit),
            self.vector_rag.search_all(query, limit=limit),
        )

        all_items = list(windowed_results) + list(rag_results)
        all_items.sort(key=lambda x: x.importance, reverse=True)
        return all_items[:limit]

    async def get_context(
        self,
        query: str,
        max_tokens: int = 4000,
    ) -> dict[str, Any]:
        """获取压缩后的上下文

        Returns:
            {"messages": [...], "task_state": {...}, "relevant_memories": [...]}
        """
        messages = await self.windowed.get_messages()
        task_state = await self.windowed.get_all_task_state()
        relevant = await self.vector_rag.search_all(query, limit=5)

        compressed_messages = await self.compressor.compress(messages, max_tokens=max_tokens)
        compressed_memories = self.compressor.compress_items(relevant, max_tokens=max_tokens // 4)

        return {
            "messages": compressed_messages,
            "task_state": task_state,
            "relevant_memories": [
                {"content": m.content, "importance": m.importance, "type": m.memory_type.value}
                for m in compressed_memories
            ],
        }

    async def add_message(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """添加对话消息，自动触发摘要"""
        item_id = await self.windowed.add_message(role, content, metadata)

        # 检查是否需要自动摘要
        if self._auto_summarize:
            count = await self.windowed.count()
            if count >= self._summarize_threshold:
                await self._auto_summarize()

        return item_id

    async def _auto_summarize(self) -> None:
        """自动摘要：保留头尾关键消息，对中间部分生成摘要"""
        messages = await self.windowed.get_messages()
        if len(messages) <= 8:
            return

        head = messages[:3]
        tail = messages[-5:]
        middle = messages[3:-5]

        if not middle:
            return

        summary = await self.summarize_conversation(middle, self._llm_client)
        if summary:
            # 用摘要替换中间消息
            await self.windowed.clear()
            for msg in head:
                await self.windowed.add_message(msg["role"], msg["content"])
            await self.windowed.add_message(
                "system", f"[对话摘要] {summary}", {"compressed": True}
            )
            for msg in tail:
                await self.windowed.add_message(msg["role"], msg["content"])
            logger.info(
                "自动摘要完成: %d条中间消息 → 1条摘要", len(middle)
            )

    async def set_task_state(self, key: str, value: Any) -> None:
        """设置任务状态"""
        await self.windowed.set_task_state(key, value)

    async def get_task_state(self, key: str, default: Any = None) -> Any:
        """获取任务状态"""
        return await self.windowed.get_task_state(key, default)

    async def promote(self, item_id: str) -> bool:
        """将短期记忆提升为长期记忆"""
        item = None
        for m in await self.windowed.get_recent(limit=1000):
            if m.id == item_id:
                item = m
                break

        if item is None:
            return False

        if item.importance < 0.5:
            return False

        collection = "semantic" if item.importance >= 0.8 else "episodic"
        await self.vector_rag.store(
            collection,
            item.content,
            importance=item.importance,
            tags=item.tags,
        )
        await self.windowed.forget(item_id)
        logger.info("记忆 %s 已提升为 %s 记忆", item_id, collection)
        return True

    async def auto_promote(self) -> int:
        """自动提升高重要性短期记忆"""
        promoted = 0
        items = await self.windowed.get_recent(limit=1000)
        for item in items:
            if item.importance >= 0.5:
                if await self.promote(item.id):
                    promoted += 1
        return promoted

    async def apply_decay(self) -> int:
        """应用衰减并清理低强度记忆

        strength < min_strength 的记忆直接删除，
        避免长期积累无效垃圾记忆。
        """
        min_strength = 0.05
        items = await self.windowed.get_recent(limit=1000)
        decayed = 0
        for item in items:
            age_hours = (time.time() - item.created_at) / 3600.0
            strength = item.strength * (0.9 ** age_hours)
            if strength < min_strength:
                if await self.windowed.forget(item.id):
                    decayed += 1
        if decayed > 0:
            logger.info("衰减清理: 删除了 %d 条低强度记忆", decayed)
        return decayed

    async def cleanup_expired(self) -> int:
        """清理所有过期/低强度记忆（短期 + 长期）"""
        short_term_cleaned = await self.apply_decay()

        # 清理向量存储中的低强度记忆
        long_term_cleaned = 0
        try:
            for collection in ("semantic", "episodic", "procedural"):
                items = await self.vector_rag.search(
                    "*", collections=[collection], limit=100
                )
                for item in items:
                    age_hours = (time.time() - item.created_at) / 3600.0
                    strength = item.strength * (0.9 ** age_hours)
                    if strength < 0.05:
                        await self.vector_rag.forget(collection, item.id)
                        long_term_cleaned += 1
        except Exception as e:
            logger.debug("长期记忆清理失败: %s", e)

        total = short_term_cleaned + long_term_cleaned
        if total > 0:
            logger.info("记忆清理完成: 短期%d条 + 长期%d条", short_term_cleaned, long_term_cleaned)
        return total

    async def summarize_conversation(
        self,
        messages: list[dict[str, str]],
        llm_client: Any | None = None,
    ) -> str | None:
        """对对话历史生成摘要并存入语义记忆"""
        if not messages or llm_client is None:
            return None

        conversation_text = "\n".join(
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages
            if msg.get("role") != "system"
        )

        if not conversation_text.strip():
            return None

        try:
            from long.llm.base import LLMMessage

            prompt_messages = [
                LLMMessage(
                    role="system",
                    content=(
                        "你是一个对话摘要助手。请对以下对话内容生成简洁的摘要，"
                        "重点提取：1)用户的核心需求和意图 2)助手的解决方案和关键结论 "
                        "3)重要的事实信息和决策。摘要应简洁但信息完整。"
                    ),
                ),
                LLMMessage(role="user", content=f"请摘要以下对话：\n\n{conversation_text}"),
            ]

            response = await llm_client.chat(prompt_messages, purpose="summarize")
            summary: str = response.content.strip()

            if summary:
                await self.vector_rag.store(
                    "semantic",
                    f"[对话摘要] {summary}",
                    importance=0.8,
                    tags=["summary", "conversation"],
                )
                logger.info("对话摘要已存入语义记忆: %s...", summary[:80])
                return summary

        except Exception as e:
            logger.warning("生成对话摘要失败: %s", e)

        return None


class _SemanticCompat:
    """语义记忆兼容层"""

    def __init__(self, vector_rag: VectorRAG) -> None:
        self._rag = vector_rag

    async def store(self, content: str, importance: float = 0.5, **kwargs: object) -> str:
        return await self._rag.store("semantic", content, importance=importance, **kwargs)

    async def search(self, query: str, limit: int = 10) -> list[MemoryItem]:
        return await self._rag.search(query, collections=["semantic"], limit=limit)

    async def recall(self, item_id: str) -> MemoryItem | None:
        return await self._rag.recall("semantic", item_id)

    async def forget(self, item_id: str) -> bool:
        return await self._rag.forget("semantic", item_id)

    async def count(self) -> int:
        return await self._rag.count("semantic")


class _EpisodicCompat:
    """情景记忆兼容层"""

    def __init__(self, vector_rag: VectorRAG) -> None:
        self._rag = vector_rag

    async def store(self, content: str, importance: float = 0.5, **kwargs: object) -> str:
        return await self._rag.store("episodic", content, importance=importance, **kwargs)

    async def search(self, query: str, limit: int = 10) -> list[MemoryItem]:
        return await self._rag.search(query, collections=["episodic"], limit=limit)

    async def recall(self, item_id: str) -> MemoryItem | None:
        return await self._rag.recall("episodic", item_id)

    async def forget(self, item_id: str) -> bool:
        return await self._rag.forget("episodic", item_id)

    async def count(self) -> int:
        return await self._rag.count("episodic")


class _ProceduralCompat:
    """过程记忆兼容层"""

    def __init__(self, vector_rag: VectorRAG) -> None:
        self._rag = vector_rag

    async def store(self, content: str, confidence: float = 0.5, **kwargs: object) -> str:
        return await self._rag.store("procedural", content, importance=confidence, **kwargs)

    async def search(self, query: str, limit: int = 10) -> list[MemoryItem]:
        return await self._rag.search(query, collections=["procedural"], limit=limit)

    async def recall(self, item_id: str) -> MemoryItem | None:
        return await self._rag.recall("procedural", item_id)

    async def forget(self, item_id: str) -> bool:
        return await self._rag.forget("procedural", item_id)

    async def count(self) -> int:
        return await self._rag.count("procedural")
