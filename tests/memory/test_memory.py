"""Memory 模块测试

覆盖记忆项、MemoryStore CRUD 和控制器。
"""

import pytest

from long.memory.base import MemoryItem, MemoryQuery, MemoryType
from long.memory.backends.in_memory import InMemoryBackend


# ========================
# MemoryItem 测试
# ========================


class TestMemoryItem:
    """记忆项测试"""

    def test_create_memory_item(self):
        item = MemoryItem(content="test content")
        assert item.content == "test content"
        assert item.memory_type == MemoryType.SHORT_TERM
        assert 0.0 <= item.importance <= 1.0
        assert 0.0 <= item.strength <= 1.0

    def test_create_with_all_fields(self):
        item = MemoryItem(
            content="test",
            importance=0.9,
            strength=0.8,
            memory_type=MemoryType.SEMANTIC,
            tags=["important", "reference"],
            metadata={"source": "test"},
        )
        assert item.importance == 0.9
        assert item.memory_type == MemoryType.SEMANTIC
        assert len(item.tags) == 2

    def test_memory_types(self):
        expected = {"short_term", "working", "semantic", "episodic", "procedural"}
        actual = {t.value for t in MemoryType}
        assert actual == expected


# ========================
# InMemoryBackend 测试
# ========================


class TestInMemoryBackend:
    """内存后端测试"""

    @pytest.fixture
    def backend(self):
        return InMemoryBackend(max_size=10)

    @pytest.mark.asyncio
    async def test_store_and_recall(self, backend):
        item = MemoryItem(content="hello")
        item_id = await backend.store(item)
        recalled = await backend.recall(item_id)
        assert recalled is not None
        assert recalled.content == "hello"

    @pytest.mark.asyncio
    async def test_forget(self, backend):
        item = MemoryItem(content="to_delete")
        item_id = await backend.store(item)
        result = await backend.forget(item_id)
        assert result is True
        assert await backend.recall(item_id) is None

    @pytest.mark.asyncio
    async def test_forget_nonexistent(self, backend):
        result = await backend.forget("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_search(self, backend):
        await backend.store(MemoryItem(content="python programming"))
        await backend.store(MemoryItem(content="rust programming"))
        await backend.store(MemoryItem(content="cooking recipes"))

        results = await backend.search(MemoryQuery(query="programming", limit=5))
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_by_type(self, backend):
        await backend.store(MemoryItem(content="short", memory_type=MemoryType.SHORT_TERM))
        await backend.store(MemoryItem(content="semantic", memory_type=MemoryType.SEMANTIC))

        results = await backend.search(
            MemoryQuery(query="*", memory_type=MemoryType.SEMANTIC, limit=10)
        )
        assert len(results) == 1
        assert results[0].memory_type == MemoryType.SEMANTIC

    @pytest.mark.asyncio
    async def test_count(self, backend):
        assert await backend.count() == 0
        await backend.store(MemoryItem(content="item1"))
        await backend.store(MemoryItem(content="item2"))
        assert await backend.count() == 2

    @pytest.mark.asyncio
    async def test_max_size_eviction(self, backend):
        for i in range(15):
            await backend.store(MemoryItem(content=f"item_{i}"))

        count = await backend.count()
        assert count <= 10  # max_size=10


# ========================
# MemoryController 测试
# ========================


class TestMemoryController:
    """记忆控制器测试"""

    @pytest.fixture
    def controller(self, tmp_path):
        from long.memory.controller import MemoryController
        return MemoryController(data_dir=str(tmp_path / "memory_data"))

    @pytest.mark.asyncio
    async def test_store_short_term(self, controller):
        item_id = await controller.store("test content", memory_type=MemoryType.SHORT_TERM)
        assert item_id is not None

    @pytest.mark.asyncio
    async def test_store_semantic(self, controller):
        item_id = await controller.store("semantic content", memory_type=MemoryType.SEMANTIC)
        assert item_id is not None

    @pytest.mark.asyncio
    async def test_cross_type_search(self, controller):
        await controller.store("python short term", memory_type=MemoryType.SHORT_TERM)
        await controller.store("python semantic", memory_type=MemoryType.SEMANTIC)

        results = await controller.search("python")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_single_type_search(self, controller):
        await controller.store("short term item", memory_type=MemoryType.SHORT_TERM)

        results = await controller.search(
            "short term", memory_type=MemoryType.SHORT_TERM
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_promote(self, controller):
        item_id = await controller.store(
            "important item", importance=0.8, memory_type=MemoryType.SHORT_TERM
        )
        result = await controller.promote(item_id)
        assert result is True

        # 短期记忆中已删除
        item = await controller.short_term.recall(item_id)
        assert item is None

    @pytest.mark.asyncio
    async def test_promote_not_important(self, controller):
        item_id = await controller.store(
            "not important", importance=0.3, memory_type=MemoryType.SHORT_TERM
        )
        result = await controller.promote(item_id)
        assert result is False
