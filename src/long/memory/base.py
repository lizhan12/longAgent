"""记忆抽象与基础模型

定义记忆系统的核心抽象和数据模型。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """记忆类型"""

    SHORT_TERM = "short_term"
    WORKING = "working"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class MemoryItem(BaseModel):
    """记忆项

    表示一个独立的记忆条目。

    Attributes:
        id: 唯一标识符
        content: 记忆内容
        metadata: 元数据
        importance: 重要性 [0, 1]
        created_at: 创建时间戳
        strength: 记忆强度 [0, 1]
        memory_type: 记忆类型
        tags: 标签
    """

    id: str = Field(default_factory=lambda: uuid4().hex[:12], description="唯一标识符")
    content: str = Field(..., min_length=1, description="记忆内容")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="重要性")
    created_at: float = Field(default=0.0, description="创建时间戳（秒）")
    strength: float = Field(default=1.0, ge=0.0, le=1.0, description="记忆强度")
    memory_type: MemoryType = Field(default=MemoryType.SHORT_TERM, description="记忆类型")
    tags: list[str] = Field(default_factory=list, description="标签")
    retrieval_count: int = Field(default=0, ge=0, description="被检索命中次数")

    model_config = {"extra": "forbid"}


class MemoryQuery(BaseModel):
    """记忆查询"""

    query: str = Field(default="*", description="查询内容")
    memory_type: MemoryType | None = Field(default=None, description="记忆类型过滤")
    limit: int = Field(default=10, ge=1, le=1000, description="返回数量限制")
    strategy: str = Field(default="relevance", description="检索策略")
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0, description="最低重要性")
    min_strength: float = Field(default=0.0, ge=0.0, le=1.0, description="最低强度")
    tags: list[str] | None = Field(default=None, description="标签过滤")

    model_config = {"extra": "forbid"}


class MemoryStore(ABC):
    """记忆存储抽象基类"""

    @abstractmethod
    async def store(self, item: MemoryItem) -> str:
        """存储记忆项

        Returns:
            记忆项 ID
        """
        pass

    @abstractmethod
    async def recall(self, item_id: str) -> MemoryItem | None:
        """根据 ID 回忆记忆项"""
        pass

    @abstractmethod
    async def forget(self, item_id: str) -> bool:
        """删除记忆项

        Returns:
            是否成功删除
        """
        pass

    @abstractmethod
    async def search(self, query: MemoryQuery) -> list[MemoryItem]:
        """搜索记忆项"""
        pass

    @abstractmethod
    async def count(self) -> int:
        """获取记忆项数量"""
        pass
