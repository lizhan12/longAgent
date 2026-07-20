"""会话模型

管理交互会话的生命周期。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class SessionState(str, Enum):
    """会话状态"""

    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


class Session(BaseModel):
    """会话模型"""

    session_id: str = Field(default_factory=lambda: uuid4().hex[:12], description="会话 ID")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    state: SessionState = Field(default=SessionState.CREATED, description="会话状态")
    metadata: dict[str, Any] = Field(default_factory=dict, description="会话元数据")

    model_config = {"extra": "forbid"}

    def activate(self) -> Session:
        """激活会话"""
        return self.model_copy(update={"state": SessionState.ACTIVE})

    def pause(self) -> Session:
        """暂停会话"""
        return self.model_copy(update={"state": SessionState.PAUSED})

    def end(self) -> Session:
        """结束会话"""
        return self.model_copy(update={"state": SessionState.ENDED})

    @property
    def is_active(self) -> bool:
        return self.state == SessionState.ACTIVE
