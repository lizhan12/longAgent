"""交互协议抽象

定义人与系统之间的交互协议接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class InteractionEventType(str, Enum):
    """交互事件类型"""

    MESSAGE = "message"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    PROGRESS = "progress"
    STREAM_START = "stream_start"
    STREAM_TOKEN = "stream_token"
    STREAM_END = "stream_end"
    TURN_COMPLETE = "turn_complete"
    TRACE = "trace"
    HITL_REQUEST = "hitl_request"
    HITL_RESPONSE = "hitl_response"
    SYSTEM = "system"
    COMMAND = "command"


class InteractionEvent(BaseModel):
    """交互事件"""

    type: InteractionEventType = Field(..., description="事件类型")
    content: str = Field(default="", description="事件内容")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")

    model_config = {"extra": "forbid"}


@dataclass
class HITLRequest:
    """Human-in-the-loop 审核请求"""

    request_id: str
    title: str
    description: str
    risk_level: str = "medium"
    options: list[str] = field(default_factory=lambda: ["approve", "reject"])
    context: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 300.0


@dataclass
class HITLResponse:
    """Human-in-the-loop 审核响应"""

    request_id: str
    decision: str
    feedback: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


class InteractionProtocol(ABC):
    """交互协议抽象接口

    定义人与系统之间的交互方式，支持 CLI、WebUI 等不同适配器。
    """

    @abstractmethod
    def send_event(self, event: InteractionEvent) -> None:
        """向用户输出事件"""
        pass

    @abstractmethod
    def receive_input(self, prompt: str = "") -> str:
        """接收用户输入"""
        pass

    @abstractmethod
    def stream_output(self, tokens: Any) -> None:
        """流式输出"""
        pass

    @abstractmethod
    def request_feedback(self, hitl_request: HITLRequest) -> HITLResponse:
        """HITL 人工审核请求"""
        pass

    @abstractmethod
    def start_session(self) -> None:
        """启动会话"""
        pass

    @abstractmethod
    def end_session(self) -> None:
        """结束会话"""
        pass
