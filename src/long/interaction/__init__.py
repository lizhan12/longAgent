"""Interaction 模块 - 交互协议与控制器"""

from .base import HITLRequest, HITLResponse, InteractionEvent, InteractionEventType, InteractionProtocol
from .controller import InteractionController
from .session import Session, SessionState
from .streaming import StreamBuffer, StreamingManager
from .adapters.webui import WebUIAdapter, WebSocketConnection

__all__ = [
    "HITLRequest",
    "HITLResponse",
    "InteractionController",
    "InteractionEvent",
    "InteractionEventType",
    "InteractionProtocol",
    "Session",
    "SessionState",
    "StreamBuffer",
    "StreamingManager",
    "WebUIAdapter",
    "WebSocketConnection",
]
