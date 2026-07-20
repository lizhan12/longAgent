"""InteractionController - 交互控制器门面类

管理会话、事件总线和交互协议适配器。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .base import HITLRequest, HITLResponse, InteractionEvent, InteractionEventType, InteractionProtocol
from .session import Session, SessionState
from .streaming import StreamBuffer, StreamingManager

logger = logging.getLogger(__name__)


class EventSubscriber:
    """事件订阅者"""

    def __init__(self, callback: Callable[[InteractionEvent], None], event_types: set[InteractionEventType] | None = None) -> None:
        self.callback = callback
        self.event_types = event_types

    def matches(self, event: InteractionEvent) -> bool:
        if self.event_types is None:
            return True
        return event.type in self.event_types


class InteractionController:
    """交互控制器门面类

    统一管理会话、事件分发和交互协议。

    Attributes:
        protocol: 当前交互协议适配器
        streaming: 流式输出管理器
    """

    def __init__(self, protocol: InteractionProtocol | None = None) -> None:
        self._protocol = protocol
        self._sessions: dict[str, Session] = {}
        self._active_session_id: str | None = None
        self._subscribers: list[EventSubscriber] = []
        self.streaming = StreamingManager()

    @property
    def protocol(self) -> InteractionProtocol | None:
        return self._protocol

    @protocol.setter
    def protocol(self, value: InteractionProtocol) -> None:
        self._protocol = value

    # --- 会话管理 ---

    def create_session(self, metadata: dict[str, Any] | None = None) -> Session:
        """创建新会话"""
        session = Session(metadata=metadata or {})
        self._sessions[session.session_id] = session
        return session

    def activate_session(self, session_id: str) -> Session | None:
        """激活会话"""
        session = self._sessions.get(session_id)
        if session is None:
            return None

        if self._active_session_id and self._active_session_id in self._sessions:
            current = self._sessions[self._active_session_id]
            self._sessions[self._active_session_id] = current.pause()

        activated = session.activate()
        self._sessions[session_id] = activated
        self._active_session_id = session_id

        if self._protocol:
            self._protocol.start_session()

        return activated

    def end_session(self, session_id: str) -> Session | None:
        """结束会话"""
        session = self._sessions.get(session_id)
        if session is None:
            return None

        ended = session.end()
        self._sessions[session_id] = ended

        if self._active_session_id == session_id:
            if self._protocol:
                self._protocol.end_session()
            self._active_session_id = None

        return ended

    def get_session(self, session_id: str) -> Session | None:
        """获取会话"""
        return self._sessions.get(session_id)

    @property
    def active_session(self) -> Session | None:
        """获取当前活跃会话"""
        if self._active_session_id is None:
            return None
        return self._sessions.get(self._active_session_id)

    # --- 事件总线 ---

    def publish(self, event: InteractionEvent) -> None:
        """发布事件"""
        for subscriber in self._subscribers:
            if subscriber.matches(event):
                try:
                    subscriber.callback(event)
                except Exception as e:
                    logger.error(f"事件订阅者处理异常: {e}")

        if self._protocol:
            self._protocol.send_event(event)

    def subscribe(
        self,
        callback: Callable[[InteractionEvent], None],
        event_types: set[InteractionEventType] | None = None,
    ) -> EventSubscriber:
        """订阅事件"""
        subscriber = EventSubscriber(callback, event_types)
        self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: EventSubscriber) -> None:
        """取消订阅"""
        self._subscribers = [s for s in self._subscribers if s != subscriber]

    # --- 交互接口 ---

    def send_message(self, content: str, **metadata: Any) -> None:
        """发送消息事件"""
        self.publish(InteractionEvent(
            type=InteractionEventType.MESSAGE,
            content=content,
            metadata=metadata,
        ))

    def send_error(self, content: str, **metadata: Any) -> None:
        """发送错误事件"""
        self.publish(InteractionEvent(
            type=InteractionEventType.ERROR,
            content=content,
            metadata=metadata,
        ))

    def send_info(self, content: str, **metadata: Any) -> None:
        """发送信息事件"""
        self.publish(InteractionEvent(
            type=InteractionEventType.INFO,
            content=content,
            metadata=metadata,
        ))

    def send_warning(self, content: str, **metadata: Any) -> None:
        """发送警告事件"""
        self.publish(InteractionEvent(
            type=InteractionEventType.WARNING,
            content=content,
            metadata=metadata,
        ))

    def get_input(self, prompt: str = "") -> str:
        """获取用户输入"""
        if self._protocol:
            return self._protocol.receive_input(prompt)
        return ""

    def request_feedback(self, hitl_request: HITLRequest) -> HITLResponse:
        """请求人工反馈"""
        if self._protocol:
            return self._protocol.request_feedback(hitl_request)
        return HITLResponse(
            request_id=hitl_request.request_id,
            decision="approve",
        )

    def start_stream(self, stream_id: str | None = None) -> StreamBuffer:
        """开始流式输出"""
        sid = stream_id or self._active_session_id or "default"
        buffer = self.streaming.create_stream(sid)
        buffer.start()
        self.publish(InteractionEvent(
            type=InteractionEventType.STREAM_START,
            content=sid,
        ))
        return buffer

    def end_stream(self, stream_id: str | None = None) -> None:
        """结束流式输出"""
        sid = stream_id or self._active_session_id or "default"
        buffer = self.streaming.get_stream(sid)
        if buffer:
            buffer.end()
            self.publish(InteractionEvent(
                type=InteractionEventType.STREAM_END,
                content=sid,
                metadata={"total_tokens": buffer.total_tokens},
            ))
