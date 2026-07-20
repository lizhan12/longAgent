"""流式输出管理

支持 LLM token 级别的流式输出。
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class StreamChunk:
    """流式输出块"""

    content: str
    token_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


class StreamBuffer:
    """流式输出缓冲区

    线程安全的流式输出缓冲区，支持订阅者模式。
    """

    def __init__(self, max_size: int = 10000) -> None:
        self._buffer: deque[StreamChunk] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[StreamChunk], None]] = []
        self._active = False
        self._total_tokens = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def start(self) -> None:
        """开始流式输出"""
        with self._lock:
            self._active = True
            self._buffer.clear()
            self._total_tokens = 0

    def write(self, chunk: StreamChunk) -> None:
        """写入流式输出块"""
        with self._lock:
            if not self._active:
                return
            self._buffer.append(chunk)
            self._total_tokens += chunk.token_count

        for subscriber in self._subscribers:
            try:
                subscriber(chunk)
            except Exception:
                pass

    def end(self) -> None:
        """结束流式输出"""
        with self._lock:
            self._active = False

    def subscribe(self, callback: Callable[[StreamChunk], None]) -> None:
        """订阅流式输出"""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[StreamChunk], None]) -> None:
        """取消订阅"""
        self._subscribers = [s for s in self._subscribers if s != callback]

    def get_content(self) -> str:
        """获取所有已缓冲的内容"""
        with self._lock:
            return "".join(chunk.content for chunk in self._buffer)

    def clear(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self._buffer.clear()
            self._total_tokens = 0


class StreamingManager:
    """流式输出管理器

    管理多个流式输出通道。
    """

    def __init__(self) -> None:
        self._streams: dict[str, StreamBuffer] = {}

    def create_stream(self, stream_id: str) -> StreamBuffer:
        """创建流式输出通道"""
        if stream_id in self._streams:
            return self._streams[stream_id]

        buffer = StreamBuffer()
        self._streams[stream_id] = buffer
        return buffer

    def get_stream(self, stream_id: str) -> StreamBuffer | None:
        """获取流式输出通道"""
        return self._streams.get(stream_id)

    def remove_stream(self, stream_id: str) -> None:
        """移除流式输出通道"""
        if stream_id in self._streams:
            self._streams[stream_id].end()
            del self._streams[stream_id]

    def end_all(self) -> None:
        """结束所有流式输出"""
        for buffer in self._streams.values():
            buffer.end()
        self._streams.clear()
