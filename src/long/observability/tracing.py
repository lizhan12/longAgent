"""全链路追踪体系

基于 Trace/Span 模型实现分布式追踪，支持上下文传播和嵌套调用。
每次用户请求生成一个 Trace，每次 LLM/工具/记忆操作生成一个 Span。

用法:
    # 在主循环入口创建 Trace
    async with tracer.trace("user_message", attributes={"user_input": msg}) as trace:
        # LLM 调用自动创建子 Span
        async with trace.span("llm.chat", attributes={"model": "gpt-4o"}) as span:
            response = await llm.chat(messages)
            span.set_attribute("tokens", response.usage.total_tokens)

    # 获取当前 trace_id 用于日志关联
    trace_id = current_trace_id()
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class SpanEvent:
    name: str
    timestamp: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    name: str
    span_id: str
    trace_id: str
    parent_span_id: str | None = None
    start_time: float = 0.0
    end_time: float | None = None
    status: SpanStatus = SpanStatus.OK
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append(SpanEvent(
            name=name,
            timestamp=time.time(),
            attributes=attributes or {},
        ))

    def finish(self, status: SpanStatus = SpanStatus.OK) -> None:
        self.end_time = time.time()
        self.status = status

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": [
                {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
                for e in self.events
            ],
        }


@dataclass
class Trace:
    trace_id: str
    name: str
    start_time: float = 0.0
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    spans: list[Span] = field(default_factory=list)
    _current_span: Span | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.start_time = self.start_time or time.time()

    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> SpanContext:
        parent_id = self._current_span.span_id if self._current_span else None
        span = Span(
            name=name,
            span_id=_gen_id(),
            trace_id=self.trace_id,
            parent_span_id=parent_id,
            start_time=time.time(),
            attributes=attributes or {},
        )
        self.spans.append(span)
        return SpanContext(trace=self, span=span)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def finish(self) -> None:
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        # 聚合 spans 的状态：如果有任何 error span 则 trace 状态为 error
        status = "ok"
        if self.spans and any(s.status != SpanStatus.OK for s in self.spans):
            status = "error"
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": status,
            "attributes": self.attributes,
            "spans": [s.to_dict() for s in self.spans],
        }

    def root_span(self) -> Span | None:
        return next((s for s in self.spans if s.parent_span_id is None), None)

    def child_spans(self, parent_span_id: str) -> list[Span]:
        return [s for s in self.spans if s.parent_span_id == parent_span_id]

    def failed_spans(self) -> list[Span]:
        return [s for s in self.spans if s.status != SpanStatus.OK]


class SpanContext:
    def __init__(self, trace: Trace, span: Span) -> None:
        self._trace = trace
        self._span = span

    async def __aenter__(self) -> Span:
        self._trace._current_span = self._span
        _current_span_var.set(self._span)
        return self._span

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self._span.finish(SpanStatus.ERROR)
            self._span.set_attribute("error.type", exc_type.__name__)
            self._span.set_attribute("error.message", str(exc_val))
        else:
            self._span.finish()
        _current_span_var.set(None)
        parent = next(
            (s for s in self._trace.spans if s.span_id == self._span.parent_span_id),
            None,
        )
        self._trace._current_span = parent

    def __enter__(self) -> Span:
        self._trace._current_span = self._span
        _current_span_var.set(self._span)
        return self._span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self._span.finish(SpanStatus.ERROR)
            self._span.set_attribute("error.type", exc_type.__name__)
            self._span.set_attribute("error.message", str(exc_val))
        else:
            self._span.finish()
        _current_span_var.set(None)
        parent = next(
            (s for s in self._trace.spans if s.span_id == self._span.parent_span_id),
            None,
        )
        self._trace._current_span = parent


def _gen_id() -> str:
    return uuid.uuid4().hex[:16]


_current_trace_var: ContextVar[Trace | None] = ContextVar("current_trace", default=None)
_current_span_var: ContextVar[Span | None] = ContextVar("current_span", default=None)


def current_trace_id() -> str:
    trace = _current_trace_var.get()
    return trace.trace_id if trace else ""


def current_span_id() -> str:
    span = _current_span_var.get()
    return span.span_id if span else ""


def current_trace() -> Trace | None:
    return _current_trace_var.get()


class Tracer:
    def __init__(self) -> None:
        self._traces: list[Trace] = []
        self._max_traces: int = 1000

    def trace(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> TraceContext:
        t = Trace(
            trace_id=_gen_id(),
            name=name,
            attributes=attributes or {},
        )
        self._traces.append(t)
        if len(self._traces) > self._max_traces:
            self._traces = self._traces[-self._max_traces:]
        return TraceContext(tracer=self, trace=t)

    def get_traces(self, limit: int = 100) -> list[Trace]:
        return self._traces[-limit:]

    def get_trace(self, trace_id: str) -> Trace | None:
        return next((t for t in self._traces if t.trace_id == trace_id), None)

    def clear(self) -> None:
        self._traces.clear()


class TraceContext:
    def __init__(self, tracer: Tracer, trace: Trace) -> None:
        self._tracer = tracer
        self._trace = trace

    async def __aenter__(self) -> Trace:
        _current_trace_var.set(self._trace)
        return self._trace

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._trace.finish()
        _current_trace_var.set(None)
        _current_span_var.set(None)

    def __enter__(self) -> Trace:
        _current_trace_var.set(self._trace)
        return self._trace

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._trace.finish()
        _current_trace_var.set(None)
        _current_span_var.set(None)
