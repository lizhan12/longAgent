"""结构化日志

基于 Python 标准 logging 模块，提供 JSON 格式结构化输出。
自动注入 trace_id/span_id，支持敏感信息脱敏和日志轮转。

用法:
    from long.observability.structured_logging import get_logger

    logger = get_logger("long.llm")
    logger.info("llm_call", model="gpt-4o", tokens=150, latency_ms=2300)
    logger.error("llm_failed", model="gpt-4o", error_type="TimeoutError")
"""

from __future__ import annotations

import json
import logging
import os
import re
from logging.handlers import RotatingFileHandler
from typing import Any

_SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "api-key",
    "secret", "password", "passwd", "token",
    "authorization", "cookie", "credential",
    "openai_api_key", "anthropic_api_key",
})

_REDACTED = "***REDACTED***"


def _redact_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 8:
        if any(kw in value.lower() for kw in ("sk-", "key-", "token-")):
            return value[:4] + "..." + value[-4:]
    return value


def _redact_dict(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in _SENSITIVE_KEYS:
            result[key] = _REDACTED
        elif isinstance(value, dict):
            result[key] = _redact_dict(value)
        elif isinstance(value, str):
            result[key] = _redact_value(value)
        else:
            result[key] = value
    return result


class StructuredFormatter(logging.Formatter):
    def __init__(self, redact: bool = True) -> None:
        super().__init__()
        self.redact = redact

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
        }

        trace_id = getattr(record, "trace_id", "")
        span_id = getattr(record, "span_id", "")
        if trace_id:
            log_entry["trace_id"] = trace_id
        if span_id:
            log_entry["span_id"] = span_id

        event = getattr(record, "structured_event", None)
        if event:
            log_entry["event"] = event
        else:
            log_entry["event"] = record.getMessage()

        extra_data = getattr(record, "structured_data", None)
        if extra_data:
            if self.redact:
                extra_data = _redact_dict(extra_data)
            log_entry.update(extra_data)

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "Unknown",
                "message": str(record.exc_info[1]),
            }

        try:
            return json.dumps(log_entry, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(log_entry)


class StructuredLogger:
    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def _log(
        self,
        level: int,
        event: str,
        **kwargs: Any,
    ) -> None:
        from long.observability.tracing import current_trace_id, current_span_id

        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=event,
            args=(),
            exc_info=None,
        )
        record.structured_event = event
        record.structured_data = kwargs
        record.trace_id = current_trace_id()
        record.span_id = current_span_id()
        self._logger.handle(record)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, event, **kwargs)


_loggers: dict[str, StructuredLogger] = {}


def get_logger(name: str) -> StructuredLogger:
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]


def setup_structured_logging(
    log_file: str | None = None,
    level: int = logging.INFO,
    max_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 5,
    redact: bool = True,
) -> None:
    """配置结构化日志

    Args:
        log_file: 日志文件路径，None 则仅输出到控制台
        level: 日志级别
        max_bytes: 单个日志文件最大字节数（默认 50MB）
        backup_count: 保留的日志文件数量（默认 5）
        redact: 是否脱敏敏感信息
    """
    formatter = StructuredFormatter(redact=redact)

    handlers: list[logging.Handler] = []

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    console_handler.setLevel(logging.WARNING)
    handlers.append(console_handler)

    root_logger = logging.getLogger("long")
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    for handler in handlers:
        root_logger.addHandler(handler)
