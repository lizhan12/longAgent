"""指标收集

从评估结果、执行反馈和 Judge 评分中收集指标。
支持 SQLite 持久化存储，重启后数据不丢失。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MetricPoint(BaseModel):
    """指标数据点"""

    timestamp: float = Field(default_factory=time.time)
    name: str = ""
    value: float = 0.0
    tags: dict[str, str] = Field(default_factory=dict)


class MetricsCollector:
    """指标收集器

    收集并聚合来自多个来源的指标数据。
    支持 SQLite 持久化存储。

    Attributes:
        _metrics: 内存指标存储
        _aggregations: 聚合缓存
        _db_path: SQLite 数据库路径
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._metrics: dict[str, list[MetricPoint]] = defaultdict(list)
        self._aggregations: dict[str, dict[str, float]] = {}
        self._db_path = Path(db_path) if db_path else None
        self._db: sqlite3.Connection | None = None

        if self._db_path:
            self._init_db()

    def _init_db(self) -> None:
        if self._db_path is None:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                tags TEXT DEFAULT '{}',
                trace_id TEXT DEFAULT '',
                span_id TEXT DEFAULT ''
            )
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_name
            ON metrics(name)
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
            ON metrics(timestamp)
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp
            ON metrics(name, timestamp)
        """)
        self._db.commit()

        self._load_recent_from_db()

    def _load_recent_from_db(self, hours: int = 24) -> None:
        if self._db is None:
            return

        cutoff = time.time() - hours * 3600
        try:
            cursor = self._db.execute(
                "SELECT timestamp, name, value, tags FROM metrics WHERE timestamp >= ?",
                (cutoff,),
            )
            count = 0
            for row in cursor:
                ts, name, value, tags_json = row
                try:
                    tags = json.loads(tags_json) if tags_json else {}
                except (json.JSONDecodeError, TypeError):
                    tags = {}
                point = MetricPoint(timestamp=ts, name=name, value=value, tags=tags)
                self._metrics[name].append(point)
                count += 1
            logger.info("从 SQLite 加载了 %d 条历史指标", count)
        except Exception as e:
            logger.warning("加载历史指标失败: %s", e)

    def _persist_point(self, point: MetricPoint, trace_id: str = "", span_id: str = "") -> None:
        if self._db is None:
            return

        try:
            self._db.execute(
                "INSERT INTO metrics (timestamp, name, value, tags, trace_id, span_id) VALUES (?, ?, ?, ?, ?, ?)",
                (point.timestamp, point.name, point.value, json.dumps(point.tags), trace_id, span_id),
            )
            self._db.commit()
        except Exception as e:
            logger.warning("持久化指标失败: %s", e)

    def record(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
        trace_id: str = "",
        span_id: str = "",
    ) -> None:
        """记录指标

        Args:
            name: 指标名称
            value: 指标值
            tags: 标签
            trace_id: 关联的 Trace ID
            span_id: 关联的 Span ID
        """
        point = MetricPoint(
            name=name,
            value=value,
            tags=tags or {},
        )
        self._metrics[name].append(point)
        self._aggregations.pop(name, None)
        self._persist_point(point, trace_id, span_id)

    def get_metrics(
        self,
        name: str,
        since: float | None = None,
    ) -> list[MetricPoint]:
        """获取指标数据

        Args:
            name: 指标名称
            since: 起始时间戳

        Returns:
            指标数据点列表
        """
        points = self._metrics.get(name, [])
        if since is not None:
            points = [p for p in points if p.timestamp >= since]
        return points

    def get_aggregation(
        self,
        name: str,
        since: float | None = None,
    ) -> dict[str, float]:
        """获取指标聚合

        Args:
            name: 指标名称
            since: 起始时间戳

        Returns:
            聚合结果 (count, mean, min, max, std)
        """
        cache_key = f"{name}:{since}"
        if cache_key in self._aggregations:
            return self._aggregations[cache_key]

        points = self.get_metrics(name, since)
        if not points:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

        values = [p.value for p in points]
        count = len(values)
        mean = sum(values) / count
        min_val = min(values)
        max_val = max(values)
        variance = sum((v - mean) ** 2 for v in values) / count
        std = variance ** 0.5

        result = {
            "count": float(count),
            "mean": mean,
            "min": min_val,
            "max": max_val,
            "std": std,
        }

        self._aggregations[cache_key] = result
        return result

    def get_all_metric_names(self) -> list[str]:
        """获取所有指标名称"""
        return list(self._metrics.keys())

    def record_eval_result(
        self,
        task_name: str,
        score: float,
        category: str = "normal",
    ) -> None:
        """记录评估结果"""
        self.record("eval.score", score, {"task": task_name, "category": category})

    def record_execution_metrics(
        self,
        step_count: int,
        duration: float,
        success: bool,
    ) -> None:
        """记录执行指标"""
        self.record("execution.steps", float(step_count))
        self.record("execution.duration", duration)
        self.record("execution.success", 1.0 if success else 0.0)

    def record_llm_call(
        self,
        model: str,
        purpose: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        success: bool,
        error_type: str = "",
    ) -> None:
        """记录 LLM 调用指标"""
        self.record("llm.latency_ms", latency_ms, {"model": model, "purpose": purpose})
        self.record("llm.prompt_tokens", float(prompt_tokens), {"model": model})
        self.record("llm.completion_tokens", float(completion_tokens), {"model": model})
        self.record("llm.success", 1.0 if success else 0.0, {"model": model, "purpose": purpose})
        if error_type:
            self.record("llm.error", 1.0, {"model": model, "error_type": error_type})

    def record_tool_call(
        self,
        tool_name: str,
        latency_ms: float,
        success: bool,
        error_type: str = "",
    ) -> None:
        """记录工具调用指标"""
        self.record("tool.latency_ms", latency_ms, {"tool_name": tool_name})
        self.record("tool.success", 1.0 if success else 0.0, {"tool_name": tool_name})
        if error_type:
            self.record("tool.error", 1.0, {"tool_name": tool_name, "error_type": error_type})

    def clear(self) -> None:
        """清除所有指标"""
        self._metrics.clear()
        self._aggregations.clear()

    def close(self) -> None:
        """关闭数据库连接"""
        if self._db is not None:
            self._db.close()
            self._db = None

    def cleanup_old_metrics(self, days: int = 30) -> int:
        """清理旧指标数据

        Args:
            days: 保留天数

        Returns:
            删除的记录数
        """
        if self._db is None:
            return 0

        cutoff = time.time() - days * 86400
        try:
            cursor = self._db.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff,))
            self._db.commit()
            deleted = cursor.rowcount
            if deleted > 0:
                logger.info("清理了 %d 条超过 %d 天的旧指标", deleted, days)
            return deleted
        except Exception as e:
            logger.warning("清理旧指标失败: %s", e)
            return 0
