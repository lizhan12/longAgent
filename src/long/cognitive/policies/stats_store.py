from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Any
from datetime import datetime

class StatsStore:
    def __init__(self, db_path: str | Path = "data/policy_stats.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS execution_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                task_id TEXT NOT NULL,
                query TEXT NOT NULL,
                execution_path TEXT NOT NULL,
                search_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                latency_seconds REAL DEFAULT 0.0,
                completed INTEGER DEFAULT 0,
                degraded INTEGER DEFAULT 0,
                retry_count INTEGER DEFAULT 0,
                error_types TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp ON execution_outcomes(timestamp);
            CREATE INDEX IF NOT EXISTS idx_outcomes_path ON execution_outcomes(execution_path);
        """)
        self._conn.commit()

    def record_outcome(self, outcome: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT INTO execution_outcomes
               (timestamp, task_id, query, execution_path, search_count, tool_call_count,
                total_tokens, latency_seconds, completed, degraded, retry_count, error_types, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                outcome.get("task_id", ""),
                outcome.get("query", ""),
                outcome.get("execution_path", "unknown"),
                outcome.get("search_count", 0),
                outcome.get("tool_call_count", 0),
                outcome.get("total_tokens", 0),
                outcome.get("latency_seconds", 0.0),
                1 if outcome.get("completed", False) else 0,
                1 if outcome.get("degraded", False) else 0,
                outcome.get("retry_count", 0),
                json.dumps(outcome.get("error_types", [])),
                json.dumps(outcome.get("metadata", {})),
            ),
        )
        self._conn.commit()

    def get_recent_outcomes(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM execution_outcomes ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_success_rate(self, execution_path: str | None = None) -> float:
        if execution_path:
            row = self._conn.execute(
                "SELECT AVG(completed) as rate FROM execution_outcomes WHERE execution_path = ?",
                (execution_path,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT AVG(completed) as rate FROM execution_outcomes").fetchone()
        return row["rate"] or 0.0

    def get_avg_search_count(self) -> float:
        row = self._conn.execute(
            "SELECT AVG(search_count) as avg FROM execution_outcomes WHERE completed = 1"
        ).fetchone()
        return row["avg"] or 0.0

    def close(self) -> None:
        self._conn.close()
