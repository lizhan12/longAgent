from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import Session

logger = logging.getLogger(__name__)


class SessionStore:
    """会话存储，按天归档到 sessions/YYYY-MM-DD/<session_id>.json"""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir / "sessions"
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, date_str: str) -> Path:
        d = self._data_dir / date_str
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, session: Session) -> None:
        d = self._session_dir(session.date_str)
        path = d / f"{session.id}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(session.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("保存会话 %s 失败: %s", session.id, e)

    def load(self, session_id: str, date_str: str) -> Session | None:
        path = self._session_dir(date_str) / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return Session.model_validate(data)
        except Exception as e:
            logger.warning("加载会话 %s 失败: %s", session_id, e)
            return None

    def load_latest_session(self) -> Session | None:
        """加载最近一天最近修改的会话"""
        if not self._data_dir.exists():
            return None

        date_dirs = sorted(
            [d for d in self._data_dir.iterdir() if d.is_dir()],
            reverse=True,
        )
        if not date_dirs:
            return None

        for date_dir in date_dirs[:3]:
            json_files = sorted(
                [f for f in date_dir.glob("*.json")],
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            for jf in json_files:
                try:
                    with open(jf, encoding="utf-8") as f:
                        data = json.load(f)
                    return Session.model_validate(data)
                except Exception:
                    continue

        return None

    def list_sessions_by_date(self, date_str: str) -> list[str]:
        d = self._data_dir / date_str
        if not d.exists():
            return []
        return sorted(f.stem for f in d.glob("*.json"))

    def list_dates(self) -> list[str]:
        if not self._data_dir.exists():
            return []
        return sorted(
            d.name for d in self._data_dir.iterdir() if d.is_dir()
        )

    def load_sessions_by_date(self, date_str: str) -> list[Session]:
        d = self._data_dir / date_str
        if not d.exists():
            return []
        sessions: list[Session] = []
        for jf in sorted(d.glob("*.json")):
            try:
                with open(jf, encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append(Session.model_validate(data))
            except Exception:
                continue
        return sessions

    def delete(self, session_id: str, date_str: str) -> bool:
        path = self._session_dir(date_str) / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False
