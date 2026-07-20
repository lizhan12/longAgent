from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_PREFERENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"我不喜欢(.{1,30})", re.IGNORECASE),
    re.compile(r"我习惯(.{1,30})", re.IGNORECASE),
    re.compile(r"以后(都|尽量|请)(.{1,30})", re.IGNORECASE),
    re.compile(r"请(不要|别)(.{1,30})", re.IGNORECASE),
    re.compile(r"我(更)?偏好(.{1,30})", re.IGNORECASE),
    re.compile(r"我(更)?倾向(.{1,30})", re.IGNORECASE),
    re.compile(r"(不要|别)用(.{1,30})", re.IGNORECASE),
    re.compile(r"默认(用|使用)(.{1,30})", re.IGNORECASE),
]


class PreferenceStore:
    """用户偏好存储

    从对话中提取偏好，持久化到 preferences.json。
    偏好始终注入 system prompt。
    支持延迟写入（debounce），减少 IO 次数。
    """

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "preferences.json"
        self._preferences: dict[str, dict[str, str]] = {}
        self._dirty: bool = False
        self._last_save_time: float = 0.0
        self._debounce_seconds: float = 5.0
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._preferences = data
        except Exception as e:
            logger.warning("加载偏好文件失败: %s，使用空偏好", e)
            self._preferences = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._preferences, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("保存偏好文件失败: %s", e)

    def get_all(self) -> dict[str, str]:
        return {k: v["value"] for k, v in self._preferences.items()}

    def get(self, key: str) -> str | None:
        entry = self._preferences.get(key)
        return entry["value"] if entry else None

    def update(self, key: str, value: str) -> None:
        self._preferences[key] = {
            "value": value,
            "updated_at": datetime.now().isoformat(),
        }
        self._dirty = True
        self._try_debounced_save()
        logger.info("偏好更新: %s = %s", key, value)

    def _try_debounced_save(self) -> None:
        """Debounced save: 只在距上次保存超过 debounce_seconds 后才写入"""
        import time
        now = time.time()
        if now - self._last_save_time >= self._debounce_seconds:
            self._save()
            self._dirty = False
            self._last_save_time = now

    def flush(self) -> None:
        """强制写入所有未保存的偏好"""
        if self._dirty:
            self._save()
            self._dirty = False
            import time
            self._last_save_time = time.time()

    def remove(self, key: str) -> bool:
        if key in self._preferences:
            del self._preferences[key]
            self._save()
            return True
        return False

    def format_for_prompt(self) -> str:
        prefs = self.get_all()
        if not prefs:
            return ""
        lines = ["## 用户偏好", ""]
        for k, v in prefs.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
        return "\n".join(lines)

    def detect_preferences(self, user_message: str) -> list[tuple[str, str]]:
        """从用户消息中检测偏好语句

        Returns:
            [(key, value), ...] 检测到的偏好列表
        """
        found: list[tuple[str, str]] = []
        for pattern in _PREFERENCE_PATTERNS:
            matches = pattern.findall(user_message)
            for match in matches:
                text = "".join(match) if isinstance(match, tuple) else match
                text = text.strip().rstrip("，。！？、")
                if len(text) < 2:
                    continue
                key = self._extract_key(user_message, text)
                found.append((key, text))
        return found

    def _extract_key(self, original: str, value: str) -> str:
        if "不要" in original or "别" in original or "不喜欢" in original:
            return f"avoid_{hash(value) % 10000}"
        if "习惯" in original or "偏好" in original or "倾向" in original:
            return f"prefer_{hash(value) % 10000}"
        if "默认" in original or "以后" in original:
            return f"default_{hash(value) % 10000}"
        return f"pref_{hash(value) % 10000}"

    def apply_detected(self, detections: list[tuple[str, str]]) -> int:
        """应用检测到的偏好，返回实际更新数"""
        count = 0
        for key, value in detections:
            existing = self._preferences.get(key)
            if existing is None or existing["value"] != value:
                self.update(key, value)
                count += 1
        return count
