from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .models import Session
from .store import SessionStore

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


class DailySummaryStore:
    """每日摘要存储

    生成/存储/加载每日对话摘要。
    失败时写 .pending 标记，下次补生成。
    """

    def __init__(self, data_dir: Path, session_store: SessionStore) -> None:
        self._data_dir = data_dir / "summaries"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._session_store = session_store
        self._summaries_path = self._data_dir / "daily_summaries.json"
        self._summaries: dict[str, str] = {}
        self._retry_counts: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not self._summaries_path.exists():
            return
        try:
            with open(self._summaries_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._summaries = data
        except Exception as e:
            logger.warning("加载每日摘要失败: %s", e)
            self._summaries = {}

    def _save(self) -> None:
        try:
            with open(self._summaries_path, "w", encoding="utf-8") as f:
                json.dump(self._summaries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("保存每日摘要失败: %s", e)

    def get_recent(self, days: int = 7) -> list[tuple[str, str]]:
        """获取最近 N 天的摘要，返回 [(date_str, summary), ...]"""
        today = datetime.now().date()
        result: list[tuple[str, str]] = []
        for i in range(1, days + 1):
            d = today - timedelta(days=i)
            ds = d.strftime("%Y-%m-%d")
            if ds in self._summaries:
                result.append((ds, self._summaries[ds]))
        return result

    def format_for_prompt(self, days: int = 7) -> str:
        """格式化摘要用于 system prompt"""
        recent = self.get_recent(days)
        if not recent:
            return ""
        lines = ["## 近期对话摘要", ""]
        for date_str, summary in recent:
            lines.append(f"- **{date_str}**: {summary}")
        lines.append("")
        return "\n".join(lines)

    async def summarize_day(
        self,
        date_str: str,
        llm_client: Any | None = None,
    ) -> str | None:
        """为指定日期生成摘要

        Args:
            date_str: 日期字符串 YYYY-MM-DD
            llm_client: LLM 客户端

        Returns:
            生成的摘要，失败返回 None
        """
        if date_str in self._summaries:
            return self._summaries[date_str]

        retries = self._retry_counts.get(date_str, 0)
        if retries >= _MAX_RETRIES:
            logger.warning("日期 %s 摘要生成已失败 %d 次，放弃", date_str, retries)
            return None

        sessions = self._session_store.load_sessions_by_date(date_str)
        if not sessions:
            return None

        all_text: list[str] = []
        for session in sessions:
            for msg in session.messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role != "system" and content:
                    all_text.append(f"{role}: {content}")

        if not all_text:
            return None

        conversation_text = "\n".join(all_text)
        if len(conversation_text) > 8000:
            conversation_text = conversation_text[:8000] + "...(截断)"

        if llm_client is None:
            summary = self._rule_based_summary(date_str, sessions)
        else:
            summary = await self._llm_summary(llm_client, conversation_text)

        if summary:
            self._summaries[date_str] = summary
            self._save()
            self._clear_pending(date_str)
            logger.info("日期 %s 摘要生成成功", date_str)
            return summary

        self._retry_counts[date_str] = retries + 1
        self._mark_pending(date_str)
        return None

    async def _llm_summary(self, llm_client: Any, conversation_text: str) -> str | None:
        """使用 LLM 生成摘要"""
        try:
            from long.llm.base import LLMMessage

            messages = [
                LLMMessage(
                    role="system",
                    content=(
                        "请对以下对话内容生成简洁的每日摘要，"
                        "重点提取：1)讨论的主要话题 2)完成的关键任务 3)重要的决策和结论。"
                        "摘要应控制在200字以内。"
                    ),
                ),
                LLMMessage(
                    role="user",
                    content=f"请摘要以下对话：\n\n{conversation_text}",
                ),
            ]

            response = await llm_client.chat(messages, purpose="summarize")
            summary: str = response.content.strip()
            return summary if summary else None
        except Exception as e:
            logger.warning("LLM 摘要生成失败: %s", e)
            return None

    def _rule_based_summary(self, date_str: str, sessions: list[Session]) -> str:
        """基于规则的摘要（LLM 不可用时的降级方案）"""
        total_msgs = sum(s.message_count for s in sessions)
        user_msgs = sum(
            1 for s in sessions for m in s.messages if m.get("role") == "user"
        )
        topics: list[str] = []
        for s in sessions:
            for m in s.messages:
                if m.get("role") == "user":
                    content = m.get("content", "")
                    first_line = content.split("\n")[0][:50]
                    if first_line:
                        topics.append(first_line)

        topic_str = "、".join(topics[:5]) if topics else "无"
        return f"共{len(sessions)}个会话、{total_msgs}条消息（用户{user_msgs}条），话题：{topic_str}"

    def _mark_pending(self, date_str: str) -> None:
        path = self._data_dir / f"{date_str}.pending"
        try:
            path.write_text(date_str, encoding="utf-8")
        except Exception:
            pass

    def _clear_pending(self, date_str: str) -> None:
        path = self._data_dir / f"{date_str}.pending"
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass

    def check_pending(self) -> list[str]:
        """返回待补生成的日期列表"""
        pending: list[str] = []
        if not self._data_dir.exists():
            return pending
        for f in self._data_dir.glob("*.pending"):
            date_str = f.stem
            if date_str not in self._summaries:
                retries = self._retry_counts.get(date_str, 0)
                if retries < _MAX_RETRIES:
                    pending.append(date_str)
        return pending
