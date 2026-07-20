from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from long.session.models import Session
from long.session.store import SessionStore
from long.session.preference import PreferenceStore
from long.session.profile import UserProfile
from long.session.summary import DailySummaryStore

if TYPE_CHECKING:
    from long.llm.client import LLMClient
    from long.memory.controller import MemoryController
    from long.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)


class SessionManager:

    def __init__(
        self,
        workspace: WorkspaceManager | None,
        session_store: SessionStore | None,
        preference_store: PreferenceStore | None,
        summary_store: DailySummaryStore | None,
        user_profile: UserProfile | None,
        memory: MemoryController | None,
        llm: LLMClient | None,
    ) -> None:
        self.workspace = workspace
        self.session_store = session_store
        self.preference_store = preference_store
        self.summary_store = summary_store
        self.user_profile = user_profile
        self.memory = memory
        self.llm = llm

        self._active_session: Session | None = None
        self._current_session_date: str | None = None
        self._session_dirty: bool = False

    @property
    def active_session(self) -> Session | None:
        return self._active_session

    @property
    def is_dirty(self) -> bool:
        return self._session_dirty

    def init_session_system(self) -> None:
        if self.workspace is None:
            return

        data_dir = self.workspace.data_dir
        self.session_store = SessionStore(data_dir)
        self.preference_store = PreferenceStore(data_dir)
        self.summary_store = DailySummaryStore(data_dir, self.session_store)
        self.user_profile = UserProfile(data_dir)

        latest = self.session_store.load_latest_session()
        if latest is not None:
            self._active_session = latest
            self._current_session_date = latest.date_str
            logger.info("恢复会话: %s (%d 条消息)", latest.id, latest.message_count)
        else:
            self._active_session = Session()
            self._current_session_date = self._active_session.date_str
            logger.info("创建新会话: %s", self._active_session.id)

        prefs = self.preference_store.get_all()
        if prefs:
            logger.info("已加载 %d 条用户偏好", len(prefs))

        pending = self.summary_store.check_pending()
        if pending:
            logger.info("待补生成摘要的日期: %s", pending)

    def ensure_session(self) -> Session:
        today = Session().date_str
        if self._active_session is None or self._current_session_date != today:
            if self._active_session is not None and self.session_store is not None:
                self.session_store.save(self._active_session)
                if self.summary_store is not None:
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.ensure_future(
                                self.daily_summary_and_profile(self._current_session_date)
                            )
                        else:
                            loop.run_until_complete(
                                self.daily_summary_and_profile(self._current_session_date)
                            )
                    except Exception as e:
                        logger.warning("日终处理失败: %s", e)

            self._active_session = Session()
            self._current_session_date = today
            logger.info("日期变更，创建新会话: %s", self._active_session.id)

        return self._active_session

    async def daily_summary_and_profile(self, date_str: str) -> None:
        if self.summary_store is None:
            return

        summary = await self.summary_store.summarize_day(date_str, self.llm)

        if summary and self.user_profile is not None:
            recent_summaries = [s for _, s in self.summary_store.get_recent(days=7)]
            await self.user_profile.extract_from_summaries(recent_summaries, self.llm)

    def save_session(self) -> None:
        if self._active_session is not None and self.session_store is not None:
            self.session_store.save(self._active_session)
            self._session_dirty = False

        self.sync_memory_to_md()

    def sync_memory_to_md(self) -> None:
        if self.workspace is None:
            return

        memory_path = self.workspace.root / "MEMORY.md"
        lines: list[str] = [
            "# MEMORY.md — Agent 长期记忆",
            "",
            "此文件由系统自动维护，记录 Agent 在对话中积累的知识和经验。",
            "每次对话结束后，新的事实会被提炼并写入此文件。",
            "",
            "---",
        ]

        seen: set[str] = set()
        if self.memory is not None:
            try:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # 在已有事件循环中，创建任务异步执行
                    all_items = asyncio.ensure_future(
                        self.memory.search("", limit=30)
                    )
                    # 同步方法中无法 await，跳过动态检索
                    all_items = []
                else:
                    all_items = asyncio.run(self.memory.search("", limit=30))

                if all_items:
                    lines.append("")
                    lines.append("## 记忆条目")
                    for item in all_items:
                        content = getattr(item, "content", str(item))[:200]
                        content_norm = content.strip()
                        if not content_norm or content_norm in seen:
                            continue
                        seen.add(content_norm)
                        lines.append(f"- {content_norm}")
            except Exception:
                pass

        if self.preference_store is not None:
            try:
                prefs = self.preference_store.get_all()
                if prefs:
                    lines.append("")
                    lines.append("## 用户偏好")
                    for k, v in prefs.items():
                        lines.append(f"- {k}: {v}")
            except Exception:
                pass

        try:
            content_text = "\n".join(lines) + "\n"
            memory_path.write_text(content_text, encoding="utf-8")
        except Exception:
            pass

    def mark_dirty(self) -> None:
        self._session_dirty = True

    def save_if_dirty(self) -> None:
        if self._session_dirty:
            self.save_session()
