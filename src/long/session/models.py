from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class Session(BaseModel):
    """会话模型"""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    messages: list[dict[str, str]] = Field(default_factory=list)
    summary: str | None = None

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self.updated_at = datetime.now()

    def pop_last_user_message(self) -> dict[str, str] | None:
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "user":
                return self.messages.pop(i)
        return None

    def recent_messages(self, limit: int = 20) -> list[dict[str, str]]:
        return self.messages[-limit:]

    @property
    def date_str(self) -> str:
        return self.created_at.strftime("%Y-%m-%d")

    @property
    def message_count(self) -> int:
        return len(self.messages)
