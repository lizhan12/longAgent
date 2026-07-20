"""决策史 — 记录重大判断的背景、选项、理由和预期

Harness Engineering 原则：经验必须外化（Memory Layer）
从"判断只留在脑子里"升级到"判断写进系统里"：
- 每次重大决策在当下记录，而非事后补写
- 决策之间可连线，模式随时间浮现
- 文件越厚，越难回到从前的盲目

设计约束：
- 格式简单，关键是持续
- 与 MEMORY.md 兼容，可被引用
- 支持事后复盘对照（事前预期 vs 事后现实）
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class DecisionCategory(str, Enum):
    ARCHITECTURE = "architecture"
    TOOL = "tool"
    MODEL = "model"
    SECURITY = "security"
    PROMPT = "prompt"
    WORKFLOW = "workflow"


class DecisionStatus(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"


@dataclass
class DecisionRecord:
    """单条决策记录"""
    decision_id: str = ""
    category: DecisionCategory = DecisionCategory.ARCHITECTURE
    status: DecisionStatus = DecisionStatus.PROPOSED
    title: str = ""
    context: str = ""           # 决策背景
    options: list[str] = field(default_factory=list)  # 考虑过的选项
    chosen: str = ""            # 选择了什么
    rationale: str = ""         # 为什么这么选
    expected_outcome: str = ""  # 事前预期
    actual_outcome: str = ""    # 事后现实（复盘时填写）
    created_at: float = field(default_factory=time.time)
    reviewed_at: float = 0.0
    superseded_by: str = ""     # 被哪条决策取代
    tags: list[str] = field(default_factory=list)


class DecisionLog:
    """决策史管理器

    用法：
        log = DecisionLog(workspace_dir)
        record = log.record(
            category="architecture",
            title="选择进程沙箱而非容器",
            context="个人工具不需要 Docker",
            options=["进程沙箱", "Docker 容器", "无沙箱"],
            chosen="进程沙箱",
            rationale="个人工具不需要 Docker，进程沙箱足够安全且更轻量",
            expected_outcome="安全隔离 + 低延迟",
        )
        log.review(record.decision_id, actual_outcome="运行稳定，隔离有效")
    """

    def __init__(self, workspace_dir: str | Path) -> None:
        self._dir = Path(workspace_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, DecisionRecord] = {}
        self._load()

    def _decisions_path(self) -> Path:
        return self._dir / "decisions.json"

    def _markdown_path(self) -> Path:
        return self._dir.parent / "DECISIONS.md"

    def _load(self) -> None:
        path = self._decisions_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for item in data:
                    record = DecisionRecord(
                        decision_id=item.get("decision_id", ""),
                        category=DecisionCategory(item.get("category", "architecture")),
                        status=DecisionStatus(item.get("status", "proposed")),
                        title=item.get("title", ""),
                        context=item.get("context", ""),
                        options=item.get("options", []),
                        chosen=item.get("chosen", ""),
                        rationale=item.get("rationale", ""),
                        expected_outcome=item.get("expected_outcome", ""),
                        actual_outcome=item.get("actual_outcome", ""),
                        created_at=item.get("created_at", 0),
                        reviewed_at=item.get("reviewed_at", 0),
                        superseded_by=item.get("superseded_by", ""),
                        tags=item.get("tags", []),
                    )
                    self._records[record.decision_id] = record
            except (json.JSONDecodeError, TypeError):
                pass

    def _save(self) -> None:
        data = []
        for r in self._records.values():
            data.append({
                "decision_id": r.decision_id,
                "category": r.category.value,
                "status": r.status.value,
                "title": r.title,
                "context": r.context,
                "options": r.options,
                "chosen": r.chosen,
                "rationale": r.rationale,
                "expected_outcome": r.expected_outcome,
                "actual_outcome": r.actual_outcome,
                "created_at": r.created_at,
                "reviewed_at": r.reviewed_at,
                "superseded_by": r.superseded_by,
                "tags": r.tags,
            })
        self._decisions_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._sync_markdown()

    def _sync_markdown(self) -> None:
        """同步到 DECISIONS.md（人类可读版本）"""
        lines = ["# 决策史 (Decision Log)", "", "> Harness 原则：经验必须外化。判断写在当下，不是事后补。", ""]

        by_category: dict[str, list[DecisionRecord]] = {}
        for r in self._records.values():
            cat = r.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r)

        for cat in sorted(by_category.keys()):
            records = sorted(by_category[cat], key=lambda r: r.created_at, reverse=True)
            lines.append(f"## {cat}")
            lines.append("")
            for r in records:
                status_icon = {"proposed": "📝", "accepted": "✅", "deprecated": "❌", "superseded": "🔄"}.get(r.status.value, "?")
                lines.append(f"### {status_icon} {r.title}")
                lines.append("")
                lines.append(f"- **ID**: {r.decision_id}")
                lines.append(f"- **背景**: {r.context}")
                if r.options:
                    lines.append(f"- **选项**: {', '.join(r.options)}")
                lines.append(f"- **选择**: {r.chosen}")
                lines.append(f"- **理由**: {r.rationale}")
                lines.append(f"- **预期**: {r.expected_outcome}")
                if r.actual_outcome:
                    lines.append(f"- **实际**: {r.actual_outcome}")
                if r.superseded_by:
                    lines.append(f"- **被取代**: {r.superseded_by}")
                if r.tags:
                    lines.append(f"- **标签**: {', '.join(r.tags)}")
                lines.append("")

        self._markdown_path().write_text("\n".join(lines), encoding="utf-8")

    def record(
        self,
        category: str,
        title: str,
        context: str = "",
        options: list[str] | None = None,
        chosen: str = "",
        rationale: str = "",
        expected_outcome: str = "",
        tags: list[str] | None = None,
    ) -> DecisionRecord:
        """记录一条决策"""
        import uuid
        try:
            cat = DecisionCategory(category)
        except ValueError:
            cat = DecisionCategory.ARCHITECTURE

        record = DecisionRecord(
            decision_id=uuid.uuid4().hex[:8],
            category=cat,
            status=DecisionStatus.ACCEPTED,
            title=title,
            context=context,
            options=options or [],
            chosen=chosen,
            rationale=rationale,
            expected_outcome=expected_outcome,
            tags=tags or [],
        )
        self._records[record.decision_id] = record
        self._save()
        logger.info("决策已记录: %s [%s] %s", record.decision_id, category, title[:80])
        return record

    def review(self, decision_id: str, actual_outcome: str) -> DecisionRecord | None:
        """复盘一条决策"""
        record = self._records.get(decision_id)
        if record is None:
            return None
        record.actual_outcome = actual_outcome
        record.reviewed_at = time.time()
        self._save()
        logger.info("决策已复盘: %s", decision_id)
        return record

    def deprecate(self, decision_id: str, superseded_by: str = "") -> DecisionRecord | None:
        """废弃一条决策"""
        record = self._records.get(decision_id)
        if record is None:
            return None
        record.status = DecisionStatus.DEPRECATED
        record.superseded_by = superseded_by
        self._save()
        return record

    def get(self, decision_id: str) -> DecisionRecord | None:
        return self._records.get(decision_id)

    def list_by_category(self, category: str) -> list[DecisionRecord]:
        try:
            cat = DecisionCategory(category)
        except ValueError:
            return []
        return [r for r in self._records.values() if r.category == cat]

    def list_unreviewed(self) -> list[DecisionRecord]:
        """列出未复盘的决策"""
        return [r for r in self._records.values() if r.reviewed_at == 0]

    def get_stats(self) -> dict[str, int]:
        stats: dict[str, int] = {}
        for r in self._records.values():
            key = r.category.value
            stats[key] = stats.get(key, 0) + 1
        stats["total"] = len(self._records)
        stats["unreviewed"] = len(self.list_unreviewed())
        return stats
