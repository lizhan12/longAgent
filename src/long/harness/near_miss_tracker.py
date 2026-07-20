"""Near-Miss 追踪 — 差点出事的那一步才是真正的风险

Harness Engineering 原则：失败即修复（Feedback Loop）扩展
从"只记录实际失败"升级到"也记录差点失败"：
- 约束验证通过但接近阈值 → near miss
- 沙箱执行成功但触发了 WARNING 级别扫描 → near miss
- 定期汇总 near miss 报告，识别系统性风险
- near miss 自动生成 FeedbackLoop 提案

设计约束：
- 零外部依赖
- 低开销：仅记录元数据，不影响执行路径
- 与 AlertManager 和 FeedbackLoop 联动
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class NearMissSeverity(str, Enum):
    """Near-miss 严重程度"""
    LOW = "low"           # 距离阈值 10-20%
    MEDIUM = "medium"     # 距离阈值 5-10%
    HIGH = "high"         # 距离阈值 < 5%


class NearMissCategory(str, Enum):
    """Near-miss 类别"""
    CONSTRAINT = "constraint"     # 约束验证接近阈值
    SANDBOX = "sandbox"           # 沙箱扫描 WARNING
    BUDGET = "budget"             # 预算接近上限
    TIMEOUT = "timeout"           # 执行时间接近超时
    RETRY = "retry"               # 重试后成功（第一次差点失败）
    TOOL_ERROR = "tool_error"     # 工具返回错误但被恢复


@dataclass
class NearMissRecord:
    """Near-miss 记录"""
    record_id: str = ""
    category: NearMissCategory = NearMissCategory.CONSTRAINT
    severity: NearMissSeverity = NearMissSeverity.LOW
    description: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    gap_ratio: float = 0.0  # (threshold - current_value) / threshold
    context: str = ""
    task_id: str = ""
    tool_name: str = ""
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False  # 是否已通过 FeedbackLoop 生成提案


class NearMissTracker:
    """Near-Miss 追踪器

    用法：
        tracker = NearMissTracker(workspace_dir)
        tracker.record(
            category="budget",
            description="Token 消耗接近日预算上限",
            current_value=0.85,
            threshold=0.80,
            task_id="task_1",
        )
        # 定期汇总
        report = tracker.get_report()
        print(f"未解决的 near-miss: {report['unresolved_count']}")
    """

    def __init__(self, workspace_dir: str | Path | None = None) -> None:
        self._records: list[NearMissRecord] = []
        self._dir = Path(workspace_dir) / "near_miss" if workspace_dir else None
        if self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._load()

    def _data_path(self) -> Path | None:
        if self._dir is None:
            return None
        return self._dir / "records.json"

    def _load(self) -> None:
        path = self._data_path()
        if path is None or not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data:
                self._records.append(NearMissRecord(
                    record_id=item.get("record_id", ""),
                    category=NearMissCategory(item.get("category", "constraint")),
                    severity=NearMissSeverity(item.get("severity", "low")),
                    description=item.get("description", ""),
                    current_value=item.get("current_value", 0.0),
                    threshold=item.get("threshold", 0.0),
                    gap_ratio=item.get("gap_ratio", 0.0),
                    context=item.get("context", ""),
                    task_id=item.get("task_id", ""),
                    tool_name=item.get("tool_name", ""),
                    timestamp=item.get("timestamp", 0),
                    resolved=item.get("resolved", False),
                ))
        except (json.JSONDecodeError, TypeError):
            pass

    def _save(self) -> None:
        path = self._data_path()
        if path is None:
            return
        data = [
            {
                "record_id": r.record_id,
                "category": r.category.value,
                "severity": r.severity.value,
                "description": r.description,
                "current_value": r.current_value,
                "threshold": r.threshold,
                "gap_ratio": r.gap_ratio,
                "context": r.context,
                "task_id": r.task_id,
                "tool_name": r.tool_name,
                "timestamp": r.timestamp,
                "resolved": r.resolved,
            }
            for r in self._records
        ]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _compute_severity(gap_ratio: float) -> NearMissSeverity:
        if gap_ratio < 0.05:
            return NearMissSeverity.HIGH
        if gap_ratio < 0.10:
            return NearMissSeverity.MEDIUM
        return NearMissSeverity.LOW

    def record(
        self,
        category: str,
        description: str,
        current_value: float = 0.0,
        threshold: float = 0.0,
        context: str = "",
        task_id: str = "",
        tool_name: str = "",
    ) -> NearMissRecord:
        """记录一条 near-miss

        Args:
            category: 类别 (constraint/sandbox/budget/timeout/retry/tool_error)
            description: 描述
            current_value: 当前值
            threshold: 阈值
            context: 上下文信息
            task_id: 关联任务 ID
            tool_name: 关联工具名
        """
        import uuid

        try:
            cat = NearMissCategory(category)
        except ValueError:
            cat = NearMissCategory.CONSTRAINT

        # 计算 gap_ratio
        gap_ratio = 0.0
        if threshold > 0:
            gap_ratio = abs(threshold - current_value) / threshold

        severity = self._compute_severity(gap_ratio)

        record = NearMissRecord(
            record_id=uuid.uuid4().hex[:8],
            category=cat,
            severity=severity,
            description=description,
            current_value=current_value,
            threshold=threshold,
            gap_ratio=gap_ratio,
            context=context,
            task_id=task_id,
            tool_name=tool_name,
        )
        self._records.append(record)
        self._save()

        logger.info(
            "Near-miss 记录 [%s] %s: %s (severity=%s, gap=%.1f%%)",
            record.record_id, category, description[:80], severity.value, gap_ratio * 100,
        )
        return record

    def mark_resolved(self, record_id: str) -> None:
        """标记为已解决（已通过 FeedbackLoop 生成提案）"""
        for r in self._records:
            if r.record_id == record_id:
                r.resolved = True
                break
        self._save()

    def get_unresolved(self) -> list[NearMissRecord]:
        """获取未解决的 near-miss"""
        return [r for r in self._records if not r.resolved]

    def get_by_category(self, category: str) -> list[NearMissRecord]:
        try:
            cat = NearMissCategory(category)
        except ValueError:
            return []
        return [r for r in self._records if r.category == cat]

    def get_by_severity(self, severity: str) -> list[NearMissRecord]:
        try:
            sev = NearMissSeverity(severity)
        except ValueError:
            return []
        return [r for r in self._records if r.severity == sev]

    def get_report(self) -> dict[str, Any]:
        """生成汇总报告"""
        total = len(self._records)
        unresolved = [r for r in self._records if not r.resolved]

        by_category: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for r in self._records:
            by_category[r.category.value] = by_category.get(r.category.value, 0) + 1
            by_severity[r.severity.value] = by_severity.get(r.severity.value, 0) + 1

        # 高频 near-miss 工具（出现 3 次以上的工具）
        tool_counts: dict[str, int] = {}
        for r in self._records:
            if r.tool_name:
                tool_counts[r.tool_name] = tool_counts.get(r.tool_name, 0) + 1
        frequent_tools = {k: v for k, v in tool_counts.items() if v >= 3}

        return {
            "total": total,
            "unresolved_count": len(unresolved),
            "resolved_count": total - len(unresolved),
            "by_category": by_category,
            "by_severity": by_severity,
            "frequent_tools": frequent_tools,
        }

    def clear_resolved(self) -> int:
        """清除已解决的记录，返回清除数量"""
        before = len(self._records)
        self._records = [r for r in self._records if not r.resolved]
        self._save()
        return before - len(self._records)

    def clear_all(self) -> None:
        self._records.clear()
        self._save()
