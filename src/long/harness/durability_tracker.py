"""耐久性追踪 — 识别长任务中的跑偏拐点

Harness Engineering 原则：耐久性（Durability）
从"只看最终结果"升级到"追踪每一步的成功率"：
- 记录长任务执行中每一步的成功/失败
- 生成耐久性曲线（step_index → success_rate）
- 识别"跑偏拐点"（从哪一步开始准确率下降）
- 与 EvalPipeline 集成

设计约束：
- 零外部依赖
- 低开销：每步仅记录一个 bool + 时间戳
- 支持跨任务聚合分析
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """单步执行记录"""
    step_index: int = 0
    task_id: str = ""
    tool_name: str = ""
    success: bool = True
    error_type: str = ""  # 错误类型（如 timeout, constraint_violation, tool_error）
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class DurabilityReport:
    """耐久性报告"""
    task_id: str = ""
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    drift_point: int = 0  # 跑偏拐点（从第几步开始成功率下降）
    drift_confidence: float = 0.0  # 拐点检测置信度
    step_success_rates: list[float] = field(default_factory=list)  # 每步累计成功率
    error_types: dict[str, int] = field(default_factory=dict)  # 错误类型分布
    avg_step_duration_ms: float = 0.0

    @property
    def overall_success_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.successful_steps / self.total_steps


class DurabilityTracker:
    """耐久性追踪器

    用法：
        tracker = DurabilityTracker(workspace_dir)
        tracker.record_step("task_1", step_index=0, tool_name="read_file", success=True)
        tracker.record_step("task_1", step_index=1, tool_name="write_file", success=True)
        tracker.record_step("task_1", step_index=2, tool_name="execute_code", success=False, error_type="tool_error")
        report = tracker.get_report("task_1")
        print(f"跑偏拐点: 第 {report.drift_point} 步")
    """

    def __init__(self, workspace_dir: str | Path | None = None) -> None:
        self._steps: dict[str, list[StepRecord]] = {}  # task_id → steps
        self._dir = Path(workspace_dir) / "durability" if workspace_dir else None
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
            for task_id, steps_data in data.items():
                steps = []
                for s in steps_data:
                    steps.append(StepRecord(
                        step_index=s.get("step_index", 0),
                        task_id=task_id,
                        tool_name=s.get("tool_name", ""),
                        success=s.get("success", True),
                        error_type=s.get("error_type", ""),
                        duration_ms=s.get("duration_ms", 0.0),
                        timestamp=s.get("timestamp", 0),
                    ))
                self._steps[task_id] = steps
        except (json.JSONDecodeError, TypeError):
            pass

    def _save(self) -> None:
        path = self._data_path()
        if path is None:
            return
        data = {}
        for task_id, steps in self._steps.items():
            data[task_id] = [
                {
                    "step_index": s.step_index,
                    "tool_name": s.tool_name,
                    "success": s.success,
                    "error_type": s.error_type,
                    "duration_ms": s.duration_ms,
                    "timestamp": s.timestamp,
                }
                for s in steps
            ]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def record_step(
        self,
        task_id: str,
        step_index: int,
        tool_name: str = "",
        success: bool = True,
        error_type: str = "",
        duration_ms: float = 0.0,
    ) -> None:
        """记录一步执行结果"""
        if task_id not in self._steps:
            self._steps[task_id] = []

        record = StepRecord(
            step_index=step_index,
            task_id=task_id,
            tool_name=tool_name,
            success=success,
            error_type=error_type,
            duration_ms=duration_ms,
        )
        self._steps[task_id].append(record)
        self._save()

        if not success:
            logger.info(
                "耐久性追踪 [%s] 第%d步失败: tool=%s, error=%s",
                task_id, step_index, tool_name, error_type,
            )

    def get_report(self, task_id: str) -> DurabilityReport:
        """生成耐久性报告"""
        steps = self._steps.get(task_id, [])
        if not steps:
            return DurabilityReport(task_id=task_id)

        total = len(steps)
        successful = sum(1 for s in steps if s.success)
        failed = total - successful

        # 计算每步累计成功率
        cumulative_success_rates: list[float] = []
        running_success = 0
        for i, s in enumerate(steps):
            if s.success:
                running_success += 1
            cumulative_success_rates.append(running_success / (i + 1))

        # 错误类型分布
        error_types: dict[str, int] = {}
        for s in steps:
            if not s.success and s.error_type:
                error_types[s.error_type] = error_types.get(s.error_type, 0) + 1

        # 平均步长
        durations = [s.duration_ms for s in steps if s.duration_ms > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        # 检测跑偏拐点：找到累计成功率下降最快的点
        drift_point = 0
        drift_confidence = 0.0
        if len(cumulative_success_rates) >= 3:
            max_drop = 0.0
            for i in range(1, len(cumulative_success_rates)):
                drop = cumulative_success_rates[i - 1] - cumulative_success_rates[i]
                if drop > max_drop:
                    max_drop = drop
                    drift_point = i
                    drift_confidence = min(drop * 10, 1.0)  # 归一化到 0-1

        return DurabilityReport(
            task_id=task_id,
            total_steps=total,
            successful_steps=successful,
            failed_steps=failed,
            drift_point=drift_point,
            drift_confidence=drift_confidence,
            step_success_rates=cumulative_success_rates,
            error_types=error_types,
            avg_step_duration_ms=avg_duration,
        )

    def get_aggregate_report(self) -> DurabilityReport:
        """跨任务聚合分析"""
        all_steps: list[StepRecord] = []
        for steps in self._steps.values():
            all_steps.extend(steps)

        if not all_steps:
            return DurabilityReport(task_id="__aggregate__")

        # 按步索引分组
        by_index: dict[int, list[StepRecord]] = {}
        for s in all_steps:
            if s.step_index not in by_index:
                by_index[s.step_index] = []
            by_index[s.step_index].append(s)

        # 计算每个步索引的成功率
        step_rates: list[float] = []
        for idx in sorted(by_index.keys()):
            records = by_index[idx]
            success_count = sum(1 for r in records if r.success)
            step_rates.append(success_count / len(records))

        total = len(all_steps)
        successful = sum(1 for s in all_steps if s.success)

        error_types: dict[str, int] = {}
        for s in all_steps:
            if not s.success and s.error_type:
                error_types[s.error_type] = error_types.get(s.error_type, 0) + 1

        # 检测全局跑偏拐点
        drift_point = 0
        drift_confidence = 0.0
        if len(step_rates) >= 3:
            max_drop = 0.0
            for i in range(1, len(step_rates)):
                drop = step_rates[i - 1] - step_rates[i]
                if drop > max_drop:
                    max_drop = drop
                    drift_point = i
                    drift_confidence = min(drop * 10, 1.0)

        return DurabilityReport(
            task_id="__aggregate__",
            total_steps=total,
            successful_steps=successful,
            failed_steps=total - successful,
            drift_point=drift_point,
            drift_confidence=drift_confidence,
            step_success_rates=step_rates,
            error_types=error_types,
        )

    def get_task_ids(self) -> list[str]:
        return list(self._steps.keys())

    def clear_task(self, task_id: str) -> None:
        if task_id in self._steps:
            del self._steps[task_id]
            self._save()

    def clear_all(self) -> None:
        self._steps.clear()
        self._save()
