"""Eval Pipeline

整合三层评估，执行 80% 自动 + 20% 人工审核流程。
"""

from __future__ import annotations

import logging
from typing import Any

from .adversarial import AdversarialTestSuite
from .dataset_manager import EvalDatasetManager
from .outcome_eval import OutcomeEvaluator
from .process_eval import ProcessEvaluator, MultiJudgeVoting
from .report import EvalReport, EvalTask, SystemResult
from .system_eval import SystemEvaluator

logger = logging.getLogger(__name__)


class EvalPipeline:
    """评估流水线

    整合三层评估（结果层、过程层、系统层），执行评估并决定是否需要人工审核。

    Attributes:
        outcome_evaluator: 结果层评估器
        process_evaluator: 过程层评估器
        system_evaluator: 系统层评估器
        dataset_manager: 数据集管理器
        auto_review_threshold: 自动审核阈值（低于此分数需要人工审核）
    """

    def __init__(
        self,
        outcome_evaluator: OutcomeEvaluator | None = None,
        process_evaluator: ProcessEvaluator | None = None,
        system_evaluator: SystemEvaluator | None = None,
        dataset_manager: EvalDatasetManager | None = None,
        auto_review_threshold: float = 0.6,
    ) -> None:
        self.outcome_evaluator = outcome_evaluator or OutcomeEvaluator()
        self.process_evaluator = process_evaluator or ProcessEvaluator()
        self.system_evaluator = system_evaluator or SystemEvaluator()
        self.dataset_manager = dataset_manager or EvalDatasetManager()
        self.auto_review_threshold = auto_review_threshold

    def run(
        self,
        task: EvalTask,
        output: str | dict[str, Any] | None = None,
        trace: list[dict[str, Any]] | None = None,
    ) -> EvalReport:
        """执行评估

        流程: 结果层评估 → 过程层评估 → 系统层评估 → 决定是否需要人工审核

        Args:
            task: 评估任务
            output: 实际输出
            trace: 执行轨迹

        Returns:
            评估报告
        """
        # 1. 结果层评估
        outcome = self.outcome_evaluator.evaluate(task, output)

        # 2. 过程层评估（过程层权重 > 结果层）
        process = self.process_evaluator.evaluate(task, trace)

        # 3. 系统层评估（如果有多次运行数据）
        system = SystemResult()  # 单次运行不做系统评估

        # 4. 计算综合分数（过程层权重 > 结果层）
        score = self._compute_composite_score(outcome, process, system)

        # 5. 决定是否需要人工审核
        # 80% 自动 + 20% 人工
        needs_human_review = score < self.auto_review_threshold
        auto_reviewed = not needs_human_review

        return EvalReport(
            task=task,
            outcome=outcome,
            process=process,
            system=system,
            needs_human_review=needs_human_review,
            score=score,
            auto_reviewed=auto_reviewed,
        )

    def run_batch(
        self,
        tasks: list[EvalTask],
        executor: Any | None = None,
    ) -> list[EvalReport]:
        """批量评估

        Args:
            tasks: 评估任务列表
            executor: 执行器（用于实际运行任务）

        Returns:
            评估报告列表
        """
        reports = []

        for task in tasks:
            if executor:
                # 使用执行器运行任务
                try:
                    result = executor(task)
                    output = result.get("output")
                    trace = result.get("trace")
                except Exception as e:
                    output = None
                    trace = None
                    logger.error("Task execution failed: %s - %s", task.name, e)
            else:
                output = None
                trace = None

            report = self.run(task, output, trace)
            reports.append(report)

        return reports

    def _compute_composite_score(
        self,
        outcome: Any,
        process: Any,
        system: Any,
    ) -> float:
        """计算综合分数

        权重: 过程层 0.5 > 结果层 0.3 > 系统层 0.2
        """
        outcome_weight = 0.3
        process_weight = 0.5
        system_weight = 0.2

        outcome_score = outcome.accuracy if hasattr(outcome, "accuracy") else 0.0
        process_score = process.score if hasattr(process, "score") else 0.0
        system_score = system.stability if hasattr(system, "stability") else 0.0

        # 如果系统评估无数据，重新分配权重
        if system_score == 0.0 and not hasattr(system, "details"):
            outcome_weight = 0.4
            process_weight = 0.6
            system_weight = 0.0

        score = (
            outcome_weight * outcome_score
            + process_weight * process_score
            + system_weight * system_score
        )

        return max(0.0, min(1.0, score))
