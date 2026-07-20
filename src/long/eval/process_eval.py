"""Process Eval - 过程层评估

使用 LLM 裁判评估执行轨迹，支持多 Judge 投票。
"""

from __future__ import annotations

import logging
from typing import Any

from ..ir.ltl import ExecutionHistory
from .report import EvalTask, ProcessResult

logger = logging.getLogger(__name__)


class ProcessEvaluator:
    """过程层评估器

    评估执行轨迹的质量，结合规则检查和 LLM 裁判。

    Attributes:
        _judge: LLM 裁判函数
    """

    def __init__(
        self,
        judge_fn: Any | None = None,
    ) -> None:
        self._judge = judge_fn

    def evaluate(
        self,
        task: EvalTask,
        trace: list[dict[str, Any]] | ExecutionHistory | None = None,
    ) -> ProcessResult:
        """评估执行过程

        Args:
            task: 评估任务
            trace: 执行轨迹

        Returns:
            过程层评估结果
        """
        rule_violations = 0
        ltl_violations = 0
        details: dict[str, Any] = {}

        # 规则检查
        if trace is not None:
            rule_violations = self._rule_based_check(trace)
            details["rule_violations"] = rule_violations

            # LTL 检查
            if isinstance(trace, ExecutionHistory):
                from ..ir.ltl import LTLValidator
                validator = LTLValidator()
                _, errors = validator.check_runtime(trace)
                ltl_violations = len(errors)
                details["ltl_violations"] = ltl_violations

        # 计算过程分数
        score = self._compute_score(rule_violations, ltl_violations, trace)

        # 效率分数
        efficiency = self._compute_efficiency(trace)
        details["efficiency"] = efficiency

        return ProcessResult(
            score=score,
            rule_violations=rule_violations,
            ltl_violations=ltl_violations,
            efficiency=efficiency,
            details=details,
        )

    def _rule_based_check(self, trace: list[dict[str, Any]] | ExecutionHistory) -> int:
        """基于规则的检查

        Returns:
            违反数
        """
        violations = 0

        if isinstance(trace, list):
            # 检查步骤顺序
            actions = [step.get("action", "") for step in trace if isinstance(step, dict)]

            # 规则1: 不能在没有 search 的情况下直接 output
            if "output" in actions and "search" not in actions:
                violations += 1

            # 规则2: 不能在没有 reason 的情况下直接 output
            if "output" in actions and "reason" not in actions:
                violations += 1

        elif isinstance(trace, ExecutionHistory):
            actions = trace.get_actions()
            if "output" in actions and "search" not in actions:
                violations += 1
            if "output" in actions and "reason" not in actions:
                violations += 1

        return violations

    def _compute_score(
        self,
        rule_violations: int,
        ltl_violations: int,
        trace: list[dict[str, Any]] | ExecutionHistory | None,
    ) -> float:
        """计算过程分数"""
        if trace is None:
            return 0.0

        base_score = 1.0
        base_score -= rule_violations * 0.2
        base_score -= ltl_violations * 0.3

        return max(0.0, min(1.0, base_score))

    def _compute_efficiency(
        self,
        trace: list[dict[str, Any]] | ExecutionHistory | None,
    ) -> float:
        """计算效率分数"""
        if trace is None:
            return 0.0

        if isinstance(trace, list):
            steps = len(trace)
        elif isinstance(trace, ExecutionHistory):
            steps = len(trace.steps)
        else:
            return 0.5

        # 3-5步效率最高
        if steps <= 0:
            return 0.0
        elif steps <= 3:
            return 1.0
        elif steps <= 5:
            return 0.9
        elif steps <= 10:
            return 0.7
        else:
            return max(0.3, 1.0 - (steps - 10) * 0.05)


class MultiJudgeVoting:
    """多 Judge 投票

    使用多个 LLM 裁判对执行轨迹进行评估，通过投票减少幻觉。

    Attributes:
        judges: 裁判函数列表
        majority_threshold: 多数阈值
    """

    def __init__(
        self,
        judges: list[Any] | None = None,
        majority_threshold: float = 0.6,
    ) -> None:
        self.judges = judges or []
        self.majority_threshold = majority_threshold

    def evaluate(
        self,
        task: EvalTask,
        trace: list[dict[str, Any]] | ExecutionHistory | None = None,
    ) -> ProcessResult:
        """投票评估

        Args:
            task: 评估任务
            trace: 执行轨迹

        Returns:
            过程层评估结果
        """
        if not self.judges:
            # 无裁判时使用规则评估
            evaluator = ProcessEvaluator()
            return evaluator.evaluate(task, trace)

        # 各裁判独立评估
        scores: list[float] = []
        for judge in self.judges:
            try:
                if callable(judge):
                    result = judge(task, trace)
                    if isinstance(result, (int, float)):
                        scores.append(float(result))
                    elif isinstance(result, ProcessResult):
                        scores.append(result.score)
            except Exception as e:
                logger.warning("Judge evaluation failed: %s", e)

        if not scores:
            # 所有裁判都失败，使用规则评估
            evaluator = ProcessEvaluator()
            return evaluator.evaluate(task, trace)

        # 多数投票
        avg_score = sum(scores) / len(scores)
        approved = sum(1 for s in scores if s >= 0.5)
        approval_rate = approved / len(scores)

        final_score = avg_score if approval_rate >= self.majority_threshold else avg_score * 0.5

        return ProcessResult(
            score=final_score,
            details={
                "judge_scores": scores,
                "approval_rate": approval_rate,
                "num_judges": len(self.judges),
            },
        )

    def _rule_based_check(self, trace: list[dict[str, Any]]) -> int:
        """规则 + LLM 混合检查"""
        violations = 0
        actions = [step.get("action", "") for step in trace if isinstance(step, dict)]
        if "output" in actions and "search" not in actions:
            violations += 1
        return violations
