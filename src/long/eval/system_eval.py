"""System Eval - 系统层评估

评估系统的稳定性、收敛性和失败模式。
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from .report import EvalTask, SystemResult

logger = logging.getLogger(__name__)


class SystemEvaluator:
    """系统层评估器

    评估系统在不同条件下的表现。

    Attributes:
        stability_threshold: 稳定性阈值
    """

    def __init__(self, stability_threshold: float = 0.8) -> None:
        self.stability_threshold = stability_threshold

    def evaluate_stability(
        self,
        task: EvalTask,
        results: list[float],
    ) -> SystemResult:
        """稳定性评估

        对同一输入多次运行，评估输出一致性。

        Args:
            task: 评估任务
            results: 多次运行的分数列表

        Returns:
            系统评估结果
        """
        if not results:
            return SystemResult(stability=0.0)

        avg_score = sum(results) / len(results)
        variance = sum((s - avg_score) ** 2 for s in results) / len(results)
        std_dev = variance ** 0.5

        # 稳定性: 低标准差 = 高稳定性
        if avg_score > 0:
            cv = std_dev / avg_score  # 变异系数
            stability = max(0.0, 1.0 - cv)
        else:
            stability = 0.0

        failure_modes = []
        if stability < self.stability_threshold:
            failure_modes.append(f"low_stability: cv={cv:.3f}" if avg_score > 0 else "zero_score")
        if any(r == 0.0 for r in results):
            failure_modes.append("complete_failure")

        return SystemResult(
            stability=stability,
            details={
                "avg_score": avg_score,
                "std_dev": std_dev,
                "min_score": min(results),
                "max_score": max(results),
                "num_runs": len(results),
            },
            failure_modes=failure_modes,
        )

    def evaluate_convergence(
        self,
        score_series: list[list[float]],
    ) -> SystemResult:
        """收敛性评估

        评估系统在多轮执行中是否逐步改进。

        Args:
            score_series: 多轮执行的分数序列

        Returns:
            系统评估结果
        """
        if not score_series:
            return SystemResult(convergence=0.0)

        # 计算各轮平均分
        round_avgs = []
        for scores in score_series:
            if scores:
                round_avgs.append(sum(scores) / len(scores))

        if len(round_avgs) < 2:
            return SystemResult(convergence=0.5)

        # 收敛: 后期分数 > 前期分数
        first_half = round_avgs[: len(round_avgs) // 2]
        second_half = round_avgs[len(round_avgs) // 2 :]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        if first_avg > 0:
            improvement = (second_avg - first_avg) / first_avg
            convergence = min(1.0, max(0.0, 0.5 + improvement))
        else:
            convergence = 0.5

        failure_modes = []
        if convergence < 0.5:
            failure_modes.append("non_convergent")

        return SystemResult(
            convergence=convergence,
            details={
                "round_averages": round_avgs,
                "first_half_avg": first_avg,
                "second_half_avg": second_avg,
                "improvement": second_avg - first_avg,
            },
            failure_modes=failure_modes,
        )

    def evaluate_failure_modes(
        self,
        failures: list[dict[str, Any]],
    ) -> SystemResult:
        """失败模式分析

        Args:
            failures: 失败记录列表

        Returns:
            系统评估结果
        """
        if not failures:
            return SystemResult(
                stability=1.0,
                convergence=1.0,
                failure_modes=[],
            )

        # 统计失败类型
        error_types = Counter()
        for failure in failures:
            error_type = failure.get("type", "unknown")
            error_types[error_type] += 1

        # 计算失败率
        total = sum(error_types.values())
        failure_modes = [
            f"{error_type}: {count} ({count / total * 100:.1f}%)"
            for error_type, count in error_types.most_common()
        ]

        stability = max(0.0, 1.0 - total / max(total + 10, 1))

        return SystemResult(
            stability=stability,
            details={
                "total_failures": total,
                "error_distribution": dict(error_types),
            },
            failure_modes=failure_modes,
        )
