"""模式分析

分析指标模式，发现优化机会。
"""

from __future__ import annotations

import logging
from typing import Any

from .base import OptimizationProposal, OptimizationTarget, OptRiskLevel
from .collector import MetricsCollector

logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """模式分析器

    分析指标数据中的模式，发现潜在优化机会。

    Attributes:
        _collector: 指标收集器
        _thresholds: 阈值配置
    """

    def __init__(
        self,
        collector: MetricsCollector,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self._collector = collector
        self._thresholds = thresholds or {
            "low_success_rate": 0.5,
            "high_avg_steps": 8.0,
            "high_duration": 60.0,
            "low_eval_score": 0.6,
        }

    def analyze(self) -> list[OptimizationProposal]:
        """分析所有指标，生成优化建议

        Returns:
            优化提案列表
        """
        proposals = []

        # 分析成功率
        success_proposals = self._analyze_success_rate()
        proposals.extend(success_proposals)

        # 分析执行步骤
        step_proposals = self._analyze_steps()
        proposals.extend(step_proposals)

        # 分析执行时长
        duration_proposals = self._analyze_duration()
        proposals.extend(duration_proposals)

        # 分析评估分数
        eval_proposals = self._analyze_eval_scores()
        proposals.extend(eval_proposals)

        return proposals

    def _analyze_success_rate(self) -> list[OptimizationProposal]:
        """分析成功率"""
        proposals = []

        agg = self._collector.get_aggregation("execution.success")
        if agg["count"] < 5:
            return proposals

        success_rate = agg["mean"]
        if success_rate < self._thresholds["low_success_rate"]:
            proposals.append(OptimizationProposal(
                target=OptimizationTarget.PROMPT,
                change=f"系统成功率偏低 ({success_rate:.1%})，建议优化 System Prompt 或增加校验步骤",
                confidence=0.7,
                risk_level=OptRiskLevel.MEDIUM,
                reasoning=f"最近 {int(agg['count'])} 次执行成功率仅 {success_rate:.1%}",
                metrics_before={"success_rate": success_rate},
                expected_improvement={"success_rate": 0.2},
            ))

        return proposals

    def _analyze_steps(self) -> list[OptimizationProposal]:
        """分析执行步骤数"""
        proposals = []

        agg = self._collector.get_aggregation("execution.steps")
        if agg["count"] < 5:
            return proposals

        avg_steps = agg["mean"]
        if avg_steps > self._thresholds["high_avg_steps"]:
            proposals.append(OptimizationProposal(
                target=OptimizationTarget.ROUTING,
                change=f"平均步骤数偏高 ({avg_steps:.1f})，建议优化路由规则减少不必要的搜索",
                confidence=0.6,
                risk_level=OptRiskLevel.LOW,
                reasoning=f"平均步骤 {avg_steps:.1f}，阈值 {self._thresholds['high_avg_steps']}",
                metrics_before={"avg_steps": avg_steps},
                expected_improvement={"avg_steps": -2.0},
            ))

        return proposals

    def _analyze_duration(self) -> list[OptimizationProposal]:
        """分析执行时长"""
        proposals = []

        agg = self._collector.get_aggregation("execution.duration")
        if agg["count"] < 5:
            return proposals

        avg_duration = agg["mean"]
        if avg_duration > self._thresholds["high_duration"]:
            proposals.append(OptimizationProposal(
                target=OptimizationTarget.BUDGET,
                change=f"平均执行时长偏高 ({avg_duration:.1f}s)，建议调整预算分配策略",
                confidence=0.5,
                risk_level=OptRiskLevel.LOW,
                reasoning=f"平均时长 {avg_duration:.1f}s，阈值 {self._thresholds['high_duration']}s",
                metrics_before={"avg_duration": avg_duration},
                expected_improvement={"avg_duration": -10.0},
            ))

        return proposals

    def _analyze_eval_scores(self) -> list[OptimizationProposal]:
        """分析评估分数"""
        proposals = []

        agg = self._collector.get_aggregation("eval.score")
        if agg["count"] < 5:
            return proposals

        avg_score = agg["mean"]
        if avg_score < self._thresholds["low_eval_score"]:
            proposals.append(OptimizationProposal(
                target=OptimizationTarget.TOOL,
                change=f"评估分数偏低 ({avg_score:.2f})，建议检查工具选择策略或增加工具覆盖",
                confidence=0.6,
                risk_level=OptRiskLevel.MEDIUM,
                reasoning=f"平均评估分数 {avg_score:.2f}，阈值 {self._thresholds['low_eval_score']}",
                metrics_before={"avg_eval_score": avg_score},
                expected_improvement={"avg_eval_score": 0.1},
            ))

        return proposals

    def identify_weak_categories(self) -> list[dict[str, Any]]:
        """识别低成功率的任务类别

        Returns:
            弱类别列表
        """
        weak_categories = []

        points = self._collector.get_metrics("eval.score")
        by_category: dict[str, list[float]] = {}

        for point in points:
            category = point.tags.get("category", "unknown")
            by_category.setdefault(category, []).append(point.value)

        for category, scores in by_category.items():
            if len(scores) < 3:
                continue

            avg = sum(scores) / len(scores)
            if avg < self._thresholds["low_eval_score"]:
                weak_categories.append({
                    "category": category,
                    "avg_score": avg,
                    "count": len(scores),
                })

        return weak_categories
