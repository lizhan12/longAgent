"""执行预算优化

分析预算使用情况并建议优化。
"""

from __future__ import annotations

from typing import Any

from ..base import OptimizationProposal, OptimizationTarget, OptRiskLevel


class BudgetTuner:
    """预算优化器"""

    def __init__(self) -> None:
        self._budget_usage: list[dict[str, Any]] = []

    def record_usage(
        self,
        allocated: float,
        used: float,
        exceeded: bool,
    ) -> None:
        """记录预算使用"""
        self._budget_usage.append({
            "allocated": allocated,
            "used": used,
            "exceeded": exceeded,
            "utilization": used / max(allocated, 0.001),
        })

    def analyze(self) -> list[OptimizationProposal]:
        """分析并生成优化建议"""
        proposals = []

        if len(self._budget_usage) < 5:
            return proposals

        avg_utilization = sum(
            u["utilization"] for u in self._budget_usage
        ) / len(self._budget_usage)
        exceeded_count = sum(1 for u in self._budget_usage if u["exceeded"])

        if avg_utilization > 0.9:
            proposals.append(OptimizationProposal(
                target=OptimizationTarget.BUDGET,
                change=f"预算利用率偏高 ({avg_utilization:.1%})，建议增加预算或优化步骤效率",
                confidence=0.7,
                risk_level=OptRiskLevel.MEDIUM,
                reasoning=f"平均利用率 {avg_utilization:.1%}, 超出次数 {exceeded_count}",
                metrics_before={"avg_utilization": avg_utilization},
                expected_improvement={"avg_utilization": -0.2},
            ))
        elif avg_utilization < 0.3:
            proposals.append(OptimizationProposal(
                target=OptimizationTarget.BUDGET,
                change=f"预算利用率偏低 ({avg_utilization:.1%})，建议减少预算分配以节约资源",
                confidence=0.5,
                risk_level=OptRiskLevel.LOW,
                reasoning=f"平均利用率 {avg_utilization:.1%}",
                metrics_before={"avg_utilization": avg_utilization},
                expected_improvement={"avg_utilization": 0.2},
            ))

        return proposals
