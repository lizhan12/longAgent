"""路由规则优化

分析路由决策效果并建议优化。
"""

from __future__ import annotations

from typing import Any

from ..base import OptimizationProposal, OptimizationTarget, OptRiskLevel


class RoutingTuner:
    """路由优化器"""

    def __init__(self) -> None:
        self._route_stats: dict[str, dict[str, list[float]]] = {}

    def record_route(
        self,
        route_name: str,
        success: bool,
        duration: float,
    ) -> None:
        """记录路由结果"""
        stats = self._route_stats.setdefault(route_name, {
            "success": [],
            "duration": [],
        })
        stats["success"].append(1.0 if success else 0.0)
        stats["duration"].append(duration)

    def analyze(self) -> list[OptimizationProposal]:
        """分析并生成优化建议"""
        proposals = []

        for route_name, stats in self._route_stats.items():
            successes = stats["success"]
            if len(successes) < 5:
                continue

            avg_success = sum(successes) / len(successes)
            if avg_success < 0.5:
                proposals.append(OptimizationProposal(
                    target=OptimizationTarget.ROUTING,
                    change=f"路由 '{route_name}' 成功率 {avg_success:.1%} 偏低，建议调整路由规则",
                    confidence=0.7,
                    risk_level=OptRiskLevel.MEDIUM,
                    reasoning=f"路由 {route_name}: {len(successes)} 次, 成功率 {avg_success:.1%}",
                    metrics_before={"avg_success_rate": avg_success},
                    expected_improvement={"avg_success_rate": 0.2},
                ))

        return proposals
