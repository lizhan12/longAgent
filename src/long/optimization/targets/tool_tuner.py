"""工具选择优化

分析工具使用效果并建议优化。
"""

from __future__ import annotations

from typing import Any

from ..base import OptimizationProposal, OptimizationTarget, OptRiskLevel


class ToolTuner:
    """工具选择优化器"""

    def __init__(self) -> None:
        self._tool_stats: dict[str, dict[str, list[float]]] = {}

    def record_tool_use(
        self,
        tool_name: str,
        success: bool,
        duration: float,
    ) -> None:
        """记录工具使用"""
        stats = self._tool_stats.setdefault(tool_name, {
            "success": [],
            "duration": [],
        })
        stats["success"].append(1.0 if success else 0.0)
        stats["duration"].append(duration)

    def analyze(self) -> list[OptimizationProposal]:
        """分析并生成优化建议"""
        proposals = []

        for tool_name, stats in self._tool_stats.items():
            successes = stats["success"]
            if len(successes) < 5:
                continue

            avg_success = sum(successes) / len(successes)
            avg_duration = sum(stats["duration"]) / len(stats["duration"])

            if avg_success < 0.5:
                proposals.append(OptimizationProposal(
                    target=OptimizationTarget.TOOL,
                    change=f"工具 '{tool_name}' 成功率 {avg_success:.1%} 偏低，建议检查参数或替换替代工具",
                    confidence=0.6,
                    risk_level=OptRiskLevel.LOW,
                    reasoning=f"工具 {tool_name}: {len(successes)} 次, 成功率 {avg_success:.1%}",
                    metrics_before={"avg_success_rate": avg_success},
                    expected_improvement={"avg_success_rate": 0.2},
                ))

            if avg_duration > 10.0:
                proposals.append(OptimizationProposal(
                    target=OptimizationTarget.TOOL,
                    change=f"工具 '{tool_name}' 平均耗时 {avg_duration:.1f}s 偏高，建议优化或缓存",
                    confidence=0.5,
                    risk_level=OptRiskLevel.LOW,
                    reasoning=f"工具 {tool_name}: 平均耗时 {avg_duration:.1f}s",
                    metrics_before={"avg_duration": avg_duration},
                    expected_improvement={"avg_duration": -5.0},
                ))

        return proposals
