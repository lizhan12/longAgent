"""Prompt 优化目标

分析 System Prompts 效果并建议优化。
"""

from __future__ import annotations

from typing import Any

from ..base import OptimizationProposal, OptimizationTarget, OptRiskLevel


class PromptTuner:
    """Prompt 优化器

    分析 Prompt 效果并生成优化建议。

    Attributes:
        _prompt_scores: Prompt 版本与分数的映射
    """

    def __init__(self) -> None:
        self._prompt_scores: dict[str, list[float]] = {}

    def record_prompt_score(
        self,
        prompt_version: str,
        score: float,
    ) -> None:
        """记录 Prompt 版本分数"""
        self._prompt_scores.setdefault(prompt_version, []).append(score)

    def analyze(self) -> list[OptimizationProposal]:
        """分析并生成优化建议"""
        proposals = []

        for version, scores in self._prompt_scores.items():
            if len(scores) < 5:
                continue

            avg = sum(scores) / len(scores)
            if avg < 0.6:
                proposals.append(OptimizationProposal(
                    target=OptimizationTarget.PROMPT,
                    change=f"Prompt 版本 '{version}' 平均分数 {avg:.2f} 偏低，建议优化措辞或结构",
                    confidence=0.6,
                    risk_level=OptRiskLevel.LOW,
                    reasoning=f"版本 {version}: {len(scores)} 次评估, 平均 {avg:.2f}",
                    metrics_before={"avg_score": avg},
                    expected_improvement={"avg_score": 0.15},
                ))

        return proposals
