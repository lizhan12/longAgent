"""优化基础模型

定义优化目标、提案和风险级别。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OptimizationTarget(str, Enum):
    """优化目标"""

    PROMPT = "prompt"
    ROUTING = "routing"
    BUDGET = "budget"
    TOOL = "tool"


class OptRiskLevel(str, Enum):
    """优化风险级别"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OptimizationProposal(BaseModel):
    """优化提案

    Attributes:
        target: 优化目标
        change: 变更描述
        confidence: 置信度 (0-1)
        risk_level: 风险级别
        reasoning: 推理依据
        metrics_before: 变更前指标
        expected_improvement: 预期改进
    """

    target: OptimizationTarget
    change: str
    confidence: float = 0.5
    risk_level: OptRiskLevel = OptRiskLevel.MEDIUM
    reasoning: str = ""
    metrics_before: dict[str, float] = Field(default_factory=dict)
    expected_improvement: dict[str, float] = Field(default_factory=dict)


class ProposalStatus(str, Enum):
    """提案状态"""

    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"


class ProposalRecord(BaseModel):
    """提案记录（不可变审计）"""

    proposal: OptimizationProposal
    status: ProposalStatus = ProposalStatus.PROPOSED
    proposed_at: float = 0.0
    reviewed_at: float | None = None
    applied_at: float | None = None
    reviewer: str | None = None
    review_comment: str | None = None
    rollback_reason: str | None = None
    metrics_after: dict[str, float] = Field(default_factory=dict)
