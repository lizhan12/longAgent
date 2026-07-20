"""人工审批门

实现 HITL 审批矩阵，所有优化变更需人工确认。
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from .base import OptimizationProposal, OptRiskLevel

logger = logging.getLogger(__name__)


class ApprovalDecision(BaseModel):
    """审批决定"""

    approved: bool = False
    reviewer: str = ""
    comment: str = ""
    conditions: list[str] = []


class ApprovalRule(BaseModel):
    """审批规则

    Attributes:
        risk_level: 适用的风险级别
        requires_approval: 是否需要审批
        max_auto_confidence: 自动批准的最大置信度
    """

    risk_level: OptRiskLevel
    requires_approval: bool = True
    max_auto_confidence: float = 0.0


# 默认审批矩阵
DEFAULT_APPROVAL_MATRIX: list[ApprovalRule] = [
    ApprovalRule(risk_level=OptRiskLevel.LOW, requires_approval=True, max_auto_confidence=0.9),
    ApprovalRule(risk_level=OptRiskLevel.MEDIUM, requires_approval=True, max_auto_confidence=0.0),
    ApprovalRule(risk_level=OptRiskLevel.HIGH, requires_approval=True, max_auto_confidence=0.0),
    ApprovalRule(risk_level=OptRiskLevel.CRITICAL, requires_approval=True, max_auto_confidence=0.0),
]


class HumanApprovalGate:
    """人工审批门

    根据审批矩阵决定是否自动批准或需要人工审批。

    Attributes:
        _rules: 审批规则
        _pending: 待审批的提案
        _change_frequency: 变更频率限制（秒）
        _last_change_time: 上次变更时间
    """

    def __init__(
        self,
        rules: list[ApprovalRule] | None = None,
        change_frequency_limit: float = 3600.0,
        auto_approve_low_risk: bool = False,
    ) -> None:
        self._rules = rules or DEFAULT_APPROVAL_MATRIX
        self._pending: dict[str, OptimizationProposal] = {}
        self._change_frequency_limit = change_frequency_limit
        self._last_change_time: float = 0.0
        self._auto_approve_low_risk = auto_approve_low_risk

    def review(
        self,
        proposal: OptimizationProposal,
        auto_review: bool = False,
    ) -> ApprovalDecision:
        """审核提案

        Args:
            proposal: 优化提案
            auto_review: 是否自动审核（跳过人工）

        Returns:
            审批决定
        """
        # 检查变更频率限制
        import time
        now = time.time()
        if (now - self._last_change_time) < self._change_frequency_limit:
            return ApprovalDecision(
                approved=False,
                comment=f"变更频率限制: 距离上次变更不足 {self._change_frequency_limit}s",
            )

        # 查找对应规则
        rule = self._find_rule(proposal.risk_level)

        if rule is None:
            return ApprovalDecision(
                approved=False,
                comment=f"No approval rule for risk level: {proposal.risk_level}",
            )

        # CRITICAL 级别永远需要人工审批
        if proposal.risk_level == OptRiskLevel.CRITICAL:
            return ApprovalDecision(
                approved=False,
                comment="CRITICAL risk level always requires human approval",
            )

        # 自动审批检查
        if self._auto_approve_low_risk and auto_review:
            if proposal.confidence >= rule.max_auto_confidence > 0:
                self._last_change_time = now
                return ApprovalDecision(
                    approved=True,
                    reviewer="auto",
                    comment=f"Auto-approved: confidence {proposal.confidence:.2f} >= {rule.max_auto_confidence}",
                )

        # 安全阈值目标永远不自动批准
        if proposal.target.value in {"state_machine", "ltl_rules", "security_policy"}:
            return ApprovalDecision(
                approved=False,
                comment=f"Target '{proposal.target.value}' requires human approval for safety",
            )

        # 默认: 需要人工审批
        if not rule.requires_approval:
            self._last_change_time = now
            return ApprovalDecision(
                approved=True,
                reviewer="auto",
                comment="No approval required for this risk level",
            )

        # 添加到待审批列表
        self._pending[proposal.change] = proposal

        return ApprovalDecision(
            approved=False,
            comment="Requires human approval",
        )

    def manual_approve(
        self,
        proposal_change: str,
        reviewer: str = "",
        comment: str = "",
    ) -> ApprovalDecision:
        """人工批准

        Args:
            proposal_change: 提案变更描述
            reviewer: 审核人
            comment: 审核意见

        Returns:
            审批决定
        """
        proposal = self._pending.pop(proposal_change, None)
        if proposal is None:
            return ApprovalDecision(
                approved=False,
                comment=f"Proposal not found in pending: {proposal_change[:50]}",
            )

        self._last_change_time = __import__("time").time()
        return ApprovalDecision(
            approved=True,
            reviewer=reviewer,
            comment=comment,
        )

    def manual_reject(
        self,
        proposal_change: str,
        reviewer: str = "",
        comment: str = "",
    ) -> ApprovalDecision:
        """人工拒绝"""
        self._pending.pop(proposal_change, None)
        return ApprovalDecision(
            approved=False,
            reviewer=reviewer,
            comment=comment,
        )

    def _find_rule(self, risk_level: OptRiskLevel) -> ApprovalRule | None:
        """查找审批规则"""
        for rule in self._rules:
            if rule.risk_level == risk_level:
                return rule
        return None

    def get_pending(self) -> list[OptimizationProposal]:
        """获取待审批提案"""
        return list(self._pending.values())
