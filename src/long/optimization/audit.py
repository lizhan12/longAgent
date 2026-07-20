"""审计日志

不可变审计日志，记录所有优化提案、审批、应用和回滚。
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from .base import OptimizationProposal, ProposalRecord, ProposalStatus


class AuditLog:
    """审计日志

    不可变追加重录，记录优化系统中的所有事件。

    Attributes:
        _records: 审计记录
    """

    def __init__(self) -> None:
        self._records: list[ProposalRecord] = []

    def log_proposal(self, proposal: OptimizationProposal) -> ProposalRecord:
        """记录提案

        Args:
            proposal: 优化提案

        Returns:
            提案记录
        """
        record = ProposalRecord(
            proposal=proposal,
            status=ProposalStatus.PROPOSED,
            proposed_at=time.time(),
        )
        self._records.append(record)
        return record

    def log_approval(
        self,
        proposal_id: str,
        decision: Any,
    ) -> None:
        """记录审批"""
        record = self._find_record(proposal_id)
        if record:
            record.status = ProposalStatus.APPROVED
            record.reviewed_at = time.time()
            record.reviewer = getattr(decision, "reviewer", None)
            record.review_comment = getattr(decision, "comment", None)

    def log_rejection(
        self,
        proposal_id: str,
        decision: Any,
    ) -> None:
        """记录拒绝"""
        record = self._find_record(proposal_id)
        if record:
            record.status = ProposalStatus.REJECTED
            record.reviewed_at = time.time()
            record.reviewer = getattr(decision, "reviewer", None)
            record.review_comment = getattr(decision, "comment", None)

    def log_application(
        self,
        proposal: OptimizationProposal,
        metrics_after: dict[str, float],
    ) -> None:
        """记录应用"""
        record = self._find_latest_by_proposal(proposal)
        if record:
            record.status = ProposalStatus.APPLIED
            record.applied_at = time.time()
            record.metrics_after = metrics_after

    def log_rollback(
        self,
        proposal: OptimizationProposal,
        reason: str,
    ) -> None:
        """记录回滚"""
        record = self._find_latest_by_proposal(proposal)
        if record:
            record.status = ProposalStatus.ROLLED_BACK
            record.rollback_reason = reason

    def get_records(
        self,
        status: ProposalStatus | None = None,
    ) -> list[ProposalRecord]:
        """获取审计记录

        Args:
            status: 可选状态过滤

        Returns:
            审计记录列表
        """
        if status is None:
            return list(self._records)
        return [r for r in self._records if r.status == status]

    def _find_record(self, proposal_id: str) -> ProposalRecord | None:
        """查找提案记录"""
        # proposal_id 暂时通过 change 内容匹配
        for record in self._records:
            if record.proposal.change == proposal_id:
                return record
        return None

    def _find_latest_by_proposal(
        self,
        proposal: OptimizationProposal,
    ) -> ProposalRecord | None:
        """查找提案的最新记录"""
        for record in reversed(self._records):
            if record.proposal.change == proposal.change:
                return record
        return None
