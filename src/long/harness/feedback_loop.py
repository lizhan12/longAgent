"""数据飞轮 — 评估结果 → Prompt 改进建议 → 人工确认

Harness Engineering 原则：反馈回路（Feedback Loop）
从"评估止于报告"升级到"评估回流到系统优化"：
- 评估结果自动生成 Prompt 改进建议
- 改进建议需要人工确认（不自动修改）
- 跟踪改进历史，支持回退

设计约束：
- 不碰模型权重（只优化 Prompt 和测试用例）
- 人工审批作为安全阀（防止自动退化）
- 改进历史记录到文件，可追溯
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ProposalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    REVERTED = "reverted"


@dataclass
class ImprovementProposal:
    proposal_id: str = ""
    created_at: float = field(default_factory=time.time)
    status: ProposalStatus = ProposalStatus.PENDING
    category: str = ""
    description: str = ""
    suggested_change: str = ""
    reason: str = ""
    eval_scores: dict[str, float] = field(default_factory=dict)
    applied_at: float = 0.0
    reverted_at: float = 0.0
    revert_reason: str = ""


class FeedbackLoop:
    """数据飞轮 — 闭环反馈优化

    用法：
        loop = FeedbackLoop(workspace_dir)
        proposal = loop.create_proposal("prompt", "效率", "优化搜索引导语", scores={"efficiency": 0.6})
        loop.approve(proposal.proposal_id)
        # 用户手动修改 AGENTS.md 后
        loop.mark_applied(proposal.proposal_id)
    """

    def __init__(self, workspace_dir: str | Path) -> None:
        self._dir = Path(workspace_dir) / "feedback"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._proposals: dict[str, ImprovementProposal] = {}
        self._load()

    def _proposals_path(self) -> Path:
        return self._dir / "proposals.json"

    def _history_path(self) -> Path:
        return self._dir / "history.json"

    def _load(self) -> None:
        path = self._proposals_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for item in data:
                    proposal = ImprovementProposal(**item)
                    self._proposals[proposal.proposal_id] = proposal
            except (json.JSONDecodeError, TypeError):
                pass

    def _save(self) -> None:
        data = [p.__dict__ for p in self._proposals.values()]
        self._proposals_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def create_proposal(
        self,
        category: str,
        description: str,
        suggested_change: str,
        *,
        reason: str = "",
        eval_scores: dict[str, float] | None = None,
    ) -> ImprovementProposal:
        """创建改进提案"""
        import uuid

        proposal = ImprovementProposal(
            proposal_id=uuid.uuid4().hex[:8],
            category=category,
            description=description,
            suggested_change=suggested_change,
            reason=reason,
            eval_scores=eval_scores or {},
        )
        self._proposals[proposal.proposal_id] = proposal
        self._save()

        logger.info(
            "改进提案已创建: %s [%s] %s", proposal.proposal_id, category, description[:100],
        )
        return proposal

    def approve(self, proposal_id: str) -> ImprovementProposal | None:
        """审批通过"""
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return None
        proposal.status = ProposalStatus.APPROVED
        self._save()
        logger.info("改进提案已审批: %s", proposal_id)
        return proposal

    def reject(self, proposal_id: str, reason: str = "") -> ImprovementProposal | None:
        """驳回"""
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return None
        proposal.status = ProposalStatus.REJECTED
        proposal.revert_reason = reason
        self._save()
        logger.info("改进提案已驳回: %s → %s", proposal_id, reason[:100])
        return proposal

    def mark_applied(self, proposal_id: str) -> ImprovementProposal | None:
        """标记为已应用"""
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return None
        proposal.status = ProposalStatus.APPLIED
        proposal.applied_at = time.time()
        self._save()
        logger.info("改进提案已应用: %s", proposal_id)
        return proposal

    def revert(self, proposal_id: str, reason: str = "") -> ImprovementProposal | None:
        """回退"""
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return None
        proposal.status = ProposalStatus.REVERTED
        proposal.reverted_at = time.time()
        proposal.revert_reason = reason
        self._save()
        logger.warning("改进提案已回退: %s → %s", proposal_id, reason[:100])
        return proposal

    def list_pending(self) -> list[ImprovementProposal]:
        return [p for p in self._proposals.values() if p.status == ProposalStatus.PENDING]

    def list_by_category(self, category: str) -> list[ImprovementProposal]:
        return [p for p in self._proposals.values() if p.category == category]

    def get(self, proposal_id: str) -> ImprovementProposal | None:
        return self._proposals.get(proposal_id)

    def generate_from_eval(
        self,
        eval_summary: dict[str, Any],
    ) -> list[ImprovementProposal]:
        """从评估结果自动生成改进提案

        评估维度 → 提案类别映射：
        - efficiency < 0.7 → prompt 效率优化
        - accuracy < 0.7 → tool 准确度优化
        - safety < 0.7 → 安全规则增强
        """
        proposals: list[ImprovementProposal] = []

        scores = eval_summary.get("scores", {})

        efficiency = scores.get("efficiency", 1.0)
        if efficiency < 0.7:
            proposals.append(self.create_proposal(
                category="prompt",
                description=f"工具调用效率优化 (当前评分: {efficiency:.2f})",
                suggested_change="在 AGENTS.md 中增强工具调用规则，减少冗余调用",
                reason=f"评估显示效率评分 {efficiency:.2f}，低于 0.7 阈值",
                eval_scores=scores,
            ))

        accuracy = scores.get("accuracy", 1.0)
        if accuracy < 0.7:
            proposals.append(self.create_proposal(
                category="tool",
                description=f"工具准确度优化 (当前评分: {accuracy:.2f})",
                suggested_change="检查工具描述是否准确，添加边界条件说明",
                reason=f"评估显示准确度评分 {accuracy:.2f}，低于 0.7 阈值",
                eval_scores=scores,
            ))

        safety = scores.get("safety", 1.0)
        if safety < 0.7:
            proposals.append(self.create_proposal(
                category="security",
                description=f"安全评分优化 (当前评分: {safety:.2f})",
                suggested_change="在 security.yaml 中收紧安全规则，增加工具限制",
                reason=f"评估显示安全评分 {safety:.2f}，低于 0.7 阈值",
                eval_scores=scores,
            ))

        return proposals

    def _has_pending_for(self, category: str, keyword: str) -> bool:
        """检查是否已存在指定类别且描述包含关键字的待处理提案（用于去重）"""
        for p in self.list_pending():
            if p.category == category and keyword in p.description:
                return True
        return False

    def generate_from_tool_failure(
        self,
        tool_name: str,
        error_message: str,
        arguments: dict | None = None,
    ) -> ImprovementProposal | None:
        """从工具执行失败生成改进提案

        当工具执行失败时，自动创建 "tool" 类别的改进提案。
        仅在不存在同 tool_name 的待处理提案时创建（去重）。
        """
        if self._has_pending_for("tool", tool_name):
            logger.debug("工具失败提案已存在，跳过: %s", tool_name)
            return None

        return self.create_proposal(
            category="tool",
            description=f"工具 {tool_name} 执行失败",
            suggested_change=f"检查工具 {tool_name} 的描述和边界条件，添加错误处理规则",
            reason=f"工具 {tool_name} 执行失败: {error_message}",
            eval_scores={"arguments": arguments} if arguments else {},
        )

    def generate_from_constraint_violation(
        self,
        violation_type: str,
        details: str,
        step_id: str = "",
    ) -> ImprovementProposal | None:
        """从约束验证失败生成改进提案

        当约束验证失败时，自动创建 "security" 类别的改进提案。
        仅在不存在同 violation_type 的待处理提案时创建（去重）。
        """
        if self._has_pending_for("security", violation_type):
            logger.debug("约束违反提案已存在，跳过: %s", violation_type)
            return None

        return self.create_proposal(
            category="security",
            description=f"约束验证失败 [{violation_type}]",
            suggested_change=f"在安全规则中增加 {violation_type} 类型的约束检查",
            reason=f"约束验证失败 [{violation_type}]: {details}",
        )

    def generate_from_sandbox_failure(
        self,
        error_type: str,
        error_message: str,
        code_snippet: str = "",
    ) -> ImprovementProposal | None:
        """从沙箱执行失败生成改进提案

        当沙箱执行失败时，自动创建 "sandbox" 类别的改进提案。
        仅在不存在同 error_type 的待处理提案时创建（去重）。
        """
        if self._has_pending_for("sandbox", error_type):
            logger.debug("沙箱失败提案已存在，跳过: %s", error_type)
            return None

        return self.create_proposal(
            category="sandbox",
            description=f"沙箱执行失败 [{error_type}]",
            suggested_change=f"增强沙箱安全规则，拦截 {error_type} 类型的代码模式",
            reason=f"沙箱执行失败 [{error_type}]: {error_message}",
        )

    def generate_from_near_miss(
        self,
        check_type: str,
        value: float,
        threshold: float,
        context: str = "",
    ) -> ImprovementProposal | None:
        """从接近阈值的检查生成改进提案

        当检查通过但值接近阈值（20% 以内）时，创建 "near_miss" 类别的改进提案。
        """
        if threshold == 0:
            return None

        gap = abs(value - threshold) / abs(threshold)
        if gap >= 0.2:
            return None

        if self._has_pending_for("near_miss", check_type):
            logger.debug("接近阈值提案已存在，跳过: %s", check_type)
            return None

        return self.create_proposal(
            category="near_miss",
            description=f"接近阈值 [{check_type}]: 当前值 {value:.2f}, 阈值 {threshold:.2f}",
            suggested_change=f"收紧 {check_type} 的阈值或增加额外检查 (当前值 {value:.2f} 接近阈值 {threshold:.2f})",
            reason=f"接近阈值 [{check_type}]: 当前值 {value:.2f}, 阈值 {threshold:.2f}, 差距仅 {gap:.1%}",
        )

    def get_stats(self) -> dict[str, int]:
        stats: dict[str, int] = {}
        for p in self._proposals.values():
            key = p.status.value
            stats[key] = stats.get(key, 0) + 1
        return stats