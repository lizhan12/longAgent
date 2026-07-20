"""自动优化器

OODA 主循环: Observe → Orient → Decide → Act
支持事件驱动和定时触发的自动优化。
所有变更需人工确认（LOW 风险可自动审批）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .analyzer import PatternAnalyzer
from .applier import ChangeApplier
from .approval import HumanApprovalGate
from .audit import AuditLog
from .base import OptimizationProposal, OptRiskLevel
from .collector import MetricsCollector

logger = logging.getLogger(__name__)


class OODATriggerConfig:
    """OODA 触发配置"""

    def __init__(
        self,
        conversation_interval: int = 20,
        time_interval_seconds: float = 3600.0,
        failure_rate_threshold: float = 0.3,
        latency_increase_threshold: float = 0.5,
        budget_utilization_threshold: float = 0.9,
        auto_approve_low_risk: bool = True,
    ) -> None:
        self.conversation_interval = conversation_interval
        self.time_interval_seconds = time_interval_seconds
        self.failure_rate_threshold = failure_rate_threshold
        self.latency_increase_threshold = latency_increase_threshold
        self.budget_utilization_threshold = budget_utilization_threshold
        self.auto_approve_low_risk = auto_approve_low_risk


class AutoOptimizer:
    """自动优化器

    使用 OODA 主循环进行持续优化。
    支持事件驱动和定时触发。

    Attributes:
        _collector: 指标收集器
        _analyzer: 模式分析器
        _applier: 变更应用器
        _approval_gate: 人工审批门
        _audit_log: 审计日志
        _safety_targets: 安全阈值目标（永不被建议变更）
        _trigger_config: 触发配置
        _conversation_count: 对话计数器
        _last_cycle_time: 上次循环时间
        _cycle_history: 循环历史
    """

    def __init__(
        self,
        collector: MetricsCollector | None = None,
        analyzer: PatternAnalyzer | None = None,
        applier: ChangeApplier | None = None,
        approval_gate: HumanApprovalGate | None = None,
        audit_log: AuditLog | None = None,
        trigger_config: OODATriggerConfig | None = None,
    ) -> None:
        self._collector = collector or MetricsCollector()
        self._analyzer = analyzer or PatternAnalyzer(self._collector)
        self._applier = applier or ChangeApplier()
        self._approval_gate = approval_gate or HumanApprovalGate()
        self._audit_log = audit_log or AuditLog()
        self._trigger_config = trigger_config or OODATriggerConfig()
        self._safety_targets = {
            "state_machine",
            "ltl_rules",
            "security_policy",
        }

        self._conversation_count: int = 0
        self._last_cycle_time: float = time.time()
        self._cycle_history: list[dict[str, Any]] = []
        self._max_history: int = 100
        self._background_task: asyncio.Task | None = None

    @property
    def collector(self) -> MetricsCollector:
        return self._collector

    @property
    def audit_log(self) -> AuditLog:
        return self._audit_log

    @property
    def trigger_config(self) -> OODATriggerConfig:
        return self._trigger_config

    def on_conversation_complete(self) -> bool:
        """对话完成事件回调

        每次对话完成后调用，检查是否满足触发条件。

        Returns:
            是否触发了 OODA 循环
        """
        self._conversation_count += 1

        if self._conversation_count >= self._trigger_config.conversation_interval:
            self._conversation_count = 0
            logger.info("OODA 触发: 对话数达到 %d", self._trigger_config.conversation_interval)
            return self._check_and_run()

        if self._check_anomaly_conditions():
            logger.info("OODA 触发: 检测到异常条件")
            return self._check_and_run()

        return False

    def _check_anomaly_conditions(self) -> bool:
        """检查异常触发条件

        Returns:
            是否满足异常触发条件
        """
        success_agg = self._collector.get_aggregation("llm.success")
        if success_agg["count"] >= 5:
            failure_rate = 1.0 - success_agg["mean"]
            if failure_rate > self._trigger_config.failure_rate_threshold:
                logger.warning(
                    "LLM 失败率异常: %.1f%% (阈值 %.1f%%)",
                    failure_rate * 100,
                    self._trigger_config.failure_rate_threshold * 100,
                )
                return True

        latency_agg = self._collector.get_aggregation("llm.latency_ms")
        if latency_agg["count"] >= 5 and latency_agg["mean"] > 0:
            recent_agg = self._collector.get_aggregation(
                "llm.latency_ms",
                since=time.time() - 300,
            )
            if recent_agg["count"] >= 3:
                increase = (recent_agg["mean"] - latency_agg["mean"]) / max(latency_agg["mean"], 1)
                if increase > self._trigger_config.latency_increase_threshold:
                    logger.warning(
                        "LLM 延迟上升: %.1f%% (阈值 %.1f%%)",
                        increase * 100,
                        self._trigger_config.latency_increase_threshold * 100,
                    )
                    return True

        return False

    def _check_and_run(self) -> bool:
        """检查并运行 OODA 循环

        Returns:
            是否成功运行
        """
        try:
            result = self.run_cycle()
            self._last_cycle_time = time.time()

            self._cycle_history.append({
                "timestamp": time.time(),
                "status": result.get("status", "unknown"),
                "proposals_count": len(result.get("proposals", [])),
                "approved_count": len(result.get("approved", [])),
            })
            if len(self._cycle_history) > self._max_history:
                self._cycle_history = self._cycle_history[-self._max_history:]

            return result.get("status") == "completed"
        except Exception as e:
            logger.error("OODA 循环异常: %s", e)
            return False

    def start_background_loop(self) -> None:
        """启动后台定时 OODA 循环"""
        if self._background_task is not None:
            return

        async def _loop() -> None:
            while True:
                await asyncio.sleep(self._trigger_config.time_interval_seconds)

                elapsed = time.time() - self._last_cycle_time
                if elapsed >= self._trigger_config.time_interval_seconds:
                    logger.info("OODA 定时触发: 距上次 %.0f 秒", elapsed)
                    self._check_and_run()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._background_task = asyncio.ensure_future(_loop())
            else:
                self._background_task = loop.create_task(_loop())
        except RuntimeError:
            logger.debug("无法启动后台 OODA 循环（无事件循环）")

    def stop_background_loop(self) -> None:
        """停止后台 OODA 循环"""
        if self._background_task is not None:
            self._background_task.cancel()
            self._background_task = None

    def observe(self) -> dict[str, Any]:
        """Observe: 观察当前指标状态"""
        snapshot: dict[str, Any] = {}

        for name in self._collector.get_all_metric_names():
            agg = self._collector.get_aggregation(name)
            snapshot[name] = agg

        return snapshot

    def orient(self, snapshot: dict[str, Any]) -> list[OptimizationProposal]:
        """Orient: 分析指标模式，发现优化机会"""
        proposals = self._analyzer.analyze()

        safe_proposals = []
        for proposal in proposals:
            if proposal.target.value in self._safety_targets:
                logger.warning(
                    "Skipping proposal for safety target: %s",
                    proposal.target.value,
                )
                continue
            safe_proposals.append(proposal)

        return safe_proposals

    def decide(
        self,
        proposals: list[OptimizationProposal],
    ) -> list[OptimizationProposal]:
        """Decide: 决定是否执行优化建议

        LOW 风险提案可自动审批，其他需要人工审批。
        """
        approved = []

        for proposal in proposals:
            record = self._audit_log.log_proposal(proposal)

            if (self._trigger_config.auto_approve_low_risk
                    and proposal.risk_level == OptRiskLevel.LOW):
                logger.info(
                    "Proposal auto-approved (LOW risk): %s (target=%s)",
                    proposal.change[:50],
                    proposal.target.value,
                )
                self._audit_log.log_approval(record.proposal_id, None)
                approved.append(proposal)
                continue

            decision = self._approval_gate.review(proposal)

            if decision.approved:
                self._audit_log.log_approval(record.proposal_id, decision)
                approved.append(proposal)
                logger.info(
                    "Proposal approved: %s (target=%s, risk=%s)",
                    proposal.change[:50],
                    proposal.target.value,
                    proposal.risk_level.value,
                )
            else:
                self._audit_log.log_rejection(record.proposal_id, decision)
                logger.info(
                    "Proposal rejected: %s",
                    proposal.change[:50],
                )

        return approved

    def act(self, proposals: list[OptimizationProposal]) -> list[dict[str, Any]]:
        """Act: 执行已审批的变更"""
        results = []

        for proposal in proposals:
            try:
                result = self._applier.apply(proposal)
                results.append(result)

                if result.get("success"):
                    self._audit_log.log_application(
                        proposal,
                        result.get("metrics_after", {}),
                    )

                    if self._applier.detect_regression(
                        proposal,
                        self._get_current_metrics(),
                    ):
                        logger.warning("检测到回归，自动回滚: %s", proposal.change[:50])
                        self._applier.rollback(proposal)
                        self._audit_log.log_rollback(
                            proposal,
                            "自动回滚: 检测到性能回归",
                        )
                else:
                    self._applier.rollback(proposal)
                    self._audit_log.log_rollback(
                        proposal,
                        result.get("error", "Unknown error"),
                    )

            except Exception as e:
                logger.error("Error applying proposal: %s", e)
                try:
                    self._applier.rollback(proposal)
                except Exception:
                    pass
                results.append({"success": False, "error": str(e)})

        return results

    def _get_current_metrics(self) -> dict[str, float]:
        """获取当前关键指标"""
        metrics: dict[str, float] = {}
        for name in ("llm.success", "llm.latency_ms", "tool.success", "execution.success"):
            agg = self._collector.get_aggregation(name)
            if agg["count"] > 0:
                metrics[f"avg_{name}"] = agg["mean"]
        return metrics

    def run_cycle(self) -> dict[str, Any]:
        """执行一次完整的 OODA 循环"""
        snapshot = self.observe()

        proposals = self.orient(snapshot)

        if not proposals:
            return {
                "status": "no_proposals",
                "snapshot": snapshot,
                "proposals": [],
            }

        approved = self.decide(proposals)

        if not approved:
            return {
                "status": "no_approved",
                "snapshot": snapshot,
                "proposals": [p.change for p in proposals],
                "approved": [],
            }

        results = self.act(approved)

        return {
            "status": "completed",
            "snapshot": snapshot,
            "proposals": [p.change for p in proposals],
            "approved": [p.change for p in approved],
            "results": results,
        }

    def get_cycle_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """获取 OODA 循环历史"""
        return self._cycle_history[-limit:]

    def get_status(self) -> dict[str, Any]:
        """获取优化器状态"""
        return {
            "conversation_count": self._conversation_count,
            "last_cycle_time": self._last_cycle_time,
            "cycle_count": len(self._cycle_history),
            "applied_changes": self._applier.get_applied_changes(),
            "metric_names": self._collector.get_all_metric_names(),
            "background_running": self._background_task is not None and not self._background_task.done(),
        }
