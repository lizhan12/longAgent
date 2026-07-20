"""Optimization 模块测试

覆盖指标收集、模式分析、审批门、变更应用、审计日志、优化器和优化目标。
"""

import time

import pytest

from long.optimization.analyzer import PatternAnalyzer
from long.optimization.applier import ChangeApplier
from long.optimization.approval import ApprovalDecision, HumanApprovalGate
from long.optimization.audit import AuditLog
from long.optimization.base import (
    OptRiskLevel,
    OptimizationProposal,
    OptimizationTarget,
    ProposalStatus,
)
from long.optimization.collector import MetricPoint, MetricsCollector
from long.optimization.optimizer import AutoOptimizer
from long.optimization.targets.budget_tuner import BudgetTuner
from long.optimization.targets.prompt_tuner import PromptTuner
from long.optimization.targets.routing_tuner import RoutingTuner
from long.optimization.targets.tool_tuner import ToolTuner


# ========================
# MetricsCollector 测试
# ========================


class TestMetricsCollector:
    """指标收集器测试"""

    @pytest.fixture
    def collector(self):
        return MetricsCollector()

    def test_record_metric(self, collector):
        collector.record("test.metric", 1.0)
        points = collector.get_metrics("test.metric")
        assert len(points) == 1
        assert points[0].value == 1.0

    def test_record_with_tags(self, collector):
        collector.record("eval.score", 0.8, {"task": "search", "category": "normal"})
        points = collector.get_metrics("eval.score")
        assert points[0].tags["task"] == "search"

    def test_get_metrics_since(self, collector):
        collector.record("test", 1.0)
        time.sleep(0.01)
        now = time.time()
        collector.record("test", 2.0)

        points = collector.get_metrics("test", since=now)
        assert len(points) == 1
        assert points[0].value == 2.0

    def test_aggregation(self, collector):
        for i in range(10):
            collector.record("test", float(i))

        agg = collector.get_aggregation("test")
        assert agg["count"] == 10
        assert agg["min"] == 0.0
        assert agg["max"] == 9.0
        assert agg["mean"] == 4.5

    def test_aggregation_empty(self, collector):
        agg = collector.get_aggregation("nonexistent")
        assert agg["count"] == 0

    def test_record_eval_result(self, collector):
        collector.record_eval_result("task1", 0.8, "normal")
        points = collector.get_metrics("eval.score")
        assert len(points) == 1
        assert points[0].value == 0.8

    def test_record_execution_metrics(self, collector):
        collector.record_execution_metrics(step_count=5, duration=10.0, success=True)
        assert len(collector.get_metrics("execution.steps")) == 1
        assert len(collector.get_metrics("execution.duration")) == 1
        assert len(collector.get_metrics("execution.success")) == 1

    def test_clear(self, collector):
        collector.record("test", 1.0)
        collector.clear()
        assert len(collector.get_metrics("test")) == 0

    def test_get_all_metric_names(self, collector):
        collector.record("a", 1.0)
        collector.record("b", 2.0)
        names = collector.get_all_metric_names()
        assert "a" in names
        assert "b" in names


# ========================
# PatternAnalyzer 测试
# ========================


class TestPatternAnalyzer:
    """模式分析器测试"""

    @pytest.fixture
    def collector_with_data(self):
        collector = MetricsCollector()
        # 记录低成功率数据
        for _ in range(10):
            collector.record("execution.success", 0.0)
        # 记录高步骤数
        for _ in range(10):
            collector.record("execution.steps", 12.0)
        # 记录低评估分数
        for _ in range(10):
            collector.record("eval.score", 0.3)
        return collector

    def test_analyze_low_success_rate(self, collector_with_data):
        analyzer = PatternAnalyzer(collector_with_data)
        proposals = analyzer.analyze()
        targets = [p.target for p in proposals]
        assert OptimizationTarget.PROMPT in targets

    def test_analyze_high_steps(self, collector_with_data):
        analyzer = PatternAnalyzer(collector_with_data)
        proposals = analyzer.analyze()
        routing_proposals = [p for p in proposals if p.target == OptimizationTarget.ROUTING]
        assert len(routing_proposals) > 0

    def test_analyze_low_eval(self, collector_with_data):
        analyzer = PatternAnalyzer(collector_with_data)
        proposals = analyzer.analyze()
        tool_proposals = [p for p in proposals if p.target == OptimizationTarget.TOOL]
        assert len(tool_proposals) > 0

    def test_analyze_insufficient_data(self):
        collector = MetricsCollector()
        collector.record("execution.success", 1.0)
        analyzer = PatternAnalyzer(collector)
        proposals = analyzer.analyze()
        assert len(proposals) == 0

    def test_identify_weak_categories(self, collector_with_data):
        analyzer = PatternAnalyzer(collector_with_data)
        # 添加分类数据
        for _ in range(5):
            collector_with_data.record("eval.score", 0.2, {"category": "adversarial"})
        weak = analyzer.identify_weak_categories()
        assert len(weak) > 0


# ========================
# HumanApprovalGate 测试
# ========================


class TestHumanApprovalGate:
    """人工审批门测试"""

    @pytest.fixture
    def gate(self):
        return HumanApprovalGate()

    def test_low_risk_needs_approval(self, gate):
        proposal = OptimizationProposal(
            target=OptimizationTarget.PROMPT,
            change="test change",
            risk_level=OptRiskLevel.LOW,
        )
        decision = gate.review(proposal)
        assert decision.approved is False  # 默认都需要审批

    def test_critical_always_rejected(self, gate):
        proposal = OptimizationProposal(
            target=OptimizationTarget.PROMPT,
            change="critical change",
            risk_level=OptRiskLevel.CRITICAL,
        )
        decision = gate.review(proposal)
        assert decision.approved is False

    def test_safety_target_rejected(self, gate):
        """安全阈值目标永远不自动批准"""
        proposal = OptimizationProposal(
            target=OptimizationTarget.PROMPT,
            change="change state machine",
            risk_level=OptRiskLevel.LOW,
        )
        decision = gate.review(proposal)
        # 非安全目标也需人工审批
        assert decision.approved is False
        assert isinstance(decision, ApprovalDecision)

    def test_manual_approve(self, gate):
        proposal = OptimizationProposal(
            target=OptimizationTarget.PROMPT,
            change="pending change",
            risk_level=OptRiskLevel.LOW,
        )
        gate.review(proposal)
        decision = gate.manual_approve("pending change", reviewer="human", comment="ok")
        assert decision.approved is True
        assert decision.reviewer == "human"

    def test_manual_reject(self, gate):
        proposal = OptimizationProposal(
            target=OptimizationTarget.PROMPT,
            change="to reject",
            risk_level=OptRiskLevel.LOW,
        )
        gate.review(proposal)
        decision = gate.manual_reject("to reject", reviewer="human", comment="bad")
        assert decision.approved is False

    def test_frequency_limit(self, gate):
        gate._change_frequency_limit = 3600
        gate._last_change_time = time.time()
        proposal = OptimizationProposal(
            target=OptimizationTarget.PROMPT,
            change="too frequent",
            risk_level=OptRiskLevel.LOW,
        )
        decision = gate.review(proposal)
        assert decision.approved is False
        assert "频率" in decision.comment or "frequency" in decision.comment.lower()

    def test_get_pending(self, gate):
        proposal = OptimizationProposal(
            target=OptimizationTarget.PROMPT,
            change="pending",
            risk_level=OptRiskLevel.MEDIUM,
        )
        gate.review(proposal)
        pending = gate.get_pending()
        assert len(pending) >= 1


# ========================
# ChangeApplier 测试
# ========================


class TestChangeApplier:
    """变更应用器测试"""

    @pytest.fixture
    def applier(self):
        return ChangeApplier()

    def test_apply_change(self, applier):
        proposal = OptimizationProposal(
            target=OptimizationTarget.PROMPT,
            change="apply this",
            metrics_before={"score": 0.5},
        )
        result = applier.apply(proposal)
        assert result["success"] is True

    def test_rollback(self, applier):
        proposal = OptimizationProposal(
            target=OptimizationTarget.PROMPT,
            change="rollback test",
            metrics_before={"score": 0.5},
        )
        applier.apply(proposal)
        result = applier.rollback(proposal)
        assert result is True

    def test_rollback_no_snapshot(self, applier):
        proposal = OptimizationProposal(
            target=OptimizationTarget.PROMPT,
            change="no snapshot",
        )
        result = applier.rollback(proposal)
        assert result is False

    def test_detect_regression_score(self, applier):
        proposal = OptimizationProposal(
            target=OptimizationTarget.PROMPT,
            change="regression test",
            metrics_before={"success_rate": 0.8, "avg_eval_score": 0.7},
        )
        # 成功率大幅下降
        current = {"success_rate": 0.5, "avg_eval_score": 0.7}
        assert applier.detect_regression(proposal, current) is True

    def test_no_regression(self, applier):
        proposal = OptimizationProposal(
            target=OptimizationTarget.PROMPT,
            change="no regression",
            metrics_before={"success_rate": 0.8},
        )
        current = {"success_rate": 0.85}
        assert applier.detect_regression(proposal, current) is False

    def test_get_applied_changes(self, applier):
        proposal = OptimizationProposal(
            target=OptimizationTarget.PROMPT,
            change="tracked change",
        )
        applier.apply(proposal)
        changes = applier.get_applied_changes()
        assert "tracked change" in changes


# ========================
# AuditLog 测试
# ========================


class TestAuditLog:
    """审计日志测试"""

    @pytest.fixture
    def audit(self):
        return AuditLog()

    def test_log_proposal(self, audit):
        proposal = OptimizationProposal(
            target=OptimizationTarget.PROMPT,
            change="test proposal",
        )
        record = audit.log_proposal(proposal)
        assert record.status == ProposalStatus.PROPOSED

    def test_log_approval(self, audit):
        proposal = OptimizationProposal(
            target=OptimizationTarget.PROMPT,
            change="approved proposal",
        )
        record = audit.log_proposal(proposal)
        decision = ApprovalDecision(approved=True, reviewer="human")
        audit.log_approval("approved proposal", decision)
        records = audit.get_records(status=ProposalStatus.APPROVED)
        assert len(records) == 1

    def test_log_rejection(self, audit):
        proposal = OptimizationProposal(
            target=OptimizationTarget.PROMPT,
            change="rejected proposal",
        )
        record = audit.log_proposal(proposal)
        decision = ApprovalDecision(approved=False, comment="bad")
        audit.log_rejection("rejected proposal", decision)
        records = audit.get_records(status=ProposalStatus.REJECTED)
        assert len(records) == 1

    def test_get_all_records(self, audit):
        for i in range(3):
            audit.log_proposal(OptimizationProposal(
                target=OptimizationTarget.PROMPT,
                change=f"proposal_{i}",
            ))
        assert len(audit.get_records()) == 3


# ========================
# AutoOptimizer 测试
# ========================


class TestAutoOptimizer:
    """自动优化器测试"""

    @pytest.fixture
    def optimizer(self):
        return AutoOptimizer()

    def test_observe(self, optimizer):
        optimizer.collector.record("test", 1.0)
        snapshot = optimizer.observe()
        assert "test" in snapshot

    def test_orient_no_data(self, optimizer):
        snapshot = optimizer.observe()
        proposals = optimizer.orient(snapshot)
        assert len(proposals) == 0

    def test_orient_with_data(self):
        collector = MetricsCollector()
        for _ in range(10):
            collector.record("execution.success", 0.0)
        analyzer = PatternAnalyzer(collector)
        optimizer = AutoOptimizer(collector=collector, analyzer=analyzer)
        snapshot = optimizer.observe()
        proposals = optimizer.orient(snapshot)
        assert len(proposals) > 0

    def test_safety_targets_not_proposed(self):
        """安全阈值目标永不被建议变更"""
        collector = MetricsCollector()
        # 创建一个会产生建议的情景
        for _ in range(10):
            collector.record("execution.success", 0.0)
        optimizer = AutoOptimizer(collector=collector)
        snapshot = optimizer.observe()
        proposals = optimizer.orient(snapshot)
        for p in proposals:
            assert p.target.value not in {"state_machine", "ltl_rules", "security_policy"}

    def test_run_cycle_no_data(self, optimizer):
        result = optimizer.run_cycle()
        assert result["status"] == "no_proposals"

    def test_audit_log_accessible(self, optimizer):
        assert optimizer.audit_log is not None


# ========================
# Optimization Targets 测试
# ========================


class TestPromptTuner:
    """Prompt 优化器测试"""

    @pytest.fixture
    def tuner(self):
        return PromptTuner()

    def test_low_score_prompt(self, tuner):
        for _ in range(10):
            tuner.record_prompt_score("v1", 0.3)
        proposals = tuner.analyze()
        assert len(proposals) > 0

    def test_good_score_prompt(self, tuner):
        for _ in range(10):
            tuner.record_prompt_score("v1", 0.9)
        proposals = tuner.analyze()
        assert len(proposals) == 0

    def test_insufficient_samples(self, tuner):
        for _ in range(3):
            tuner.record_prompt_score("v1", 0.3)
        proposals = tuner.analyze()
        assert len(proposals) == 0


class TestRoutingTuner:
    """路由优化器测试"""

    @pytest.fixture
    def tuner(self):
        return RoutingTuner()

    def test_low_success_route(self, tuner):
        for _ in range(10):
            tuner.record_route("route_a", success=False, duration=5.0)
        proposals = tuner.analyze()
        assert len(proposals) > 0

    def test_good_route(self, tuner):
        for _ in range(10):
            tuner.record_route("route_a", success=True, duration=2.0)
        proposals = tuner.analyze()
        assert len(proposals) == 0


class TestBudgetTuner:
    """预算优化器测试"""

    @pytest.fixture
    def tuner(self):
        return BudgetTuner()

    def test_high_utilization(self, tuner):
        for _ in range(10):
            tuner.record_usage(allocated=100, used=95, exceeded=False)
        proposals = tuner.analyze()
        assert len(proposals) > 0

    def test_low_utilization(self, tuner):
        for _ in range(10):
            tuner.record_usage(allocated=100, used=20, exceeded=False)
        proposals = tuner.analyze()
        assert len(proposals) > 0

    def test_normal_utilization(self, tuner):
        for _ in range(10):
            tuner.record_usage(allocated=100, used=60, exceeded=False)
        proposals = tuner.analyze()
        assert len(proposals) == 0


class TestToolTuner:
    """工具优化器测试"""

    @pytest.fixture
    def tuner(self):
        return ToolTuner()

    def test_low_success_tool(self, tuner):
        for _ in range(10):
            tuner.record_tool_use("bad_tool", success=False, duration=5.0)
        proposals = tuner.analyze()
        assert len(proposals) > 0

    def test_slow_tool(self, tuner):
        for _ in range(10):
            tuner.record_tool_use("slow_tool", success=True, duration=15.0)
        proposals = tuner.analyze()
        assert len(proposals) > 0

    def test_good_tool(self, tuner):
        for _ in range(10):
            tuner.record_tool_use("good_tool", success=True, duration=2.0)
        proposals = tuner.analyze()
        assert len(proposals) == 0
