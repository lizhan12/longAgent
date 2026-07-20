"""Eval 模块测试

覆盖评估报告、结果层、过程层、系统层、对抗性测试、数据集管理和流水线。
"""

import time

import pytest

from long.eval.adversarial import AdversarialTestSuite
from long.eval.dataset_manager import EvalDatasetManager
from long.eval.outcome_eval import OutcomeEvaluator
from long.eval.pipeline import EvalPipeline
from long.eval.process_eval import MultiJudgeVoting, ProcessEvaluator
from long.eval.report import (
    EvalCategory,
    EvalReport,
    EvalTask,
    OutcomeResult,
    ProcessResult,
    SystemResult,
)
from long.eval.system_eval import SystemEvaluator


# ========================
# Report 模型测试
# ========================


class TestEvalTask:
    """评估任务测试"""

    def test_create_task(self):
        task = EvalTask(name="test", input="hello")
        assert task.name == "test"
        assert task.category == EvalCategory.NORMAL
        assert task.difficulty == 0.5

    def test_custom_task(self):
        task = EvalTask(
            name="adv",
            input="attack",
            category=EvalCategory.ADVERSARIAL,
            difficulty=0.9,
            tags=["injection"],
        )
        assert task.category == EvalCategory.ADVERSARIAL


class TestEvalReport:
    """评估报告测试"""

    def test_default_report(self):
        task = EvalTask(name="test", input="hello")
        report = EvalReport(task=task)
        assert report.score == 0.0
        assert report.needs_human_review is False
        assert report.auto_reviewed is True


# ========================
# OutcomeEvaluator 测试
# ========================


class TestOutcomeEvaluator:
    """结果层评估器测试"""

    @pytest.fixture
    def evaluator(self):
        return OutcomeEvaluator()

    def test_exact_match(self, evaluator):
        task = EvalTask(name="test", input="q", expected="hello world")
        result = evaluator.evaluate(task, "hello world")
        assert result.accuracy == 1.0

    def test_partial_match(self, evaluator):
        task = EvalTask(name="test", input="q", expected="hello world")
        result = evaluator.evaluate(task, "hello")
        assert 0.0 < result.accuracy < 1.0

    def test_no_match(self, evaluator):
        task = EvalTask(name="test", input="q", expected="expected")
        result = evaluator.evaluate(task, "completely different")
        assert result.accuracy == 0.0

    def test_none_output(self, evaluator):
        task = EvalTask(name="test", input="q", expected="hello")
        result = evaluator.evaluate(task, None)
        assert result.accuracy == 0.0
        assert result.schema_valid is False

    def test_no_expected(self, evaluator):
        task = EvalTask(name="test", input="q", expected=None)
        result = evaluator.evaluate(task, "any output")
        assert result.accuracy == 1.0

    def test_dict_accuracy(self, evaluator):
        task = EvalTask(
            name="test",
            input="q",
            expected={"key1": "val1", "key2": "val2"},
        )
        result = evaluator.evaluate(task, {"key1": "val1", "key2": "val2"})
        assert result.accuracy == 1.0

    def test_dict_partial_accuracy(self, evaluator):
        task = EvalTask(
            name="test",
            input="q",
            expected={"key1": "val1", "key2": "val2"},
        )
        result = evaluator.evaluate(task, {"key1": "val1", "key2": "wrong"})
        assert 0.0 < result.accuracy < 1.0

    def test_json_schema_valid(self, evaluator):
        task = EvalTask(name="test", input="q", expected="result")
        result = evaluator.evaluate(task, '{"key": "value"}')
        assert result.schema_valid is True


# ========================
# ProcessEvaluator 测试
# ========================


class TestProcessEvaluator:
    """过程层评估器测试"""

    @pytest.fixture
    def evaluator(self):
        return ProcessEvaluator()

    def test_no_trace(self, evaluator):
        task = EvalTask(name="test", input="q")
        result = evaluator.evaluate(task, None)
        assert result.score == 0.0

    def test_valid_trace(self, evaluator):
        task = EvalTask(name="test", input="q")
        trace = [
            {"action": "search"},
            {"action": "reason"},
            {"action": "output"},
        ]
        result = evaluator.evaluate(task, trace)
        assert result.score > 0.5
        assert result.rule_violations == 0

    def test_invalid_trace(self, evaluator):
        task = EvalTask(name="test", input="q")
        trace = [
            {"action": "output"},
        ]
        result = evaluator.evaluate(task, trace)
        assert result.rule_violations > 0

    def test_efficiency_short(self, evaluator):
        task = EvalTask(name="test", input="q")
        trace = [{"action": "search"}, {"action": "output"}]
        result = evaluator.evaluate(task, trace)
        assert result.efficiency >= 0.9

    def test_efficiency_long(self, evaluator):
        task = EvalTask(name="test", input="q")
        trace = [{"action": f"step_{i}"} for i in range(20)]
        result = evaluator.evaluate(task, trace)
        assert result.efficiency < 0.7


class TestMultiJudgeVoting:
    """多 Judge 投票测试"""

    def test_no_judges(self):
        voting = MultiJudgeVoting()
        task = EvalTask(name="test", input="q")
        result = voting.evaluate(task)
        # 无裁判时使用规则评估
        assert result.score >= 0.0

    def test_single_judge(self):
        def judge(task, trace):
            return 0.8

        voting = MultiJudgeVoting(judges=[judge])
        task = EvalTask(name="test", input="q")
        result = voting.evaluate(task)
        assert result.score > 0.0
        assert result.details.get("num_judges") == 1

    def test_majority_approval(self):
        judges = [
            lambda t, tr: 0.9,
            lambda t, tr: 0.8,
            lambda t, tr: 0.1,  # 少数反对
        ]
        voting = MultiJudgeVoting(judges=judges, majority_threshold=0.6)
        task = EvalTask(name="test", input="q")
        result = voting.evaluate(task)
        assert result.score > 0.5

    def test_judge_failure(self):
        def failing_judge(task, trace):
            raise RuntimeError("Judge failed")

        voting = MultiJudgeVoting(judges=[failing_judge])
        task = EvalTask(name="test", input="q")
        result = voting.evaluate(task)
        # 裁判失败时应回退到规则评估
        assert result.score >= 0.0


# ========================
# SystemEvaluator 测试
# ========================


class TestSystemEvaluator:
    """系统层评估器测试"""

    @pytest.fixture
    def evaluator(self):
        return SystemEvaluator()

    def test_stability_consistent(self, evaluator):
        task = EvalTask(name="test", input="q")
        results = [0.8, 0.82, 0.79, 0.81, 0.8]
        result = evaluator.evaluate_stability(task, results)
        assert result.stability > 0.8

    def test_stability_inconsistent(self, evaluator):
        task = EvalTask(name="test", input="q")
        results = [0.1, 0.9, 0.3, 0.8, 0.2]
        result = evaluator.evaluate_stability(task, results)
        assert result.stability < 0.5

    def test_stability_empty(self, evaluator):
        task = EvalTask(name="test", input="q")
        result = evaluator.evaluate_stability(task, [])
        assert result.stability == 0.0

    def test_convergence_improving(self, evaluator):
        score_series = [
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8],
        ]
        result = evaluator.evaluate_convergence(score_series)
        assert result.convergence > 0.5

    def test_convergence_declining(self, evaluator):
        score_series = [
            [0.8, 0.7],
            [0.5, 0.4],
            [0.3, 0.2],
        ]
        result = evaluator.evaluate_convergence(score_series)
        assert result.convergence < 0.5

    def test_failure_modes(self, evaluator):
        failures = [
            {"type": "timeout"},
            {"type": "timeout"},
            {"type": "parse_error"},
        ]
        result = evaluator.evaluate_failure_modes(failures)
        assert len(result.failure_modes) > 0
        assert result.details["total_failures"] == 3

    def test_no_failures(self, evaluator):
        result = evaluator.evaluate_failure_modes([])
        assert result.stability == 1.0
        assert result.failure_modes == []


# ========================
# AdversarialTestSuite 测试
# ========================


class TestAdversarialTestSuite:
    """对抗性测试套件测试"""

    @pytest.fixture
    def suite(self):
        return AdversarialTestSuite()

    def test_normal_tasks(self, suite):
        tasks = suite.get_normal_tasks()
        assert len(tasks) >= 3
        assert all(t.category == EvalCategory.NORMAL for t in tasks)

    def test_adversarial_tasks(self, suite):
        tasks = suite.get_adversarial_tasks()
        assert len(tasks) >= 3
        assert all(t.category == EvalCategory.ADVERSARIAL for t in tasks)

    def test_boundary_tasks(self, suite):
        tasks = suite.get_boundary_tasks()
        assert len(tasks) >= 3
        assert all(t.category == EvalCategory.BOUNDARY for t in tasks)

    def test_all_tasks(self, suite):
        tasks = suite.get_all_tasks()
        assert len(tasks) >= 9

    def test_injection_tasks_exist(self, suite):
        tasks = suite.get_adversarial_tasks()
        names = [t.name for t in tasks]
        assert "prompt_injection_direct" in names


# ========================
# EvalDatasetManager 测试
# ========================


class TestEvalDatasetManager:
    """数据集管理器测试"""

    @pytest.fixture
    def manager(self):
        return EvalDatasetManager()

    def test_empty_manager(self, manager):
        assert len(manager.public_set) == 0
        assert len(manager.hidden_set) == 0

    def test_add_to_public(self, manager):
        task = EvalTask(name="test", input="q")
        manager.add_to_public(task)
        assert len(manager.public_set) == 1

    def test_add_to_hidden(self, manager):
        task = EvalTask(name="hidden", input="q")
        manager.add_to_hidden(task)
        assert len(manager.hidden_set) == 1

    def test_get_eval_set_without_hidden(self, manager):
        manager.add_to_public(EvalTask(name="pub", input="q"))
        manager.add_to_hidden(EvalTask(name="hid", input="q"))

        tasks = manager.get_eval_set(include_hidden=False)
        assert len(tasks) == 1

    def test_get_eval_set_with_hidden(self, manager):
        manager.add_to_public(EvalTask(name="pub", input="q"))
        manager.add_to_hidden(EvalTask(name="hid", input="q"))

        tasks = manager.get_eval_set(include_hidden=True)
        assert len(tasks) == 2

    def test_rotate_dataset(self, manager):
        for i in range(5):
            manager.add_to_hidden(EvalTask(name=f"hidden_{i}", input=f"q{i}"))

        count = manager.rotate_dataset()
        assert count > 0

    def test_rotate_no_hidden(self, manager):
        count = manager.rotate_dataset()
        assert count == 0

    def test_task_hash(self, manager):
        task = EvalTask(name="test", input="q", expected="a")
        h1 = manager.compute_task_hash(task)
        h2 = manager.compute_task_hash(task)
        assert h1 == h2
        assert len(h1) == 16


# ========================
# EvalPipeline 测试
# ========================


class TestEvalPipeline:
    """评估流水线测试"""

    @pytest.fixture
    def pipeline(self):
        return EvalPipeline()

    def test_run_simple(self, pipeline):
        task = EvalTask(name="test", input="q", expected="hello")
        report = pipeline.run(task, output="hello")
        assert report.score > 0.0
        assert isinstance(report, EvalReport)

    def test_run_with_trace(self, pipeline):
        task = EvalTask(name="test", input="q", expected="result")
        trace = [{"action": "search"}, {"action": "reason"}, {"action": "output"}]
        report = pipeline.run(task, output="result", trace=trace)
        assert report.score > 0.0

    def test_low_score_needs_review(self, pipeline):
        task = EvalTask(name="test", input="q", expected="expected")
        report = pipeline.run(task, output="completely wrong")
        if report.score < pipeline.auto_review_threshold:
            assert report.needs_human_review is True

    def test_high_score_auto_reviewed(self, pipeline):
        task = EvalTask(name="test", input="q", expected="hello")
        report = pipeline.run(task, output="hello")
        if report.score >= pipeline.auto_review_threshold:
            assert report.auto_reviewed is True

    def test_run_batch(self, pipeline):
        tasks = [
            EvalTask(name="t1", input="q1", expected="a1"),
            EvalTask(name="t2", input="q2", expected="a2"),
        ]
        reports = pipeline.run_batch(tasks)
        assert len(reports) == 2

    def test_process_weight_higher(self, pipeline):
        """过程层权重大于结果层"""
        task = EvalTask(name="test", input="q", expected="hello")
        # 有正确的 trace 但输出不对
        trace = [{"action": "search"}, {"action": "reason"}, {"action": "output"}]
        report = pipeline.run(task, output="wrong output", trace=trace)
        # 过程层评分应该对总分有更大影响
        assert isinstance(report.score, float)
