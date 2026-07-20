"""Cognitive 模块测试

覆盖 TaskIR, SubtaskIR, TaskPlanner, StrategyCritique, PlanRepair,
SemanticCompressor, KeyInfoProtector, CognitiveContext, StateGraph, CognitiveRuntime
"""

import asyncio
import json
import time

import pytest

from long.cognitive.task_ir import TaskIR, SubtaskIR, parse_task_ir_from_message
from long.cognitive.planner import TaskPlanner, PlanResult
from long.cognitive.reflection import (
    StrategyCritique,
    PlanRepair,
    StrategyCritiqueResult,
    StrategyIssue,
)
from long.cognitive.compression import (
    SemanticCompressor,
    KeyInfoProtector,
    CompressionResult,
)
from long.cognitive.runtime import (
    CognitiveContext,
    CognitiveRuntime,
    ExecutionMode,
    NodeKind,
    Reflector,
    StateGraph,
    ToolRouter,
)
from long.capabilities.tool_capability import (
    ToolCapability,
    ToolCapabilityRegistry,
    ToolStats,
)


# ========================
# TaskIR / SubtaskIR 测试
# ========================


class TestSubtaskIR:
    def test_auto_id(self):
        s = SubtaskIR(description="test")
        assert s.id
        assert len(s.id) == 8

    def test_custom_id(self):
        s = SubtaskIR(id="custom", description="test")
        assert s.id == "custom"

    def test_default_status(self):
        s = SubtaskIR(description="test")
        assert s.status == "pending"

    def test_dependencies(self):
        s = SubtaskIR(description="test", depends_on=["s1", "s2"])
        assert len(s.depends_on) == 2


class TestTaskIR:
    def _make_task(self):
        return TaskIR(
            goal="临泉县天气预报",
            constraints=["使用中文"],
            deliverables=["天气预报", "折线图"],
            subtasks=[
                SubtaskIR(id="s1", description="搜索天气", tool_hint="tavily_search"),
                SubtaskIR(id="s2", description="生成图表", tool_hint="execute_code", depends_on=["s1"]),
                SubtaskIR(id="s3", description="写报告", tool_hint="write_file", depends_on=["s2"]),
            ],
        )

    def test_pending_subtasks(self):
        task = self._make_task()
        assert len(task.pending_subtasks()) == 3

    def test_next_executable_subtask(self):
        task = self._make_task()
        next_sub = task.next_executable_subtask()
        assert next_sub is not None
        assert next_sub.id == "s1"

    def test_mark_and_complete_subtask(self):
        task = self._make_task()
        task.mark_subtask_in_progress("s1")
        assert task.subtasks[0].status == "in_progress"

        task.complete_subtask("s1", "Found weather data")
        assert task.subtasks[0].status == "completed"
        assert task.subtasks[0].result_summary == "Found weather data"
        assert "s1" in task.completed_subtasks

    def test_next_after_complete(self):
        task = self._make_task()
        task.complete_subtask("s1")
        next_sub = task.next_executable_subtask()
        assert next_sub is not None
        assert next_sub.id == "s2"

    def test_fail_subtask(self):
        task = self._make_task()
        task.fail_subtask("s1")
        assert task.subtasks[0].status == "failed"

    def test_progress_ratio(self):
        task = self._make_task()
        assert task.progress_ratio() == 0.0
        task.complete_subtask("s1")
        assert abs(task.progress_ratio() - 1 / 3) < 0.01
        task.complete_subtask("s2")
        task.complete_subtask("s3")
        assert task.progress_ratio() == 1.0

    def test_is_all_complete(self):
        task = self._make_task()
        assert not task.is_all_complete()
        task.complete_subtask("s1")
        task.complete_subtask("s2")
        task.complete_subtask("s3")
        assert task.is_all_complete()

    def test_add_key_fact(self):
        task = self._make_task()
        task.add_key_fact("气温2°C")
        task.add_key_fact("降水15%")
        assert len(task.key_facts) == 2
        task.add_key_fact("气温2°C")
        assert len(task.key_facts) == 2

    def test_add_conclusion(self):
        task = self._make_task()
        task.add_conclusion("天气寒冷")
        assert len(task.intermediate_conclusions) == 1
        task.add_conclusion("天气寒冷")
        assert len(task.intermediate_conclusions) == 1

    def test_to_prompt_text(self):
        task = self._make_task()
        task.complete_subtask("s1", "Got data")
        text = task.to_prompt_text()
        assert "临泉县天气预报" in text
        assert "已完成" in text
        assert "待完成" in text


class TestParseTaskIR:
    def test_weather_with_chart(self):
        task = parse_task_ir_from_message("临泉县未来7天的预报，带有折线图，并生成报告")
        assert task.goal == "临泉县未来7天的预报，带有折线图，并生成报告"
        assert len(task.subtasks) >= 2
        tool_hints = [s.tool_hint for s in task.subtasks if s.tool_hint]
        assert "tavily_search" in tool_hints
        assert "图表/可视化" in task.deliverables
        assert "报告/文档" in task.deliverables

    def test_simple_question(self):
        task = parse_task_ir_from_message("什么是Python？")
        assert len(task.subtasks) >= 1

    def test_code_execution(self):
        task = parse_task_ir_from_message("写代码计算斐波那契数列并执行")
        tool_hints = [s.tool_hint for s in task.subtasks if s.tool_hint]
        assert "execute_code" in tool_hints

    def test_file_save(self):
        task = parse_task_ir_from_message("保存文件到output.txt")
        tool_hints = [s.tool_hint for s in task.subtasks if s.tool_hint]
        assert "write_file" in tool_hints

    def test_plain_greeting(self):
        task = parse_task_ir_from_message("你好")
        assert len(task.subtasks) >= 1

    def test_dependencies_chain(self):
        task = parse_task_ir_from_message("搜索天气，画折线图，生成报告")
        non_empty_deps = [s.depends_on for s in task.subtasks if s.depends_on]
        assert len(non_empty_deps) > 0


# ========================
# TaskPlanner 测试
# ========================


class TestTaskPlanner:
    def _make_context(self, **overrides):
        defaults = {
            "user_message": "test",
            "round_count": 1,
            "max_rounds": 8,
            "search_count": 0,
            "max_search_count": 3,
            "errors": [],
            "tool_history": [],
            "task_ir": None,
            "retry_count": 0,
            "max_retries": 3,
        }
        defaults.update(overrides)
        ctx = CognitiveContext(**defaults)
        return ctx

    def test_plan_no_task_ir(self):
        planner = TaskPlanner()
        ctx = self._make_context()
        result = planner.plan(ctx)
        assert result.should_continue

    def test_plan_all_complete(self):
        planner = TaskPlanner()
        task = TaskIR(
            goal="done",
            subtasks=[
                SubtaskIR(id="s1", description="done", status="completed"),
            ],
            completed_subtasks=["s1"],
        )
        ctx = self._make_context(task_ir=task)
        result = planner.plan(ctx)
        assert result.is_complete

    def test_plan_rounds_exhausted(self):
        planner = TaskPlanner()
        ctx = self._make_context(round_count=10, max_rounds=8)
        result = planner.plan(ctx)
        assert result.is_complete

    def test_plan_with_pending_subtask(self):
        planner = TaskPlanner()
        task = TaskIR(
            goal="test",
            subtasks=[
                SubtaskIR(id="s1", description="search", tool_hint="tavily_search"),
            ],
        )
        ctx = self._make_context(task_ir=task)
        result = planner.plan(ctx)
        assert result.should_continue
        assert result.next_subtask is not None

    def test_plan_with_errors(self):
        planner = TaskPlanner()
        ctx = self._make_context(errors=["some error"], retry_count=1, max_retries=3)
        result = planner.plan(ctx)
        assert result.should_continue

    def test_plan_search_exhausted_with_code(self):
        planner = TaskPlanner()
        task = TaskIR(
            goal="test",
            subtasks=[
                SubtaskIR(id="s1", description="search", status="completed"),
                SubtaskIR(id="s2", description="chart", tool_hint="execute_code"),
            ],
            completed_subtasks=["s1"],
        )
        ctx = self._make_context(
            task_ir=task,
            search_count=3,
            max_search_count=3,
            tool_history=[{"name": "execute_code", "result": "ok"}],
        )
        result = planner.plan(ctx)
        assert result.should_continue


# ========================
# StrategyCritique 测试
# ========================


class TestStrategyCritique:
    def test_no_issues(self):
        critique = StrategyCritique()
        ctx = CognitiveContext(user_message="test", tool_history=[])
        result = critique.critique(ctx)
        assert result.overall_assessment == "ok"
        assert len(result.issues) == 0

    def test_redundant_search(self):
        critique = StrategyCritique()
        ctx = CognitiveContext(
            user_message="test",
            tool_history=[
                {"name": "tavily_search", "arguments": {"query": "python"}, "result": "ok"},
                {"name": "tavily_search", "arguments": {"query": "python"}, "result": "ok"},
            ],
        )
        result = critique.critique(ctx)
        issue_types = [i.type for i in result.issues]
        assert "redundant_search" in issue_types

    def test_stalled_progress(self):
        critique = StrategyCritique()
        task = TaskIR(
            goal="test",
            subtasks=[SubtaskIR(id="s1", description="search")],
        )
        ctx = CognitiveContext(
            user_message="test",
            round_count=4,
            task_ir=task,
        )
        result = critique.critique(ctx)
        issue_types = [i.type for i in result.issues]
        assert "stalled_progress" in issue_types


class TestPlanRepair:
    def test_repair_redundant_search(self):
        repair = PlanRepair()
        task = TaskIR(
            goal="test",
            subtasks=[
                SubtaskIR(id="s1", description="search", tool_hint="tavily_search", status="pending"),
            ],
        )
        critique = StrategyCritiqueResult(
            issues=[StrategyIssue(type="redundant_search", description="dup", severity="medium")],
            needs_plan_repair=False,
        )
        repairs = repair.repair(task, critique)
        assert len(repairs) > 0
        assert task.subtasks[0].tool_hint is None

    def test_repair_stalled_progress(self):
        repair = PlanRepair()
        task = TaskIR(
            goal="test",
            subtasks=[
                SubtaskIR(id="s1", description="step1", depends_on=["s0"]),
                SubtaskIR(id="s2", description="step2", depends_on=["s1"]),
            ],
        )
        critique = StrategyCritiqueResult(
            issues=[StrategyIssue(type="stalled_progress", description="stalled", severity="high")],
            needs_plan_repair=True,
        )
        repairs = repair.repair(task, critique)
        assert len(repairs) > 0
        assert task.subtasks[0].depends_on == []


# ========================
# SemanticCompressor 测试
# ========================


class TestKeyInfoProtector:
    def test_extract_numbers(self):
        protector = KeyInfoProtector()
        text = "气温2°C，降水概率15%，风速3级"
        sents = protector.extract_key_sentences(text, max_sentences=3)
        assert len(sents) > 0

    def test_extract_dates(self):
        protector = KeyInfoProtector()
        text = "2024年1月15日晴天。今天心情不错。2024年2月20日下雨。"
        sents = protector.extract_key_sentences(text, max_sentences=2)
        assert len(sents) > 0

    def test_extract_errors(self):
        protector = KeyInfoProtector()
        text = "程序运行正常。Error: connection timeout。请重试。"
        sents = protector.extract_key_sentences(text, max_sentences=2)
        assert any("Error" in s for s in sents)

    def test_empty_text(self):
        protector = KeyInfoProtector()
        sents = protector.extract_key_sentences("", max_sentences=3)
        assert sents == []


class TestSemanticCompressor:
    def test_short_text_unchanged(self):
        compressor = SemanticCompressor()
        text = "短文本"
        result = compressor.compress("tavily_search", text)
        assert result == text

    def test_search_compression(self):
        compressor = SemanticCompressor()
        text = "A" * 2000
        result = compressor.compress("tavily_search", text)
        assert len(result) <= 1000

    def test_code_compression(self):
        compressor = SemanticCompressor()
        text = "B" * 3000
        result = compressor.compress("execute_code", text)
        assert len(result) <= 1600

    def test_general_compression(self):
        compressor = SemanticCompressor()
        text = "C" * 2000
        result = compressor.compress("some_tool", text)
        assert len(result) <= 1200

    def test_preserves_key_info(self):
        compressor = SemanticCompressor()
        text = (
            "临泉县2024年1月15日天气预报：气温2°C，降水概率15%。"
            "以下是详细内容" + "X" * 2000
        )
        result = compressor.compress("tavily_search", text)
        assert "2°C" in result or "2024" in result or "15%" in result

    @pytest.mark.asyncio
    async def test_async_compression(self):
        compressor = SemanticCompressor()
        text = "D" * 2000
        result = await compressor.compress_async("tavily_search", text)
        assert isinstance(result, CompressionResult)
        assert result.compressed_length <= 1000
        assert result.original_length == 2000


# ========================
# ToolCapability 测试
# ========================


class TestToolCapability:
    def test_default_capability(self):
        cap = ToolCapability()
        assert cap.reliability == 0.95
        assert cap.capability_tags == []

    def test_custom_capability(self):
        cap = ToolCapability(
            capability_tags=["search", "info"],
            reliability=0.9,
            latency_p50=2.0,
        )
        assert "search" in cap.capability_tags
        assert cap.reliability == 0.9


class TestToolStats:
    def test_initial_stats(self):
        stats = ToolStats()
        assert stats.call_count == 0
        assert stats.success_rate == 0.0
        assert stats.avg_latency == 0.0

    def test_record_calls(self):
        stats = ToolStats()
        stats.record_call(True, 1.0)
        stats.record_call(True, 2.0)
        stats.record_call(False, 0.5, "timeout")
        assert stats.call_count == 3
        assert abs(stats.success_rate - 2 / 3) < 0.01
        assert abs(stats.avg_latency - 3.5 / 3) < 0.01
        assert stats.last_error == "timeout"


class TestToolCapabilityRegistry:
    def test_default_tools(self):
        registry = ToolCapabilityRegistry()
        cap = registry.get_capability("tavily_search")
        assert cap is not None
        assert "search" in cap.capability_tags

    def test_recommend_search_tools(self):
        registry = ToolCapabilityRegistry()
        tools = registry.recommend_tools(["search"])
        assert "tavily_search" in tools

    def test_recommend_code_tools(self):
        registry = ToolCapabilityRegistry()
        tools = registry.recommend_tools(["code_exec"])
        assert len(tools) > 0

    def test_infer_tool_hint_search(self):
        registry = ToolCapabilityRegistry()
        hint = registry.infer_tool_hint("搜索天气信息")
        assert hint == "tavily_search"

    def test_infer_tool_hint_code(self):
        registry = ToolCapabilityRegistry()
        hint = registry.infer_tool_hint("执行代码生成图表")
        assert hint in ("execute_code", "execute_file")

    def test_infer_tool_hint_file(self):
        registry = ToolCapabilityRegistry()
        hint = registry.infer_tool_hint("保存报告文件")
        assert hint in ("write_file", "read_file")

    def test_record_call_updates_stats(self):
        registry = ToolCapabilityRegistry()
        registry.record_call("tavily_search", True, 2.0)
        stats = registry.get_stats("tavily_search")
        assert stats is not None
        assert stats.call_count == 1
        assert stats.success_count == 1

    def test_register_custom_capability(self):
        registry = ToolCapabilityRegistry()
        cap = ToolCapability(
            capability_tags=["custom"],
            reliability=0.99,
            latency_p50=0.1,
        )
        registry.register_capability("my_tool", cap)
        assert registry.get_capability("my_tool") is not None
        tools = registry.recommend_tools(["custom"])
        assert "my_tool" in tools


# ========================
# CognitiveContext 测试
# ========================


class TestCognitiveContext:
    def test_default_context(self):
        ctx = CognitiveContext()
        assert ctx.round_count == 0
        assert ctx.is_complete is False
        assert ctx.task_ir is None

    def test_to_dict(self):
        ctx = CognitiveContext(user_message="test", round_count=3)
        d = ctx.to_dict()
        assert d["user_message"] == "test"
        assert d["round_count"] == 3
        assert d["has_task_ir"] is False

    def test_with_task_ir(self):
        task = TaskIR(goal="test")
        ctx = CognitiveContext(user_message="test", task_ir=task)
        d = ctx.to_dict()
        assert d["has_task_ir"] is True


# ========================
# StateGraph 测试
# ========================


class TestStateGraph:
    @pytest.mark.asyncio
    async def test_simple_graph(self):
        graph = StateGraph()

        async def start_handler(ctx):
            return {"next": "end"}

        async def end_handler(ctx):
            return {"done": True}

        graph.add_node("start", NodeKind.THINK, start_handler)
        graph.add_node("end", NodeKind.OUTPUT, end_handler)
        graph.set_entry("start")
        graph.add_edge("start", "end", condition="next")

        result = await graph.run({"next": None})
        assert result.get("done") is True

    @pytest.mark.asyncio
    async def test_loop_detection(self):
        graph = StateGraph()
        call_count = 0

        async def loop_handler(ctx):
            nonlocal call_count
            call_count += 1
            return {"loop_again": True}

        graph.add_node("loop", NodeKind.THINK, loop_handler, max_visits=3)
        graph.set_entry("loop")
        graph.add_edge("loop", "loop", condition="loop_again")

        result = await graph.run({})
        assert call_count <= 3
        assert result.get("_loop_detected") is True

    @pytest.mark.asyncio
    async def test_checkpoint_restore(self):
        graph = StateGraph()

        async def handler(ctx):
            return {"value": 42}

        graph.add_node("n1", NodeKind.THINK, handler)
        graph.set_entry("n1")

        snapshot = graph.checkpoint()
        assert "visit_counts" in snapshot

        graph.restore(snapshot)
        assert graph._nodes["n1"].visit_count == 0


# ========================
# Reflector 测试
# ========================


class TestReflector:
    @pytest.mark.asyncio
    async def test_no_tool_history(self):
        reflector = Reflector()
        ctx = CognitiveContext()
        result = await reflector.reflect(ctx)
        assert result["needs_retry"] is False

    @pytest.mark.asyncio
    async def test_search_success(self):
        reflector = Reflector()
        ctx = CognitiveContext(
            tool_history=[{"name": "tavily_search", "result": "Found relevant data about weather in Linquan county with temperature details"}],
        )
        result = await reflector.reflect(ctx)
        assert result["needs_retry"] is False

    @pytest.mark.asyncio
    async def test_search_too_short(self):
        reflector = Reflector()
        ctx = CognitiveContext(
            tool_history=[{"name": "tavily_search", "result": "No"}],
        )
        result = await reflector.reflect(ctx)
        assert result["needs_retry"] is False

    @pytest.mark.asyncio
    async def test_code_failure_retry(self):
        reflector = Reflector()
        ctx = CognitiveContext(
            tool_history=[{"name": "execute_code", "result": "Error: syntax error\nTraceback: ..."}],
            retry_count=0,
            max_retries=3,
        )
        result = await reflector.reflect(ctx)
        assert result["needs_retry"] is True

    @pytest.mark.asyncio
    async def test_tool_error_retry(self):
        reflector = Reflector()
        ctx = CognitiveContext(
            tool_history=[{"name": "tavily_search", "result": "", "error": "timeout"}],
            retry_count=1,
            max_retries=3,
        )
        result = await reflector.reflect(ctx)
        assert result["needs_retry"] is True


# ========================
# ToolRouter 测试
# ========================


class TestToolRouter:
    def test_search_limit(self):
        router = ToolRouter()
        ctx = CognitiveContext(search_count=3, max_search_count=3)
        ok, msg = router.validate_tool_call("tavily_search", {}, ctx)
        assert not ok
        assert "限制" in msg

    def test_consecutive_search_blocked(self):
        router = ToolRouter()
        ctx = CognitiveContext(search_count=1, max_search_count=3, last_action_was_search=True)
        ok, msg = router.validate_tool_call("tavily_search", {}, ctx)
        assert not ok
        assert "ReAct" in msg

    def test_delete_blocked(self):
        router = ToolRouter()
        ctx = CognitiveContext()
        ok, msg = router.validate_tool_call("delete_file", {}, ctx)
        assert not ok

    def test_normal_tool_allowed(self):
        router = ToolRouter()
        ctx = CognitiveContext(search_count=0, max_search_count=3)
        ok, msg = router.validate_tool_call("execute_code", {}, ctx)
        assert ok

    def test_record_execution(self):
        router = ToolRouter()
        router.record_execution("tavily_search", {})
        assert router._search_count == 1
        assert router._last_was_search is True


# ========================
# CognitiveRuntime 集成测试
# ========================


class TestCognitiveRuntimeIntegration:
    def _make_runtime(self):
        async def mock_chat(msgs, **kw):
            class MockResponse:
                content = "测试回复"
                tool_calls = None
            return MockResponse()

        async def mock_chat_with_tools(msgs, tools, **kw):
            class MockResponse:
                content = "测试回复"
                tool_calls = None
            return MockResponse()

        async def mock_tool_execute(name, args):
            return f"工具 {name} 执行成功"

        async def mock_output(text):
            pass

        return CognitiveRuntime(
            llm_chat_fn=mock_chat,
            llm_chat_with_tools_fn=mock_chat_with_tools,
            tool_execute_fn=mock_tool_execute,
            output_fn=mock_output,
        )

    def test_runtime_creation(self):
        runtime = self._make_runtime()
        assert runtime._memory is None
        assert runtime._tool_capability is None

    def test_runtime_with_capability(self):
        runtime = self._make_runtime()
        registry = ToolCapabilityRegistry()
        runtime._tool_capability = registry
        hint = registry.infer_tool_hint("搜索天气")
        assert hint == "tavily_search"

    @pytest.mark.asyncio
    async def test_runtime_run_simple(self):
        runtime = self._make_runtime()
        ctx = CognitiveContext(
            user_message="你好",
            messages=[{"role": "user", "content": "你好"}],
            max_rounds=2,
        )
        result = await runtime.run(ctx)
        assert result.is_complete

    @pytest.mark.asyncio
    async def test_runtime_with_task_ir(self):
        runtime = self._make_runtime()
        task = parse_task_ir_from_message("搜索Python最新版本")
        ctx = CognitiveContext(
            user_message="搜索Python最新版本",
            messages=[{"role": "user", "content": "搜索Python最新版本"}],
            max_rounds=2,
            task_ir=task,
        )
        result = await runtime.run(ctx)
        assert result.is_complete
