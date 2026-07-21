"""端到端模拟测试 — 模拟完整任务执行流程

模拟 "十五五规划都是讲了一些什么，绘制图表，生成一份详细的报告" 的完整执行过程：
  1. TaskIR 生成
  2. THINK → ACT (搜索) → OBSERVE → REFLECT → PLAN
  3. THINK → ACT (代码执行) → OBSERVE → REFLECT → PLAN
  4. THINK → ACT (文件写入) → OBSERVE → REFLECT → PLAN
  5. THINK → OUTPUT

验证：
  - TaskIR 在整个流程中被正确更新
  - Memory 被正确调用
  - Strategy Critique 检测到问题
  - Plan Repair 正确修复
  - 最终输出包含完整内容
"""

import asyncio
import json
import time
from typing import Any

import pytest

from long.cognitive.task_ir import TaskIR, SubtaskIR, parse_task_ir_from_message
from long.cognitive.planner import TaskPlanner, PlanResult
from long.cognitive.reflection import StrategyCritique, PlanRepair
from long.cognitive.compression import SemanticCompressor, KeyInfoProtector
from long.cognitive.runtime import (
    CognitiveContext,
    CognitiveRuntime,
    ExecutionMode,
    Reflector,
    ToolRouter,
)
from long.capabilities.tool_capability import ToolCapabilityRegistry


# ========================
# Mock LLM 响应
# ========================

SEARCH_RESULT = """十五五规划（2026-2030年）核心内容：
1. 经济发展：GDP年均增长5%左右，到2030年达到180万亿元
2. 科技创新：研发投入占GDP比重达3.0%，约5.4万亿元
3. 数字经济：核心产业占GDP比重达12%，约21.6万亿元
4. 绿色转型：非化石能源占比达25%，单位GDP能耗降13.5%
5. 民生改善：人均预期寿命达80岁，基本养老参保率95%
6. 城镇化：常住人口城镇化率达70%
7. 粮食安全：粮食产能1.45万亿斤以上
8. 产业升级：先进制造业占比35%
9. 乡村振兴：农村居民收入增长与GDP同步
10. 对外开放：共建"一带一路"高质量发展"""

CODE_RESULT = """图表已生成:
- 柱状图: 十四五vs十五五关键指标对比
- 饼图: 重点领域投资占比
- 折线图: GDP增长趋势
文件保存至: output/plan_charts.html"""

FILE_RESULT = "✅ 文件写入成功: output/十五五规划报告.md (5.8KB)"


class MockLLMResponse:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class MockLLM:
    """模拟 LLM 的多轮对话行为

    Round 1: 返回搜索工具调用
    Round 2: 返回代码执行工具调用
    Round 3: 返回文件写入工具调用
    Round 4: 返回最终文本回答
    """

    def __init__(self):
        self.call_count = 0
        self.chat_calls = 0

    async def chat_with_tools(self, messages, tools, **kwargs):
        self.call_count += 1

        if self.call_count == 1:
            return MockLLMResponse(
                tool_calls=[{
                    "id": "call_1",
                    "name": "tavily_search",
                    "arguments": {"query": "十五五规划 2026-2030 核心内容"},
                }]
            )
        elif self.call_count == 2:
            return MockLLMResponse(
                tool_calls=[{
                    "id": "call_2",
                    "name": "execute_code",
                    "arguments": {
                        "code": "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n...",
                        "language": "python",
                    },
                }]
            )
        elif self.call_count == 3:
            return MockLLMResponse(
                tool_calls=[{
                    "id": "call_3",
                    "name": "write_file",
                    "arguments": {
                        "path": "output/十五五规划报告.md",
                        "content": "# 十五五规划详细报告\n...",
                    },
                }]
            )
        else:
            return MockLLMResponse(
                content=(
                    "# 十五五规划核心内容分析报告\n\n"
                    "## 一、规划概述\n"
                    "十五五规划（2026-2030年）是中国第十四个五年规划...\n\n"
                    "## 二、核心指标\n"
                    "- GDP年均增长5%左右\n"
                    "- 研发投入占GDP比重达3.0%\n"
                    "- 数字经济核心产业占GDP比重达12%\n\n"
                    "## 三、图表说明\n"
                    "已生成对比图表，详见附件。\n\n"
                    "## 四、总结\n"
                    "十五五规划聚焦高质量发展、科技自立自强和绿色转型三大方向。"
                )
            )

    async def chat(self, messages, **kwargs):
        self.chat_calls += 1
        return MockLLMResponse(content="模拟回复")


class MockToolExecutor:
    """模拟工具执行"""

    def __init__(self):
        self.executed_tools = []

    async def execute(self, tool_name, arguments):
        self.executed_tools.append((tool_name, arguments))

        if tool_name == "tavily_search":
            return SEARCH_RESULT
        elif tool_name == "execute_code":
            return CODE_RESULT
        elif tool_name == "write_file":
            return FILE_RESULT
        elif tool_name == "read_file":
            return "文件内容"
        else:
            return f"工具 {tool_name} 执行成功"


class MockMemory:
    """模拟记忆系统"""

    def __init__(self):
        self.stored = []
        self.search_results = []

    async def store(self, content, **kwargs):
        self.stored.append({"content": content, **kwargs})

    async def search(self, query, **kwargs):
        return self.search_results

    async def auto_promote(self):
        pass


# ========================
# 端到端测试
# ========================


class TestEndToEndFlow:
    """端到端模拟测试 — 完整任务执行流程"""

    @pytest.mark.asyncio
    async def test_full_task_flow(self):
        """模拟完整任务：搜索→代码→文件→输出"""
        mock_llm = MockLLM()
        mock_tool = MockToolExecutor()
        mock_memory = MockMemory()
        output_text = []

        async def mock_output(text):
            output_text.append(text)

        registry = ToolCapabilityRegistry()

        runtime = CognitiveRuntime(
            llm_chat_fn=mock_llm.chat,
            llm_chat_with_tools_fn=mock_llm.chat_with_tools,
            tool_execute_fn=mock_tool.execute,
            output_fn=mock_output,
            memory_controller=mock_memory,
            tool_capability_registry=registry,
        )

        user_message = "十五五规划都是讲了一些什么，绘制图表，生成一份详细的报告"

        task_ir = parse_task_ir_from_message(user_message)
        assert task_ir is not None
        assert len(task_ir.subtasks) >= 2

        for subtask in task_ir.subtasks:
            if not subtask.tool_hint:
                subtask.tool_hint = registry.infer_tool_hint(subtask.description)

        context = CognitiveContext(
            user_message=user_message,
            messages=[
                {"role": "system", "content": "你是一个智能助手"},
                {"role": "user", "content": user_message},
            ],
            max_rounds=8,
            task_ir=task_ir,
        )

        result = await runtime.run(
            context,
            extra={
                "_tools": [
                    {"type": "function", "function": {"name": "tavily_search", "description": "搜索", "parameters": {}}},
                    {"type": "function", "function": {"name": "execute_code", "description": "执行代码", "parameters": {}}},
                    {"type": "function", "function": {"name": "write_file", "description": "写入文件", "parameters": {}}},
                ],
            },
        )

        assert result.is_complete, f"任务未完成: errors={result.errors}"

        assert mock_llm.call_count >= 2, f"LLM 调用次数不足: {mock_llm.call_count}"

        assert len(mock_tool.executed_tools) >= 1, f"工具执行次数不足: {mock_tool.executed_tools}"

        assert result.final_output, "最终输出为空"

        print(f"\n✅ 端到端测试通过!")
        print(f"  LLM 调用次数: {mock_llm.call_count}")
        print(f"  工具执行: {[t[0] for t in mock_tool.executed_tools]}")
        print(f"  最终输出长度: {len(result.final_output)} 字符")
        print(f"  TaskIR 进度: {task_ir.progress_ratio():.0%}")
        print(f"  Memory 存储: {len(mock_memory.stored)} 条")

    @pytest.mark.asyncio
    async def test_task_ir_lifecycle(self):
        """测试 TaskIR 在完整生命周期中的状态变化"""
        task = parse_task_ir_from_message("搜索天气，画折线图，生成报告")

        assert len(task.subtasks) >= 2
        assert task.progress_ratio() == 0.0

        next_sub = task.next_executable_subtask()
        assert next_sub is not None
        assert next_sub.tool_hint == "execute_file"

        task.mark_subtask_in_progress(next_sub.id)
        assert next_sub.status == "in_progress"

        task.complete_subtask(next_sub.id, "搜索到天气数据")
        assert next_sub.status == "completed"
        assert task.progress_ratio() > 0

        task.add_key_fact("气温2°C")
        task.add_key_fact("降水15%")
        assert len(task.key_facts) == 2

        next_sub = task.next_executable_subtask()
        assert next_sub is not None
        assert next_sub.tool_hint == "execute_code"

        task.mark_subtask_in_progress(next_sub.id)
        task.complete_subtask(next_sub.id, "图表已生成")

        next_sub = task.next_executable_subtask()
        if next_sub:
            task.mark_subtask_in_progress(next_sub.id)
            task.complete_subtask(next_sub.id, "报告已保存")

        prompt_text = task.to_prompt_text()
        assert "已完成" in prompt_text
        assert "关键事实" in prompt_text

        print(f"\n✅ TaskIR 生命周期测试通过!")
        print(f"  进度: {task.progress_ratio():.0%}")
        print(f"  关键事实: {task.key_facts}")
        print(f"  Prompt 文本长度: {len(prompt_text)} 字符")

    @pytest.mark.asyncio
    async def test_strategy_critique_in_flow(self):
        """测试策略批判在执行流程中的检测能力"""
        critique = StrategyCritique()

        ctx = CognitiveContext(
            user_message="搜索天气",
            tool_history=[
                {"name": "tavily_search", "arguments": {"query": "天气"}, "result": "ok"},
                {"name": "tavily_search", "arguments": {"query": "天气"}, "result": "ok"},
            ],
            round_count=3,
            task_ir=TaskIR(
                goal="搜索天气",
                subtasks=[SubtaskIR(id="s1", description="搜索", status="pending")],
            ),
        )

        result = critique.critique(ctx)
        assert len(result.issues) > 0
        issue_types = [i.type for i in result.issues]
        assert "redundant_search" in issue_types

        repair = PlanRepair()
        repairs = repair.repair(ctx.task_ir, result)
        assert len(repairs) > 0

        print(f"\n✅ 策略批判测试通过!")
        print(f"  检测到问题: {issue_types}")
        print(f"  修复操作: {repairs}")

    @pytest.mark.asyncio
    async def test_planner_with_task_ir(self):
        """测试 Planner 基于 TaskIR 的规划决策"""
        planner = TaskPlanner()

        task = TaskIR(
            goal="十五五规划报告",
            subtasks=[
                SubtaskIR(id="s1", description="搜索", tool_hint="tavily_search"),
                SubtaskIR(id="s2", description="图表", tool_hint="execute_code", depends_on=["s1"]),
                SubtaskIR(id="s3", description="报告", tool_hint="write_file", depends_on=["s2"]),
            ],
        )

        ctx = CognitiveContext(
            user_message="十五五规划报告",
            task_ir=task,
            round_count=1,
            max_rounds=8,
            search_count=0,
            max_search_count=3,
        )

        plan = planner.plan(ctx)
        assert plan.should_continue
        assert plan.next_subtask is not None
        assert plan.next_subtask.id == "s1"

        task.complete_subtask("s1", "搜索完成")
        ctx.search_count = 1

        plan = planner.plan(ctx)
        assert plan.should_continue
        assert plan.next_subtask.id == "s2"

        task.complete_subtask("s2", "图表完成")
        ctx.search_count = 3
        ctx.max_search_count = 3

        plan = planner.plan(ctx)
        assert plan.should_continue

        task.complete_subtask("s3", "报告完成")

        plan = planner.plan(ctx)
        assert plan.is_complete

        print(f"\n✅ Planner 测试通过!")
        print(f"  最终状态: is_complete={plan.is_complete}")

    @pytest.mark.asyncio
    async def test_compression_preserves_key_info(self):
        """测试压缩保留关键信息"""
        compressor = SemanticCompressor()

        search_result = (
            "十五五规划（2026-2030年）核心内容：\n"
            "1. GDP年均增长5%左右，到2030年达到180万亿元\n"
            "2. 研发投入占GDP比重达3.0%，约5.4万亿元\n"
            "3. 非化石能源占比达25%\n"
            "4. 人均预期寿命达80岁\n"
            "5. 城镇化率达70%\n"
            + "详细内容..." * 200
        )

        compressed = compressor.compress("tavily_search", search_result)
        assert len(compressed) <= 1000

        key_info_preserved = any(
            kw in compressed
            for kw in ["5%", "3.0%", "25%", "180万亿", "2026"]
        )
        assert key_info_preserved, f"关键信息丢失: {compressed[:200]}"

        print(f"\n✅ 压缩保留关键信息测试通过!")
        print(f"  原始: {len(search_result)} 字符")
        print(f"  压缩后: {len(compressed)} 字符")
        print(f"  压缩比: {len(compressed)/len(search_result):.1%}")

    @pytest.mark.asyncio
    async def test_tool_capability_in_flow(self):
        """测试 Tool Capability 在执行流程中的推荐能力"""
        registry = ToolCapabilityRegistry()

        hint1 = registry.infer_tool_hint("搜索十五五规划信息")
        assert hint1 == "tavily_search"

        hint2 = registry.infer_tool_hint("绘制折线图对比数据")
        assert hint2 in ("execute_code", "execute_file")

        hint3 = registry.infer_tool_hint("保存报告文件")
        assert hint3 in ("write_file", "read_file")

        search_tools = registry.recommend_tools(["search"])
        assert "tavily_search" in search_tools

        code_tools = registry.recommend_tools(["code_exec", "data_viz"])
        assert len(code_tools) > 0

        registry.record_call("tavily_search", True, 2.5)
        registry.record_call("tavily_search", True, 1.8)
        registry.record_call("tavily_search", False, 5.0, "timeout")

        stats = registry.get_stats("tavily_search")
        assert stats.call_count == 3
        assert abs(stats.success_rate - 2 / 3) < 0.01

        print(f"\n✅ Tool Capability 测试通过!")
        print(f"  搜索推荐: {search_tools}")
        print(f"  代码推荐: {code_tools}")
        print(f"  搜索统计: calls={stats.call_count}, rate={stats.success_rate:.2f}")

    @pytest.mark.asyncio
    async def test_reflector_three_layers(self):
        """测试三层反思系统"""
        reflector = Reflector()

        ctx = CognitiveContext(
            user_message="搜索天气",
            tool_history=[
                {"name": "tavily_search", "result": "临泉县2024年1月15日晴天，气温2°C，降水概率15%"},
            ],
        )

        result = await reflector.reflect(ctx)
        assert not result["needs_retry"]

        ctx2 = CognitiveContext(
            user_message="搜索天气",
            tool_history=[
                {"name": "execute_code", "result": "Error: ModuleNotFoundError: No module named 'matplotlib'"},
            ],
            retry_count=0,
            max_retries=3,
        )

        result2 = await reflector.reflect(ctx2)
        assert result2["needs_retry"]
        assert "修复" in result2["reflection"]

        ctx3 = CognitiveContext(
            user_message="搜索天气",
            tool_history=[
                {"name": "tavily_search", "arguments": {"query": "天气"}, "result": "ok"},
                {"name": "tavily_search", "arguments": {"query": "天气"}, "result": "ok"},
            ],
            round_count=4,
            task_ir=TaskIR(
                goal="搜索天气",
                subtasks=[SubtaskIR(id="s1", description="搜索", status="pending")],
            ),
        )

        result3 = await reflector.reflect(ctx3)
        assert ctx3.strategy_critique is not None
        assert len(ctx3.strategy_critique.issues) > 0

        print(f"\n✅ 三层反思测试通过!")
        print(f"  Layer 1 (搜索成功): {result['reflection']}")
        print(f"  Layer 1 (代码失败): {result2['reflection']}")
        print(f"  Layer 2 (策略问题): {[i.type for i in ctx3.strategy_critique.issues]}")

    @pytest.mark.asyncio
    async def test_memory_integration(self):
        """测试记忆系统集成"""
        mock_memory = MockMemory()

        runtime = CognitiveRuntime(
            llm_chat_fn=MockLLM().chat,
            llm_chat_with_tools_fn=MockLLM().chat_with_tools,
            tool_execute_fn=MockToolExecutor().execute,
            output_fn=lambda text: None,
            memory_controller=mock_memory,
        )

        assert runtime._memory is mock_memory

        ctx = CognitiveContext(user_message="test")
        await runtime._ensure_task_ir(ctx)
        assert ctx.task_ir is not None

        print(f"\n✅ 记忆集成测试通过!")
        print(f"  Memory 存储: {len(mock_memory.stored)} 条")


class TestDuplicateToolCall:
    """重复工具调用检测测试"""

    @pytest.mark.asyncio
    async def test_duplicate_tool_calls_are_cached(self):
        """验证相同工具+参数调用被缓存，第二次直接返回缓存结果"""
        runtime = CognitiveRuntime(
            llm_chat_fn=lambda msgs, **kw: MockLLMResponse(content="ok"),
            llm_chat_with_tools_fn=lambda msgs, tools, **kw: MockLLMResponse(
                tool_calls=[{"id": "call_1", "name": "query_weather", "arguments": {"city": "杭州"}}],
            ),
            tool_execute_fn=lambda name, args: f"结果:{name}({args})",
            output_fn=lambda text: None,
        )

        # 验证 _make_tool_call_key
        key1 = runtime._make_tool_call_key("query_weather", {"city": "杭州"})
        key2 = runtime._make_tool_call_key("query_weather", {"city": "杭州"})
        key3 = runtime._make_tool_call_key("query_weather", {"city": "北京"})
        assert key1 == key2, "相同参数应生成相同 key"
        assert key1 != key3, "不同参数应生成不同 key"

        # 验证 _tool_call_cache
        runtime._tool_call_cache[key1] = ("结果:query_weather({'city': '杭州'})", 0.0)
        assert key1 in runtime._tool_call_cache
        cached, _ts = runtime._tool_call_cache[key1]
        assert "杭州" in cached

    @pytest.mark.asyncio
    async def test_make_tool_call_key_stability(self):
        """验证缓存 key 的稳定性和可读性"""
        runtime = CognitiveRuntime.__new__(CognitiveRuntime)

        key = runtime._make_tool_call_key("query_weather", {"city": "杭州", "units": "metric"})
        assert "query_weather" in key
        assert "杭州" in key
        assert "metric" in key

        # 不同参数顺序应生成相同 key
        key_a = runtime._make_tool_call_key("test", {"b": 2, "a": 1})
        key_b = runtime._make_tool_call_key("test", {"a": 1, "b": 2})
        assert key_a == key_b, "不同参数顺序应生成相同 key"

    @pytest.mark.asyncio
    async def test_different_tools_different_keys(self):
        """验证不同工具名生成不同 key"""
        runtime = CognitiveRuntime.__new__(CognitiveRuntime)

        key1 = runtime._make_tool_call_key("query_weather", {"city": "杭州"})
        key2 = runtime._make_tool_call_key("tavily_search", {"city": "杭州"})
        assert key1 != key2, "不同工具名应生成不同 key"
