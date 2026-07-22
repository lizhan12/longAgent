#!/usr/bin/env python3
"""Cognitive Runtime 单元测试 — 不依赖外部 LLM API"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from long.cognitive.runtime import (
    StateGraph, CognitiveContext, CognitiveRuntime,
    Reflector, ToolRouter, ObservationCompressor,
    NodeKind, GraphNode, GraphEdge,
)


# ── Mock 函数 ──

class MockLLMResponse:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


call_log = []


async def mock_llm_chat(messages, purpose="chat"):
    call_log.append(("chat", purpose, len(messages)))
    return MockLLMResponse(content="这是模拟回复")


async def mock_llm_chat_with_tools(messages, tools_list, purpose="chat", **kwargs):
    call_log.append(("chat_with_tools", purpose, len(messages)))

    if len(call_log) <= 1:
        return MockLLMResponse(tool_calls=[
            {"id": "tc_1", "name": "tavily_search", "arguments": {"query": "余杭天气"}},
        ])
    elif len(call_log) <= 2:
        return MockLLMResponse(tool_calls=[
            {"id": "tc_2", "name": "write_file", "arguments": {"path": "report.md", "content": "天气报告"}},
        ])
    else:
        return MockLLMResponse(content="任务完成，天气报告已生成。")


async def mock_tool_execute(tool_name, arguments):
    call_log.append(("tool", tool_name, str(arguments)[:50]))
    if tool_name == "tavily_search":
        return "✅ 搜索成功: 余杭区未来7天天气 - 晴转多云，气温18-28°C"
    elif tool_name == "write_file":
        return "✅ 文件写入成功"
    elif tool_name == "execute_code":
        return "✅ 代码执行成功"
    return f"✅ {tool_name} 执行成功"


output_results = []


async def mock_output(text):
    output_results.append(text)


# ── 测试用例 ──

async def test_state_graph_basic():
    """测试 StateGraph 基本功能"""
    print("\n" + "=" * 60)
    print("测试 1: StateGraph 基本功能")
    print("=" * 60)

    graph = StateGraph()
    execution_order = []

    async def handler_start(ctx):
        execution_order.append("start")
        return {"next": True}

    async def handler_middle(ctx):
        execution_order.append("middle")
        return {"done": False}

    async def handler_end(ctx):
        execution_order.append("end")
        return {"done": True}

    graph.add_node("start", NodeKind.THINK, handler_start)
    graph.add_node("middle", NodeKind.ACT, handler_middle)
    graph.add_node("end", NodeKind.OUTPUT, handler_end)
    graph.set_entry("start")

    graph.add_edge("start", "middle", condition="next")
    graph.add_edge("middle", "end")

    result = await graph.run({})
    assert execution_order == ["start", "middle", "end"], f"执行顺序错误: {execution_order}"
    assert result.get("done") is True
    print("✅ StateGraph 基本流程正确")


async def test_state_graph_conditional():
    """测试 StateGraph 条件分支"""
    print("\n" + "=" * 60)
    print("测试 2: StateGraph 条件分支")
    print("=" * 60)

    graph = StateGraph()
    execution_order = []

    async def handler_think(ctx):
        execution_order.append("think")
        return {"has_tool_calls": True}

    async def handler_act(ctx):
        execution_order.append("act")
        return {"has_tool_calls": True}

    async def handler_output(ctx):
        execution_order.append("output")
        return {"is_complete": True}

    graph.add_node("think", NodeKind.THINK, handler_think)
    graph.add_node("act", NodeKind.ACT, handler_act)
    graph.add_node("output", NodeKind.OUTPUT, handler_output)
    graph.set_entry("think")

    graph.add_edge("think", "act", condition="has_tool_calls")
    graph.add_edge("think", "output", condition="has_final_text")
    graph.add_edge("act", "output")

    result = await graph.run({})
    assert execution_order == ["think", "act", "output"], f"执行顺序错误: {execution_order}"
    print("✅ StateGraph 条件分支正确")


async def test_state_graph_loop_detection():
    """测试 StateGraph 循环检测"""
    print("\n" + "=" * 60)
    print("测试 3: StateGraph 循环检测")
    print("=" * 60)

    graph = StateGraph()
    visit_count = 0

    async def handler_loop(ctx):
        nonlocal visit_count
        visit_count += 1
        return {"should_continue": True}

    async def handler_exit(ctx):
        return {"is_complete": True}

    graph.add_node("loop_node", NodeKind.THINK, handler_loop, max_visits=3)
    graph.add_node("exit", NodeKind.OUTPUT, handler_exit)
    graph.set_entry("loop_node")

    graph.add_edge("loop_node", "loop_node", condition="should_continue")
    graph.add_edge("loop_node", "exit", condition="is_complete")

    result = await graph.run({})
    assert visit_count == 3, f"循环检测失败，访问次数: {visit_count}"
    assert result.get("_loop_detected") is True
    print(f"✅ 循环检测正确，访问 {visit_count} 次后停止")


async def test_tool_router():
    """测试 ToolRouter 策略路由"""
    print("\n" + "=" * 60)
    print("测试 4: ToolRouter 策略路由")
    print("=" * 60)

    router = ToolRouter()
    ctx = CognitiveContext(user_message="test", max_search_count=2)

    allowed, reason = router.validate_tool_call("tavily_search", {"query": "test"}, ctx)
    assert allowed, f"第一次搜索应该被允许: {reason}"
    print("  ✅ 第一次搜索允许")

    router.record_execution("tavily_search", {"query": "test"})
    ctx.search_count = 1
    ctx.last_action_was_search = True

    allowed, reason = router.validate_tool_call("tavily_search", {"query": "test2"}, ctx)
    assert not allowed, "连续搜索应该被阻止"
    assert "ReAct" in reason
    print(f"  ✅ 连续搜索被阻止: {reason}")

    ctx.last_action_was_search = False
    allowed, reason = router.validate_tool_call("tavily_search", {"query": "test3"}, ctx)
    assert allowed, f"非连续搜索应该被允许: {reason}"
    print("  ✅ 非连续搜索允许")

    router.record_execution("tavily_search", {"query": "test3"})
    ctx.search_count = 2

    allowed, reason = router.validate_tool_call("tavily_search", {"query": "test4"}, ctx)
    assert not allowed, "搜索次数超限应该被阻止"
    assert "搜索限制" in reason
    print(f"  ✅ 搜索次数超限被阻止: {reason}")

    allowed, reason = router.validate_tool_call("delete_file", {"path": "test"}, ctx)
    assert not allowed, "危险工具应该被阻止"
    print(f"  ✅ 危险工具被阻止: {reason}")


async def test_observation_compressor():
    """测试 ObservationCompressor"""
    print("\n" + "=" * 60)
    print("测试 5: ObservationCompressor")
    print("=" * 60)

    short_result = "这是一个短结果"
    assert ObservationCompressor.compress("tavily_search", short_result) == short_result
    print("  ✅ 短结果不压缩")

    long_result = "标题1: 内容1\n" * 200
    compressed = ObservationCompressor.compress("tavily_search", long_result)
    assert len(compressed) < len(long_result)
    assert "压缩" in compressed
    print(f"  ✅ 搜索结果压缩: {len(long_result)} → {len(compressed)}")

    code_result = "print('hello')\n" * 200
    compressed = ObservationCompressor.compress("execute_code", code_result)
    assert len(compressed) < len(code_result)
    assert "省略" in compressed
    print(f"  ✅ 代码结果压缩: {len(code_result)} → {len(compressed)}")


async def test_cognitive_context():
    """测试 CognitiveContext"""
    print("\n" + "=" * 60)
    print("测试 6: CognitiveContext")
    print("=" * 60)

    ctx = CognitiveContext(
        user_message="余杭天气",
        max_rounds=8,
        max_search_count=2,
    )

    d = ctx.to_dict()
    assert d["user_message"] == "余杭天气"
    assert d["current_phase"] == "think"
    assert d["round_count"] == 0
    assert d["is_complete"] is False

    ctx.round_count = 5
    ctx.search_count = 2
    ctx.errors.append("test error")
    d = ctx.to_dict()
    assert d["round_count"] == 5
    assert d["has_errors"] is True
    print("  ✅ CognitiveContext 状态转换正确")


async def test_reflector():
    """测试 Reflector 反思系统"""
    print("\n" + "=" * 60)
    print("测试 7: Reflector 反思系统")
    print("=" * 60)

    reflector = Reflector(mock_llm_chat)
    ctx = CognitiveContext(user_message="test")

    result = await reflector.reflect(ctx)
    assert "needs_retry" in result
    print(f"  ✅ 空历史反思: {result}")

    ctx.tool_history.append({"name": "tavily_search", "result": "", "error": "网络错误"})
    result = await reflector.reflect(ctx)
    assert result["needs_retry"] is True
    print(f"  ✅ 错误重试反思: {result}")

    ctx.retry_count = 3
    result = await reflector.reflect(ctx)
    assert result["needs_retry"] is False
    print(f"  ✅ 重试耗尽反思: {result}")

    ctx2 = CognitiveContext(user_message="test")
    ctx2.tool_history.append({"name": "tavily_search", "result": "✅ 搜索成功", "error": ""})
    result = await reflector.reflect(ctx2)
    assert result["needs_retry"] is False
    print(f"  ✅ 成功反思: {result}")


async def test_cognitive_runtime_full():
    """测试 CognitiveRuntime 完整流程"""
    print("\n" + "=" * 60)
    print("测试 8: CognitiveRuntime 完整流程")
    print("=" * 60)

    global call_log, output_results
    call_log = []
    output_results = []

    runtime = CognitiveRuntime(
        llm_chat_fn=mock_llm_chat,
        llm_chat_with_tools_fn=mock_llm_chat_with_tools,
        tool_execute_fn=mock_tool_execute,
        output_fn=mock_output,
    )

    context = CognitiveContext(
        user_message="余杭区未来7天天气预报",
        messages=[
            {"role": "system", "content": "你是天气助手"},
            {"role": "user", "content": "余杭区未来7天天气预报"},
        ],
        max_rounds=8,
    )

    result = await runtime.run(context, extra={"_tools": []})

    print(f"  调用日志: {len(call_log)} 次调用")
    for i, (call_type, detail, msg_count) in enumerate(call_log):
        print(f"    [{i}] {call_type}: purpose={detail}, msgs={msg_count}")

    print(f"  输出结果: {len(output_results)} 条")
    for i, text in enumerate(output_results):
        print(f"    [{i}] {text[:100]}")

    print(f"  完成状态: is_complete={result.is_complete}")
    print(f"  工具历史: {len(result.tool_history)} 条")
    for t in result.tool_history:
        print(f"    - {t['name']}: {t.get('result', '')[:60]}")

    assert result.is_complete, "认知运行时应该完成"
    print("✅ CognitiveRuntime 完整流程测试通过")


async def test_cognitive_runtime_search_limit():
    """测试 CognitiveRuntime 搜索限制"""
    print("\n" + "=" * 60)
    print("测试 9: CognitiveRuntime 搜索限制")
    print("=" * 60)

    search_call_count = 0

    async def search_only_chat_with_tools(messages, tools_list, purpose="chat", **kwargs):
        nonlocal search_call_count
        search_call_count += 1
        if search_call_count <= 3:
            return MockLLMResponse(tool_calls=[
                {"id": f"tc_{search_call_count}", "name": "tavily_search",
                 "arguments": {"query": f"搜索{search_call_count}"}},
            ])
        return MockLLMResponse(content="搜索完成")

    runtime = CognitiveRuntime(
        llm_chat_fn=mock_llm_chat,
        llm_chat_with_tools_fn=search_only_chat_with_tools,
        tool_execute_fn=mock_tool_execute,
        output_fn=mock_output,
    )

    context = CognitiveContext(
        user_message="搜索天气",
        messages=[
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "搜索天气"},
        ],
        max_rounds=10,
        max_search_count=2,
    )

    result = await runtime.run(context, extra={"_tools": []})

    actual_searches = sum(1 for t in result.tool_history if t["name"] == "tavily_search" and not t.get("result", "").startswith("["))
    intercepted = sum(1 for t in result.tool_history if t["name"] == "tavily_search" and t.get("result", "").startswith("["))

    print(f"  实际搜索次数: {actual_searches}")
    print(f"  被拦截次数: {intercepted}")
    print(f"  完成状态: is_complete={result.is_complete}")

    assert actual_searches <= context.max_search_count, f"实际搜索次数超限: {actual_searches}"
    print("✅ 搜索限制测试通过")


async def test_state_graph_checkpoint():
    """测试 StateGraph Checkpoint/Resume"""
    print("\n" + "=" * 60)
    print("测试 10: StateGraph Checkpoint/Resume")
    print("=" * 60)

    graph = StateGraph()
    step = 0

    async def handler_a(ctx):
        nonlocal step
        step += 1
        return {"step": step}

    async def handler_b(ctx):
        nonlocal step
        step += 1
        return {"step": step}

    graph.add_node("a", NodeKind.THINK, handler_a)
    graph.add_node("b", NodeKind.ACT, handler_b)
    graph.set_entry("a")
    graph.add_edge("a", "b")

    snapshot = graph.checkpoint()
    assert "state" in snapshot
    assert "history" in snapshot
    assert "visit_counts" in snapshot
    print("  ✅ Checkpoint 创建成功")

    result = await graph.run({})
    assert step == 2
    print("  ✅ 正常执行完成")

    graph.restore(snapshot)
    assert graph._history == []
    for node in graph._nodes.values():
        assert node.visit_count == 0
    print("  ✅ Restore 恢复成功")


# ── 主函数 ──

async def main():
    print("=" * 60)
    print("Cognitive Runtime 单元测试")
    print("=" * 60)

    tests = [
        test_state_graph_basic,
        test_state_graph_conditional,
        test_state_graph_loop_detection,
        test_tool_router,
        test_observation_compressor,
        test_cognitive_context,
        test_reflector,
        test_cognitive_runtime_full,
        test_cognitive_runtime_search_limit,
        test_state_graph_checkpoint,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"测试结果: ✅ {passed} 通过, ❌ {failed} 失败")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
