"""防幻觉机制单元测试 — 验证 StateGraph 状态传播和幻觉检测"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from long.cognitive.runtime import (
    CognitiveContext,
    CognitiveRuntime,
    NodeKind,
    StateGraph,
)


def _make_mock_response(content=None, tool_calls=None):
    resp = MagicMock()
    resp.content = content
    resp.tool_calls = tool_calls or []
    return resp


def _make_tool_call(name, arguments, call_id="call_1"):
    return {"id": call_id, "name": name, "arguments": arguments}


async def test_retry_think_flag_cleared_on_success():
    """测试：THINK节点成功输出时 _retry_think 标志被正确清除"""
    graph = StateGraph()

    think_call_count = 0

    async def think_handler(ctx):
        nonlocal think_call_count
        think_call_count += 1
        if think_call_count == 1:
            return {"has_tool_calls": False, "has_final_text": False, "_retry_think": True}
        return {
            "has_final_text": True,
            "has_tool_calls": False,
            "_retry_think": False,
            "_final_text": "最终输出",
        }

    async def output_handler(ctx):
        return {"is_complete": True, "_retry_think": False}

    graph.add_node("think", NodeKind.THINK, think_handler)
    graph.add_node("output", NodeKind.OUTPUT, output_handler)
    graph.set_entry("think")
    graph.add_edge("think", "output", condition="has_final_text", priority=2)
    graph.add_edge("think", "think", condition="_retry_think", priority=1)

    result = await graph.run({})

    assert result.get("has_final_text") is True, f"期望 has_final_text=True, 实际={result.get('has_final_text')}"
    assert result.get("_retry_think") is False, f"期望 _retry_think=False, 实际={result.get('_retry_think')}"
    assert result.get("_final_text") == "最终输出", f"期望 _final_text='最终输出', 实际={result.get('_final_text')}"
    assert think_call_count == 2, f"期望 THINK 被调用2次, 实际={think_call_count}"
    print("✅ test_retry_think_flag_cleared_on_success 通过")


async def test_retry_think_flag_not_cleared_causes_loop():
    """测试：如果 _retry_think 不清除且优先级更高，会导致无限循环（被 max_visits 拦截）
    这模拟了修复前的原始Bug场景
    """
    graph = StateGraph()

    think_call_count = 0

    async def think_handler(ctx):
        nonlocal think_call_count
        think_call_count += 1
        if think_call_count == 1:
            return {"has_tool_calls": False, "has_final_text": False, "_retry_think": True}
        return {
            "has_final_text": True,
            "has_tool_calls": False,
            "_final_text": "最终输出",
        }

    async def output_handler(ctx):
        return {"is_complete": True}

    graph.add_node("think", NodeKind.THINK, think_handler, max_visits=5)
    graph.add_node("output", NodeKind.OUTPUT, output_handler)
    graph.set_entry("think")
    graph.add_edge("think", "think", condition="_retry_think", priority=2)
    graph.add_edge("think", "output", condition="has_final_text", priority=1)

    result = await graph.run({})

    assert result.get("_loop_detected") is True, "期望检测到循环（_retry_think优先级更高时）"
    assert think_call_count >= 5, f"期望 THINK 被调用>=5次(被max_visits拦截), 实际={think_call_count}"
    print("✅ test_retry_think_flag_not_cleared_causes_loop 通过 (验证了Bug场景)")


async def test_edge_priority_final_text_over_retry():
    """测试：has_final_text 优先级高于 _retry_think 时，即使两者都为True也走OUTPUT"""
    graph = StateGraph()

    async def think_handler(ctx):
        return {
            "has_final_text": True,
            "has_tool_calls": False,
            "_retry_think": True,
            "_final_text": "最终输出",
        }

    output_called = False

    async def output_handler(ctx):
        nonlocal output_called
        output_called = True
        return {"is_complete": True, "_retry_think": False}

    graph.add_node("think", NodeKind.THINK, think_handler)
    graph.add_node("output", NodeKind.OUTPUT, output_handler)
    graph.set_entry("think")
    graph.add_edge("think", "act", condition="has_tool_calls", priority=3)
    graph.add_edge("think", "output", condition="has_final_text", priority=2)
    graph.add_edge("think", "think", condition="_retry_think", priority=1)

    result = await graph.run({})

    assert output_called is True, "期望 OUTPUT 节点被调用"
    assert result.get("is_complete") is True, "期望 is_complete=True"
    print("✅ test_edge_priority_final_text_over_retry 通过")


async def test_cognitive_runtime_strip_fabricated_content():
    """测试：_strip_fabricated_content 方法能正确剥离幻觉内容"""
    runtime = CognitiveRuntime(
        llm_chat_fn=AsyncMock(),
        llm_chat_with_tools_fn=AsyncMock(),
        tool_execute_fn=AsyncMock(),
        output_fn=AsyncMock(),
    )

    hallucinated_text = """以下是树排序的实现：

```python
def tree_sort(arr):
    # 树排序实现
    pass
```

测试结果：
- 输入: [3, 1, 4, 1, 5]
- 输出: [1, 1, 3, 4, 5]
- 测试通过！"""

    cleaned = runtime._strip_fabricated_content(hallucinated_text)

    assert "测试结果" not in cleaned, f"剥离后不应包含'测试结果', 实际: {cleaned[:200]}"
    assert "测试通过" not in cleaned, f"剥离后不应包含'测试通过', 实际: {cleaned[:200]}"
    assert "⚠️" in cleaned, f"剥离后应包含警告标记, 实际: {cleaned[:200]}"
    print("✅ test_cognitive_runtime_strip_fabricated_content 通过")


async def test_cognitive_runtime_think_needs_code_no_tools():
    """测试：THINK节点检测到需要代码但LLM未调用工具时，触发重试"""
    call_count = 0

    async def mock_llm_chat_with_tools(messages, tools_list, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return _make_mock_response(content="这是树排序的测试结果：排序成功！\n```python\ndef tree_sort(arr): pass\n```")
        return _make_mock_response(
            tool_calls=[_make_tool_call("write_file", {"path": "output/tree_sort.py", "content": "code"})]
        )

    async def mock_tool_execute(name, args):
        if name == "write_file":
            return "✅ 文件已保存"
        if name == "execute_file":
            return "执行完成: [1, 1, 3, 4, 5]"
        return f"工具 {name} 执行成功"

    runtime = CognitiveRuntime(
        llm_chat_fn=AsyncMock(),
        llm_chat_with_tools_fn=mock_llm_chat_with_tools,
        tool_execute_fn=mock_tool_execute,
        output_fn=AsyncMock(),
    )

    context = CognitiveContext(
        user_message="树排序",
        messages=[
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "树排序"},
        ],
        max_rounds=8,
    )

    result = await runtime.run(context, extra={"_tools": []})

    assert call_count >= 2, f"期望 LLM 被调用>=2次(重试), 实际={call_count}"
    print(f"✅ test_cognitive_runtime_think_needs_code_no_tools 通过 (LLM调用{call_count}次)")


async def test_cognitive_runtime_max_rounds_exhausted():
    """测试：轮次耗尽时不会放行幻觉内容"""
    call_count = 0

    async def mock_llm_chat_with_tools(messages, tools_list, **kwargs):
        nonlocal call_count
        call_count += 1
        return _make_mock_response(
            content="测试结果：排序成功！\n```python\ndef tree_sort(arr): pass\n```"
        )

    async def mock_tool_execute(name, args):
        if name == "write_file":
            return "✅ 文件已保存"
        if name == "execute_code":
            return "执行完成: [1, 1, 3, 4, 5]"
        return f"工具 {name} 执行成功"

    runtime = CognitiveRuntime(
        llm_chat_fn=AsyncMock(),
        llm_chat_with_tools_fn=mock_llm_chat_with_tools,
        tool_execute_fn=mock_tool_execute,
        output_fn=AsyncMock(),
    )

    context = CognitiveContext(
        user_message="树排序",
        messages=[
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "树排序"},
        ],
        max_rounds=3,
    )

    result = await runtime.run(context, extra={"_tools": []})

    final_output = result.final_output
    has_fabricated = any(p in final_output for p in ("测试结果", "测试通过", "排序成功"))
    assert not has_fabricated, f"轮次耗尽后不应包含幻觉内容, 实际: {final_output[:300]}"
    print(f"✅ test_cognitive_runtime_max_rounds_exhausted 通过 (输出: {final_output[:100]}...)")


async def test_final_text_propagation():
    """测试：_final_text 在节点间正确传播"""
    graph = StateGraph()

    async def think_handler(ctx):
        return {
            "has_final_text": True,
            "has_tool_calls": False,
            "_retry_think": False,
            "_final_text": "这是正确的输出",
        }

    received_final_text = None

    async def output_handler(ctx):
        nonlocal received_final_text
        received_final_text = ctx.get("_final_text", "")
        return {"is_complete": True, "_retry_think": False}

    graph.add_node("think", NodeKind.THINK, think_handler)
    graph.add_node("output", NodeKind.OUTPUT, output_handler)
    graph.set_entry("think")
    graph.add_edge("think", "output", condition="has_final_text", priority=2)
    graph.add_edge("think", "think", condition="_retry_think", priority=1)

    result = await graph.run({})

    assert received_final_text == "这是正确的输出", f"期望 _final_text='这是正确的输出', 实际='{received_final_text}'"
    assert result.get("_final_text") == "这是正确的输出", f"上下文中 _final_text 也应正确传播"
    print("✅ test_final_text_propagation 通过")


async def main():
    tests = [
        test_retry_think_flag_cleared_on_success,
        test_retry_think_flag_not_cleared_causes_loop,
        test_edge_priority_final_text_over_retry,
        test_cognitive_runtime_strip_fabricated_content,
        test_cognitive_runtime_think_needs_code_no_tools,
        test_cognitive_runtime_max_rounds_exhausted,
        test_final_text_propagation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} 失败: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} 异常: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"测试结果: {passed} 通过, {failed} 失败, 共 {len(tests)} 个")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
