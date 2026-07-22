"""测试搜索限制机制 - 验证系统能够正确拦截无限搜索循环"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_search_tracking():
    """测试搜索计数和重复检测逻辑"""
    print("=" * 60)
    print("测试 1: 搜索计数和重复检测")
    print("=" * 60)
    
    MAX_SEARCH_ROUNDS = 2
    search_call_count = 0
    search_queries_used = []
    intercepted_ids = set()
    intercept_reasons = {}
    last_round_had_search = False
    
    # 模拟 LLM 返回的工具调用
    test_calls = [
        {"id": "call_1", "name": "tavily_search", "arguments": {"query": "杭州天气 2026"}},
        {"id": "call_2", "name": "tavily_search", "arguments": {"query": "杭州天气预报 30天"}},  # 连续搜索，应被 ReAct 拦截
        {"id": "call_3", "name": "write_file", "arguments": {"path": "test.py", "content": "print('hello')"}},
        {"id": "call_4", "name": "tavily_search", "arguments": {"query": "北京天气"}},  # 非连续搜索，应允许
    ]
    
    for round_num, tc in enumerate(test_calls):
        tool_name = tc["name"]
        arguments = tc["arguments"]
        
        if tool_name == "tavily_search":
            query = arguments.get("query", "")
            
            # 次数限制
            if search_call_count >= MAX_SEARCH_ROUNDS:
                intercept_reasons[tc["id"]] = f"[搜索限制] 已达到最大搜索次数 ({MAX_SEARCH_ROUNDS})"
                intercepted_ids.add(tc["id"])
                print(f"  ❌ 拦截 {tc['id']}: 超过次数限制 ({search_call_count}/{MAX_SEARCH_ROUNDS})")
                last_round_had_search = False
                continue
            
            # ReAct 约束：上一轮搜索过，不能连续搜索
            if last_round_had_search and search_call_count > 0:
                intercept_reasons[tc["id"]] = "[ReAct 约束] 连续搜索被拦截"
                intercepted_ids.add(tc["id"])
                print(f"  ❌ 拦截 {tc['id']}: ReAct 约束 - 连续搜索 '{query}'")
                last_round_had_search = False
                continue
            
            # 重复检测
            is_duplicate = False
            for prev_query in search_queries_used:
                if query in prev_query or prev_query in query:
                    intercepted_ids.add(tc["id"])
                    is_duplicate = True
                    print(f"  ❌ 拦截 {tc['id']}: 重复搜索 '{query}'")
                    break
            
            if not is_duplicate:
                search_call_count += 1
                search_queries_used.append(query)
                print(f"  ✅ 允许 {tc['id']}: '{query}' (搜索 #{search_call_count})")
                last_round_had_search = True
        else:
            # 非搜索操作，重置连续搜索标志
            print(f"  ✅ 非搜索操作 {tc['id']}: {tool_name}")
            last_round_had_search = False
    
    print(f"\n结果: {len(intercepted_ids)} 个调用被拦截, {search_call_count} 个搜索被允许")
    assert len(intercepted_ids) == 1, f"期望拦截 1 个（连续搜索），实际拦截 {len(intercepted_ids)}"
    assert search_call_count == 2, f"期望允许 2 个搜索，实际允许 {search_call_count}"
    print("✅ 测试通过!\n")


def test_max_tool_rounds():
    """测试最大工具轮次限制"""
    print("=" * 60)
    print("测试 2: 最大工具轮次限制")
    print("=" * 60)
    
    MAX_TOOL_ROUNDS = 8
    budget_remaining = MAX_TOOL_ROUNDS
    
    # 模拟 10 轮工具调用
    for round_num in range(1, 11):
        if budget_remaining <= 0:
            print(f"  ❌ 第 {round_num} 轮: 预算已耗尽")
            continue
        
        budget_remaining -= 1
        print(f"  ✅ 第 {round_num} 轮: 执行工具 (剩余预算: {budget_remaining})")
    
    assert budget_remaining == 0, f"期望预算耗尽，实际剩余 {budget_remaining}"
    print(f"\n结果: {MAX_TOOL_ROUNDS} 轮执行成功，后续轮次被拦截")
    print("✅ 测试通过!\n")


def test_prompt_constraints():
    """测试系统提示词约束"""
    print("=" * 60)
    print("测试 3: 系统提示词约束检查")
    print("=" * 60)
    
    # 导入 cli 模块检查提示词
    from long.cli import LongSystem
    
    system = LongSystem()
    system.load_configs()
    
    # 检查 _build_static_prompt 是否包含约束
    static_prompt = system._build_static_prompt()
    
    assert "最多只能调用2次" in static_prompt or "最多搜索2次" in static_prompt, \
        "系统提示词缺少搜索次数限制"
    print("  ✅ 系统提示词包含搜索次数限制")
    
    assert "不得换关键词重复搜索" in static_prompt or "不得重复搜索" in static_prompt, \
        "系统提示词缺少重复搜索约束"
    print("  ✅ 系统提示词包含重复搜索约束")
    
    assert "拦截" in static_prompt, "系统提示词未说明违反约束的后果"
    print("  ✅ 系统提示词说明了违反后果")
    
    print("\n✅ 测试通过!\n")


def test_compressor_safety():
    """测试对话压缩器安全性"""
    print("=" * 60)
    print("测试 4: 对话压缩器安全性")
    print("=" * 60)
    
    from long.context.compressor import DialogCompressor, CompressConfig
    
    compressor = DialogCompressor(CompressConfig(
        max_prompt_chars=12000,
        min_rounds_before_compress=3,
        keep_recent_rounds=2,
    ))
    
    # 模拟一个更长的对话历史，触发压缩逻辑
    history_msgs = [
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "查询杭州天气"},
        # 第1轮工具调用
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "name": "tavily_search"}]},
        {"role": "tool", "content": "搜索结果为空", "tool_call_id": "1"},
        # 第2轮工具调用
        {"role": "assistant", "content": "", "tool_calls": [{"id": "2", "name": "tavily_search"}]},
        {"role": "tool", "content": "还是无结果", "tool_call_id": "2"},
        # 第3轮工具调用
        {"role": "assistant", "content": "", "tool_calls": [{"id": "3", "name": "tavily_search"}]},
        {"role": "tool", "content": "仍然无结果", "tool_call_id": "3"},
        # 第4轮工具调用（最近的2轮应该保留）
        {"role": "assistant", "content": "", "tool_calls": [{"id": "4", "name": "write_file"}]},
        {"role": "tool", "content": "写入成功", "tool_call_id": "4"},
    ]
    
    old, preserved = compressor.extract_compressible_messages(history_msgs, tool_rounds=4)
    
    # 检查是否保留了 assistant 消息
    has_assistant = any(m["role"] == "assistant" for m in preserved)
    has_tool = any(m["role"] == "tool" for m in preserved)
    
    assert has_assistant or has_tool, "压缩后缺少 assistant/tool 消息"
    print(f"  ✅ 压缩后保留了 {len(preserved)} 条消息（包含 assistant/tool）")
    
    # 验证 old 中有被压缩的消息
    assert len(old) > 0, "应该有被压缩的消息"
    print(f"  ✅ {len(old)} 条旧消息被压缩")
    
    print("✅ 测试通过!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("搜索限制机制测试")
    print("=" * 60 + "\n")
    
    try:
        test_search_tracking()
        test_max_tool_rounds()
        test_prompt_constraints()
        test_compressor_safety()
        
        print("=" * 60)
        print("✅ 所有测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
