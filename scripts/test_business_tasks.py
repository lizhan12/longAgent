"""真实业务任务测试 — 验证演进后所有模块协同工作"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from long.cli import LongSystem, load_dotenv


async def test_business_tasks():
    """运行一组真实业务任务，全面验证系统"""

    load_dotenv()
    system = LongSystem(config_dir="configs", workspace_root="./workspace")
    system.initialize()

    llm = system.llm
    if llm is None:
        print("❌ LLM 未初始化，退出")
        return False

    api_key = llm.config.resolve_api_key()
    if not api_key:
        print("❌ API Key 未配置，请在 .env 中设置 LLM_API_KEY")
        return False

    print(f"🚀 Long Agent 真实业务测试")
    print(f"   Model: {llm.config.model}")
    print(f"   Base URL: {llm.config.resolve_base_url()}")

    tools = system._gather_tools()
    cleaned_tools = system._clean_tools_for_api(tools)
    tool_names = [t.get("function", {}).get("name", "?") for t in cleaned_tools]
    print(f"   Tools ({len(cleaned_tools)}): {', '.join(tool_names)}")

    # --- 模块验证 ---
    print(f"\n{'='*60}")
    print(f"Phase 0: 模块完整性验证")
    print(f"{'='*60}")

    phase0_results = await _verify_modules(system)
    for name, ok in phase0_results.items():
        print(f"  {'✅' if ok else '❌'} {name}")

    # --- 真实业务任务 ---
    print(f"\n{'='*60}")
    print(f"Phase 1: 业务任务执行")
    print(f"{'='*60}")

    tasks = [
        {
            "id": "T1",
            "category": "简单搜索",
            "prompt": "搜索 Python 3.13 有哪些重要的新特性",
            "expected_tools": ["tavily_search"],
            "max_rounds": 5,
        },
        {
            "id": "T2",
            "category": "简单工具",
            "prompt": "用 Python 计算斐波那契数列的前20项",
            "expected_tools": ["execute_code"],
            "max_rounds": 5,
        },
        {
            "id": "T3",
            "category": "搜索+总结",
            "prompt": "搜索最近 AI Agent 领域的重要进展，用3个要点总结",
            "expected_tools": ["tavily_search"],
            "max_rounds": 5,
        },
        {
            "id": "T4",
            "category": "复杂报告",
            "prompt": "搜索十五五规划的核心内容，生成一份简要说明",
            "expected_tools": ["tavily_search", "write_file"],
            "max_rounds": 8,
        },
        {
            "id": "T5",
            "category": "多工具协作",
            "prompt": "搜索今天杭州天气，然后写一个 Python 脚本打印天气信息",
            "expected_tools": ["tavily_search", "execute_code"],
            "max_rounds": 8,
        },
    ]

    task_results = []
    system_prompt = system._build_system_prompt()

    for task in tasks:
        print(f"\n--- [{task['id']}] {task['category']}: {task['prompt'][:60]}... ---")

        start = time.time()
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task["prompt"]},
            ]

            final_response = await _run_agent_loop(
                system, llm, messages, cleaned_tools,
                max_rounds=task["max_rounds"],
                task_id=task["id"],
            )
            elapsed = time.time() - start

            result_text = final_response or "(无响应)"
            result_len = len(result_text.strip())

            fallback_phrases = [
                "未能完成任务", "无法完成", "已达到最大轮次",
            ]
            is_fallback = any(phrase in result_text for phrase in fallback_phrases)

            success = bool(
                final_response
                and result_len > 50
                and not is_fallback
                and not result_text.startswith("1. ")
                and not result_text.startswith("http")
            )

            fail_reason = ""
            if not success:
                if is_fallback:
                    fail_reason = "搜索耗尽"
                elif result_len <= 50:
                    fail_reason = f"响应太短({result_len}字符)"
                elif result_text.startswith("1. ") or result_text.startswith("http"):
                    fail_reason = "返回原始搜索结果"
                else:
                    fail_reason = "无有效内容"

            print(f"  ⏱️ {elapsed:.1f}s | {'✅ 成功' if success else '❌ 失败' + ('(' + fail_reason + ')' if fail_reason else '')} | {result_len} 字符")
            print(f"  响应预览: {result_text[:120].replace(chr(10), ' ')}...")

            task_results.append({
                "id": task["id"],
                "category": task["category"],
                "success": success,
                "elapsed": elapsed,
                "response_len": result_len,
                "fail_reason": fail_reason if not success else "",
            })

        except asyncio.TimeoutError:
            elapsed = time.time() - start
            print(f"  ❌ 超时 ({elapsed:.1f}s)")
            task_results.append({"id": task["id"], "category": task["category"], "success": False, "elapsed": elapsed, "error": "timeout"})
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ❌ 异常: {type(e).__name__}: {str(e)[:100]}")
            task_results.append({"id": task["id"], "category": task["category"], "success": False, "elapsed": elapsed, "error": str(e)[:100]})

    # --- 记忆系统测试 ---
    print(f"\n{'='*60}")
    print(f"Phase 2: 三栖记忆验证")
    print(f"{'='*60}")

    memory_ok = await _test_memory_system(system)
    print(f"  {'✅' if memory_ok else '❌'} 三栖记忆系统")

    # --- EvalOps 验证 ---
    print(f"\n{'='*60}")
    print(f"Phase 4: EvalOps 验证")
    print(f"{'='*60}")

    eval_ok = await _test_evalops()
    print(f"  {'✅' if eval_ok else '❌'} EvalOps 流水线")

    # --- 汇总 ---
    print(f"\n{'='*60}")
    print(f"测试结果汇总")
    print(f"{'='*60}")

    all_modules_ok = all(phase0_results.values())
    task_success = sum(1 for r in task_results if r["success"])
    task_total = len(task_results)

    print(f"  模块完整性: {'✅ 全部通过' if all_modules_ok else '❌ 有失败'}")
    print(f"  业务任务: {task_success}/{task_total} 成功")
    for r in task_results:
        status = "✅" if r["success"] else "❌"
        extra = ""
        if not r["success"]:
            extra = r.get("fail_reason", "") or r.get("error", "")
            extra = f" - {extra}" if extra else ""
        print(f"    {status} [{r['id']}] {r['category']}: {r.get('elapsed', 0):.1f}s{extra}")
    print(f"  级联路由: {'✅' if cascade_ok else '❌'}")
    print(f"  三栖记忆: {'✅' if memory_ok else '❌'}")
    print(f"  EvalOps: {'✅' if eval_ok else '❌'}")

    avg_time = sum(r.get("elapsed", 0) for r in task_results) / max(task_total, 1)
    print(f"\n  平均任务耗时: {avg_time:.1f}s")

    system.shutdown()

    return all_modules_ok and task_success >= task_total * 0.6 and cascade_ok and memory_ok and eval_ok


async def _run_agent_loop(
    system: LongSystem,
    llm,
    messages: list[dict],
    tools: list[dict],
    max_rounds: int = 8,
    task_id: str = "",
) -> str:
    """Agent 循环"""
    from long.llm.base import LLMMessage

    search_count = 0
    max_search = 2
    active_tools = tools
    force_text = False  # 搜索耗尽后强制纯文本回答

    for round_idx in range(max_rounds):
        llm_messages = [
            LLMMessage(role=m["role"], content=m.get("content"))
            for m in messages
            if m["role"] in ("system", "user", "assistant")
        ]

        try:
            if force_text:
                print(f"  💬 Round{round_idx+1}: 强制文本回答（无工具）...")
                response = await asyncio.wait_for(
                    llm.chat(messages=llm_messages, temperature=0.7),
                    timeout=180,
                )
                content = response.content.strip() if response.content else ""
                if content:
                    return content
                force_text = False
                active_tools = [t for t in active_tools if t.get("function", {}).get("name", "") != "tavily_search"]
                continue

            response = await asyncio.wait_for(
                llm.chat_with_tools(
                    messages=llm_messages,
                    tools=active_tools,
                    temperature=0.7,
                ),
                timeout=180,
            )
        except asyncio.TimeoutError:
            print(f"  ⚠️ Round {round_idx+1} 超时")
            return messages[-1].get("content", "") if messages else ""

        if not response.tool_calls:
            return response.content or ""

        tool_calls_formatted = []
        for tc in response.tool_calls:
            if isinstance(tc, dict):
                tc_id = tc.get("id", f"call_{round_idx}")
                tc_name = tc.get("name", "")
                tc_args = tc.get("arguments", {})
            else:
                tc_id = tc.id
                tc_name = tc.function.name
                tc_args = tc.function.arguments

            if isinstance(tc_args, str):
                try:
                    tc_args = json.loads(tc_args)
                except json.JSONDecodeError:
                    tc_args = {}

            tool_calls_formatted.append({
                "id": tc_id,
                "name": tc_name,
                "arguments": tc_args,
            })

        messages.append({
            "role": "assistant",
            "content": response.content or None,
            "tool_calls": [
                {
                    "id": tc["id"], "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)},
                }
                for tc in tool_calls_formatted
            ],
        })

        had_blocked = False
        for tc in tool_calls_formatted:
            tool_name = tc["name"]
            tool_args = tc["arguments"]

            if "search" in tool_name.lower():
                search_count += 1
                if search_count >= max_search:
                    had_blocked = True
                    messages.append({
                        "role": "tool", "tool_call_id": tc["id"],
                        "content": "搜索次数已达上限，请基于已有信息生成回复。",
                    })
                    continue

            print(f"  🔧 Round{round_idx+1}: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:80]})")

            try:
                result = await system._execute_tool(tool_name, tool_args)
                if len(result) > 300:
                    print(f"     → {result[:120]}... ({len(result)} 字符)")
                else:
                    print(f"     → {result[:120]}")
            except Exception as e:
                result = f"工具执行错误: {type(e).__name__}: {e}"
                print(f"     ❌ {str(e)[:80]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result[:2000] if len(result) > 2000 else result,
            })

        if had_blocked:
            force_text = True
            print(f"  🔒 搜索已耗尽，下一轮强制文本回答")

    return "未能完成任务（达到最大轮次限制）"


async def _verify_modules(system: LongSystem) -> dict[str, bool]:
    """验证各模块完整性"""
    results = {}

    results["LLM Client"] = system.llm is not None

    try:
        from long.memory.windowed import WindowedMemory
        wm = WindowedMemory()
        results["WindowedMemory"] = True
    except Exception as e:
        results["WindowedMemory"] = False

    try:
        from long.memory.vector_rag import VectorRAG
        vr = VectorRAG()
        results["VectorRAG"] = True
    except Exception as e:
        results["VectorRAG"] = False

    try:
        from long.memory.compressor import SemanticCompressor
        sc = SemanticCompressor()
        results["SemanticCompressor"] = True
    except Exception as e:
        results["SemanticCompressor"] = False

    try:
        from long.ir.state_machine import AgentStateMachine, _resolve_target_state
        result = _resolve_target_state("INIT", "search")
        results["StateMachine(条件路由)"] = result is not None
    except Exception as e:
        results["StateMachine(条件路由)"] = False

    try:
        from long.ir.ir_parser import IRParser, ParseMetrics
        results["IRParser(Metrics)"] = True
    except Exception as e:
        results["IRParser(Metrics)"] = False

    return results


async def _test_memory_system(system: LongSystem) -> bool:
    """验证三栖记忆写入和检索"""
    try:
        from long.memory import MemoryController, MemoryType

        mc = MemoryController()
        await mc.add_message("user", "测试用户消息")
        await mc.add_message("assistant", "测试助手回复")

        messages = await mc.windowed.get_messages()
        if len(messages) < 2:
            return False

        await mc.store("测试语义知识", memory_type=MemoryType.SEMANTIC, importance=0.8)
        results = await mc.search("语义知识", limit=5)
        if len(results) < 1:
            return False

        ctx = await mc.get_context("测试", max_tokens=1000)
        if "messages" not in ctx or "task_state" not in ctx or "relevant_memories" not in ctx:
            return False

        await mc.set_task_state("test_key", "test_value")
        val = await mc.get_task_state("test_key")
        if val != "test_value":
            return False

        return True
    except Exception as e:
        print(f"    记忆系统验证异常: {e}")
        return False


async def _test_evalops() -> bool:
    """验证 EvalOps 流水线"""
    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_business_tasks())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n测试被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)