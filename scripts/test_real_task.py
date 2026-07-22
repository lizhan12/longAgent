"""端到端真实任务测试 — 直接调用 LLM + 工具执行"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from long.cli import LongSystem, load_dotenv


async def test_real_task():
    load_dotenv()
    system = LongSystem(config_dir="configs", workspace_root="./workspace")
    system.initialize()

    llm = system.llm
    if llm is None:
        print("LLM 未初始化，跳过测试")
        return False

    api_key = llm.config.resolve_api_key()
    if not api_key:
        print("API Key 未配置，请在 .env 中设置 LLM_API_KEY")
        return False

    print(f"LLM: model={llm.config.model}, base_url={llm.config.resolve_base_url()}")

    tools = system._gather_tools()
    cleaned_tools = system._clean_tools_for_api(tools)
    print(f"可用工具: {len(cleaned_tools)} 个")

    tasks = [
        ("简单", "今天杭州天气怎么样"),
        ("中等", "搜索 Python 3.13 新特性，用3个要点总结"),
        ("中等", "帮我写一个 Python 快速排序函数"),
    ]

    results = []

    for category, task in tasks:
        print(f"\n{'='*60}")
        print(f"[{category}] 任务: {task}")
        print(f"{'='*60}")

        start = time.time()
        try:
            messages = [
                {"role": "system", "content": system._build_system_prompt()},
                {"role": "user", "content": task},
            ]

            final_response = await _run_agent_loop(system, llm, messages, cleaned_tools, max_rounds=8)
            elapsed = time.time() - start

            result_text = final_response or "(无响应)"
            success = bool(final_response and len(result_text) > 10)

            print(f"\n响应 ({elapsed:.1f}s):")
            print(result_text[:800])
            if len(result_text) > 800:
                print(f"... (共 {len(result_text)} 字符)")

            results.append({
                "task": task,
                "category": category,
                "success": success,
                "elapsed": elapsed,
                "response_len": len(result_text),
            })

        except asyncio.TimeoutError:
            elapsed = time.time() - start
            print(f"\n超时 ({elapsed:.1f}s)")
            results.append({"task": task, "category": category, "success": False, "elapsed": elapsed, "response_len": 0, "error": "timeout"})
        except Exception as e:
            elapsed = time.time() - start
            print(f"\n错误: {type(e).__name__}: {e}")
            results.append({"task": task, "category": category, "success": False, "elapsed": elapsed, "response_len": 0, "error": str(e)})

    print(f"\n\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")

    total = len(results)
    success = sum(1 for r in results if r["success"])
    avg_time = sum(r["elapsed"] for r in results) / total if total > 0 else 0

    for r in results:
        status = "✅" if r["success"] else "❌"
        error = f" ({r.get('error', '')})" if not r["success"] else ""
        print(f"  {status} [{r['category']}] {r['task'][:35]}: {r['elapsed']:.1f}s, {r['response_len']}字符{error}")

    print(f"\n成功率: {success}/{total} ({success/total:.0%})")
    print(f"平均耗时: {avg_time:.1f}s")

    system.shutdown()
    return success == total


async def _run_agent_loop(
    system: LongSystem,
    llm,
    messages: list[dict],
    tools: list[dict],
    max_rounds: int = 8,
) -> str:
    """简化的 Agent 循环 — 验证 LLM + 工具调用链路"""
    from long.llm.base import LLMMessage

    search_count = 0
    max_search = 3

    for round_idx in range(max_rounds):
        llm_messages = [
            LLMMessage(role=m["role"], content=m.get("content"))
            for m in messages
            if m["role"] in ("system", "user", "assistant")
        ]

        response = await llm.chat_with_tools(
            messages=llm_messages,
            tools=tools,
            temperature=0.7,
        )

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
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)},
                }
                for tc in tool_calls_formatted
            ],
        })

        for tc in tool_calls_formatted:
            tool_name = tc["name"]
            tool_args = tc["arguments"]

            if "search" in tool_name.lower():
                search_count += 1
                if search_count > max_search:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "搜索次数已达上限，请基于已有信息生成回复。",
                    })
                    continue

            print(f"  调用工具: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:80]})")

            try:
                result = await system._execute_tool(tool_name, tool_args)
                if len(result) > 500:
                    print(f"  结果: {result[:200]}... (共 {len(result)} 字符)")
                else:
                    print(f"  结果: {result[:200]}")
            except Exception as e:
                result = f"工具执行错误: {type(e).__name__}: {e}"
                print(f"  错误: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result[:1500] if len(result) > 1500 else result,
            })

    return "未能完成任务（达到最大轮次限制）"


if __name__ == "__main__":
    success = asyncio.run(test_real_task())
    sys.exit(0 if success else 1)
