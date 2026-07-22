#!/usr/bin/env python3
"""
对比测试：框架式调用 vs 直接 SDK 调用

假设验证：
1. AsyncOpenAI + httpx.AsyncClient（连接池） vs OpenAI（同步）
2. 连续多轮 tool-calling（5+ 轮） vs 2 轮
3. 请求间有间隔（模拟工具执行耗时）

用法:
    uv run python scripts/test_tool_call_round2.py
"""

import asyncio
import json
import os
import sys
import time

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from openai import AsyncOpenAI

load_dotenv()

API_KEY = os.environ.get("LLM_API_KEY", "")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
MODEL = "Pro/zai-org/GLM-5"

if not API_KEY:
    print("❌ LLM_API_KEY 未设置")
    sys.exit(1)

# ── 工具定义（6 个，和框架一致） ──────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": "使用 Tavily API 搜索网络获取最新信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "max_results": {"type": "integer", "description": "最大返回结果数", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "在沙箱中执行代码并返回结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的代码"},
                    "language": {"type": "string", "description": "语言(python/javascript/bash)", "default": "python"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取精确的当前日期和时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_PROMPT = """你是 Long，一个智能 AI 助手，可以调用工具和技能来帮助用户完成任务。

## 当前时间
2026年05月16日 星期六 16:30

## 重要：工具调用效率规则
- tavily_search 一次就够了。
"""

USER_MESSAGE = "余杭区 未来 7天的预报，带有折线图，并生成报告"

# ── 模拟搜索结果 ───────────────────────────────────────────────────────
SEARCH_RESULTS = [
    "1. 余杭区未来30天天气预报 - 和风天气\nhttps://www.qweather.com/weather30d/yuhang-101210106.html\n- 05/16 多云 22~30℃, 05/17 阴转小雨 20~28℃, 05/18 小雨 19~26℃, 05/19 多云 21~29℃, 05/20 晴 22~31℃, 05/21 晴转多云 23~32℃, 05/22 多云 22~30℃",
    "2. 余杭区7天天气预报 - 天气网\nhttps://www.tianqi.com/yuhang/7/\n- 05/16(六) 多云 21~29℃, 05/17(日) 小雨转阴 18~26℃, 05/18(一) 阵雨 17~25℃, 05/19(二) 阴转多云 20~28℃, 05/20(三) 晴 21~30℃",
]


def print_delim(title=""):
    print(f"\n{'='*70}")
    if title:
        print(f"  {title}")
        print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
#  测试 A: 同步 SDK（原始脚本的方式）
# ═══════════════════════════════════════════════════════════════════════
def test_sync_sdk():
    print_delim("测试 A: 同步 OpenAI SDK（无连接池，每次新建连接）")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    results = []

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_MESSAGE},
    ]

    for round_num in range(1, 8):  # 最多 7 轮
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
            "tools": TOOLS,
            "tool_choice": "auto",
            "stream": True,
        }

        t0 = time.monotonic()
        try:
            stream_obj = client.chat.completions.create(**kwargs)
            first_chunk = True
            tool_calls = []
            content = ""

            for chunk in stream_obj:
                if first_chunk:
                    first_t = time.monotonic() - t0
                    first_chunk = False
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    content += delta.content
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.function and tc.function.name:
                            if len(tool_calls) <= tc.index:
                                tool_calls.append({"name": tc.function.name, "args": ""})
                            tool_calls[tc.index]["args"] += tc.function.arguments or ""

            total_t = time.monotonic() - t0
            status = f"✅ OK ({total_t:.1f}s, first_token={first_t:.1f}s)"

            if tool_calls:
                # 模拟 tool 结果追加
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": f"call_{round_num}_{i}", "type": "function", "function": {"name": tc["name"], "arguments": tc["args"]}} for i, tc in enumerate(tool_calls)],
                })
                for i, tc in enumerate(tool_calls):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": f"call_{round_num}_{i}",
                        "content": SEARCH_RESULTS[i % len(SEARCH_RESULTS)],
                    })

                print(f"  Round {round_num}: {status} msgs={len(messages)} tool_calls={len(tool_calls)}")
                for tc in tool_calls:
                    print(f"    → {tc['name']}: {tc['args'][:60]}")
                results.append(("OK", total_t))
            else:
                # LLM 返回了文本回复，任务完成
                print(f"  Round {round_num}: {status} msgs={len(messages)} TEXT: {content[:100]}...")
                results.append(("OK", total_t))
                break  # 任务完成，停止循环

        except Exception as e:
            total_t = time.monotonic() - t0
            status = f"❌ {type(e).__name__}: {str(e)[:80]}"
            print(f"  Round {round_num}: {status}")
            results.append(("FAIL", total_t))

        # 模拟工具执行间隔（框架的真实情况）
        time.sleep(2)

    ok_count = sum(1 for s, _ in results if s == "OK")
    print(f"\n  结果: {ok_count}/5 成功")
    return results


# ═══════════════════════════════════════════════════════════════════════
#  测试 B: 异步 SDK + 框架式 httpx.AsyncClient（有连接池）
# ═══════════════════════════════════════════════════════════════════════
async def test_async_sdk_with_pool():
    print_delim("测试 B: 异步 OpenAI SDK + httpx.AsyncClient（连接池，和框架一致）")

    # 使用和框架完全相同的 httpx.AsyncClient 配置
    http_client = httpx.AsyncClient(
        http2=False,
        timeout=httpx.Timeout(
            connect=15.0,
            read=180.0,
            write=30.0,
            pool=30.0,
        ),
        limits=httpx.Limits(
            max_connections=10,
            max_keepalive_connections=5,
            keepalive_expiry=120.0,
        ),
    )

    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        max_retries=0,
        http_client=http_client,
    )

    results = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_MESSAGE},
    ]

    for round_num in range(1, 6):
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
            "tools": TOOLS,
            "tool_choice": "auto",
            "stream": True,
        }

        t0 = time.monotonic()
        try:
            stream = await asyncio.wait_for(
                client.chat.completions.create(**kwargs),
                timeout=180.0,
            )

            tool_calls = []
            content = ""
            first_t = None
            first_chunk = True

            async for chunk in stream:
                if first_chunk:
                    first_t = time.monotonic() - t0
                    first_chunk = False
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    content += delta.content
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.function and tc.function.name:
                            if len(tool_calls) <= tc.index:
                                tool_calls.append({"name": tc.function.name, "args": ""})
                            tool_calls[tc.index]["args"] += tc.function.arguments or ""

            total_t = time.monotonic() - t0
            status = f"✅ OK ({total_t:.1f}s, first_token={first_t:.1f}s)"

            # 模拟 tool 结果追加
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"call_{round_num}_{i}", "type": "function", "function": {"name": tc["name"], "arguments": tc["args"]}} for i, tc in enumerate(tool_calls)],
            })
            for i, tc in enumerate(tool_calls):
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{round_num}_{i}",
                    "content": SEARCH_RESULTS[i % len(SEARCH_RESULTS)],
                })

            print(f"  Round {round_num}: {status} msgs={len(messages)} tool_calls={len(tool_calls)}")
            for tc in tool_calls:
                print(f"    → {tc['name']}: {tc['args'][:60]}...")
            results.append(("OK", total_t))

        except asyncio.TimeoutError:
            total_t = time.monotonic() - t0
            status = f"❌ TIMEOUT ({total_t:.1f}s)"
            print(f"  Round {round_num}: {status}")
            results.append(("TIMEOUT", total_t))

        except Exception as e:
            total_t = time.monotonic() - t0
            status = f"❌ {type(e).__name__}: {str(e)[:80]}"
            print(f"  Round {round_num}: {status}")
            results.append(("FAIL", total_t))

        # 模拟工具执行间隔
        await asyncio.sleep(2)

    await http_client.aclose()
    ok_count = sum(1 for s, _ in results if s == "OK")
    print(f"\n  结果: {ok_count}/5 成功")
    return results


# ═══════════════════════════════════════════════════════════════════════
#  测试 C: 异步 SDK + 每次新建连接（无连接池复用）
# ═══════════════════════════════════════════════════════════════════════
async def test_async_sdk_no_pool():
    print_delim("测试 C: 异步 OpenAI SDK + 每次新建 httpx.AsyncClient（无连接池复用）")

    results = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_MESSAGE},
    ]

    for round_num in range(1, 6):
        # 每次创建新的 client（模拟同步行为）
        http_client = httpx.AsyncClient(
            http2=False,
            timeout=httpx.Timeout(
                connect=15.0,
                read=180.0,
                write=30.0,
                pool=30.0,
            ),
        )

        client = AsyncOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            max_retries=0,
            http_client=http_client,
        )

        kwargs = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
            "tools": TOOLS,
            "tool_choice": "auto",
            "stream": True,
        }

        t0 = time.monotonic()
        try:
            stream = await asyncio.wait_for(
                client.chat.completions.create(**kwargs),
                timeout=180.0,
            )

            tool_calls = []
            content = ""
            first_t = None

            async for chunk in stream:
                if first_t is None:
                    first_t = time.monotonic() - t0
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    content += delta.content
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.function and tc.function.name:
                            if len(tool_calls) <= tc.index:
                                tool_calls.append({"name": tc.function.name, "args": ""})
                            tool_calls[tc.index]["args"] += tc.function.arguments or ""

            total_t = time.monotonic() - t0
            status = f"✅ OK ({total_t:.1f}s, first_token={first_t:.1f}s)"

            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"call_{round_num}_{i}", "type": "function", "function": {"name": tc["name"], "arguments": tc["args"]}} for i, tc in enumerate(tool_calls)],
            })
            for i, tc in enumerate(tool_calls):
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{round_num}_{i}",
                    "content": SEARCH_RESULTS[i % len(SEARCH_RESULTS)],
                })

            print(f"  Round {round_num}: {status} msgs={len(messages)} tool_calls={len(tool_calls)}")
            for tc in tool_calls:
                print(f"    → {tc['name']}: {tc['args'][:60]}...")
            results.append(("OK", total_t))

        except asyncio.TimeoutError:
            total_t = time.monotonic() - t0
            status = f"❌ TIMEOUT ({total_t:.1f}s)"
            print(f"  Round {round_num}: {status}")
            results.append(("TIMEOUT", total_t))

        except Exception as e:
            total_t = time.monotonic() - t0
            status = f"❌ {type(e).__name__}: {str(e)[:80]}"
            print(f"  Round {round_num}: {status}")
            results.append(("FAIL", total_t))

        await http_client.aclose()
        await asyncio.sleep(2)

    ok_count = sum(1 for s, _ in results if s == "OK")
    print(f"\n  结果: {ok_count}/5 成功")
    return results


# ═══════════════════════════════════════════════════════════════════════
#  测试 D: 异步 SDK + 连接池，但强制关闭 keepalive
# ═══════════════════════════════════════════════════════════════════════
async def test_async_sdk_no_keepalive():
    print_delim("测试 D: 异步 SDK + 连接池（keepalive=0s，不复用连接）")

    http_client = httpx.AsyncClient(
        http2=False,
        timeout=httpx.Timeout(
            connect=15.0,
            read=180.0,
            write=30.0,
            pool=30.0,
        ),
        limits=httpx.Limits(
            max_connections=10,
            max_keepalive_connections=5,
            keepalive_expiry=0.0,  # ← 禁用 keepalive
        ),
    )

    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        max_retries=0,
        http_client=http_client,
    )

    results = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_MESSAGE},
    ]

    for round_num in range(1, 6):
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
            "tools": TOOLS,
            "tool_choice": "auto",
            "stream": True,
        }

        t0 = time.monotonic()
        try:
            stream = await asyncio.wait_for(
                client.chat.completions.create(**kwargs),
                timeout=180.0,
            )

            tool_calls = []
            first_t = None

            async for chunk in stream:
                if first_t is None:
                    first_t = time.monotonic() - t0
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    content = ""  # not needed
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.function and tc.function.name:
                            if len(tool_calls) <= tc.index:
                                tool_calls.append({"name": tc.function.name, "args": ""})
                            tool_calls[tc.index]["args"] += tc.function.arguments or ""

            total_t = time.monotonic() - t0
            status = f"✅ OK ({total_t:.1f}s, first_token={first_t:.1f}s)"

            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"call_{round_num}_{i}", "type": "function", "function": {"name": tc["name"], "arguments": tc["args"]}} for i, tc in enumerate(tool_calls)],
            })
            for i, tc in enumerate(tool_calls):
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{round_num}_{i}",
                    "content": SEARCH_RESULTS[i % len(SEARCH_RESULTS)],
                })

            print(f"  Round {round_num}: {status} msgs={len(messages)} tool_calls={len(tool_calls)}")
            for tc in tool_calls:
                print(f"    → {tc['name']}: {tc['args'][:60]}...")
            results.append(("OK", total_t))

        except asyncio.TimeoutError:
            total_t = time.monotonic() - t0
            status = f"❌ TIMEOUT ({total_t:.1f}s)"
            print(f"  Round {round_num}: {status}")
            results.append(("TIMEOUT", total_t))

        except Exception as e:
            total_t = time.monotonic() - t0
            status = f"❌ {type(e).__name__}: {str(e)[:80]}"
            print(f"  Round {round_num}: {status}")
            results.append(("FAIL", total_t))

        await asyncio.sleep(2)

    await http_client.aclose()
    ok_count = sum(1 for s, _ in results if s == "OK")
    print(f"\n  结果: {ok_count}/5 成功")
    return results


# ═══════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════
async def main():
    print(f"  Model: {MODEL}")
    print(f"  Base URL: {BASE_URL}")
    print(f"  Tools: {len(TOOLS)} 个")
    print(f"  每轮间隔: 2s（模拟工具执行耗时）")

    # 测试 A: 同步 SDK
    results_a = test_sync_sdk()

    # 测试 B: 异步 + 连接池
    results_b = await test_async_sdk_with_pool()

    # 测试 C: 异步 + 每次新建连接
    results_c = await test_async_sdk_no_pool()

    # 测试 D: 异步 + 连接池但禁用 keepalive
    results_d = await test_async_sdk_no_keepalive()

    # ── 总结 ─────────────────────────────────────────────────────
    print_delim("对比总结")
    print(f"  {'测试':<30} {'结果':<10}")
    print(f"  {'─' * 40}")
    print(f"  {'A: 同步 SDK（无连接池）':<30} {'{}/5'.format(sum(1 for s, _ in results_a if s == 'OK')):<10}")
    print(f"  {'B: 异步 + 连接池':<30} {'{}/5'.format(sum(1 for s, _ in results_b if s == 'OK')):<10}")
    print(f"  {'C: 异步 + 每次新建连接':<30} {'{}/5'.format(sum(1 for s, _ in results_c if s == 'OK')):<10}")
    print(f"  {'D: 异步 + 连接池(keepalive=0)':<30} {'{}/5'.format(sum(1 for s, _ in results_d if s == 'OK')):<10}")

    # 分析
    print("\n  分析:")
    if sum(1 for s, _ in results_a if s == "OK") == 5 and sum(1 for s, _ in results_b if s == "OK") < 5:
        print("  → 同步正常但异步超时 → 连接池/keepalive 导致问题")
    elif sum(1 for s, _ in results_b if s == "OK") == 5 and sum(1 for s, _ in results_d if s == "OK") < 5:
        print("  → keepalive 正常但禁用后超时 → 不是连接池问题")
    elif sum(1 for s, _ in results_c if s == "OK") == 5:
        print("  → 每次新建连接正常 → 框架的长生命周期 httpx.AsyncClient 有问题")
    else:
        print("  → 需要进一步分析")


if __name__ == "__main__":
    asyncio.run(main())
