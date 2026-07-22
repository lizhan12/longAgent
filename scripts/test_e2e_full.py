#!/usr/bin/env python3
"""端到端自动化测试 - 模拟完整的天气查询流程"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from long.cli import LongSystem
from long.llm.base import LLMMessage


async def test_full_weather_flow():
    """模拟完整的天气查询流程"""
    print("=" * 60)
    print("端到端测试: 余杭区未来7天天气预报+折线图+报告")
    print("=" * 60)

    system = LongSystem()
    system.initialize()

    tools = system._gather_tools()
    clean_tools = system._clean_tools_for_api(tools)
    system_prompt = system._build_system_prompt()

    print(f"工具数量: {len(clean_tools)}")
    print(f"系统提示词长度: {len(system_prompt)} 字符")

    history_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "余杭区未来7天天气预报，带有折线图，并生成报告"},
    ]

    MAX_ROUNDS = 8
    search_count = 0

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n--- 第 {round_num} 轮 ---")

        llm_messages = [
            LLMMessage(
                role=m["role"],
                content=m.get("content", ""),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
            )
            for m in history_msgs
        ]

        t0 = time.monotonic()
        try:
            response = await asyncio.wait_for(
                system.llm.chat_with_tools(llm_messages, clean_tools, purpose="chat"),
                timeout=150,
            )
        except asyncio.TimeoutError:
            print(f"❌ 第 {round_num} 轮超时")
            break
        except Exception as e:
            print(f"❌ 第 {round_num} 轮失败: {type(e).__name__}: {e}")
            break

        elapsed = time.monotonic() - t0
        print(f"LLM 响应耗时: {elapsed:.1f}s")

        if not response.tool_calls:
            # LLM 返回文本回复
            if response.content:
                print(f"\n✅ 最终回复:\n{response.content[:500]}")
            else:
                print("⚠️ LLM 返回空内容")
            break

        # 执行工具调用
        all_tool_calls = []
        for tc in response.tool_calls:
            all_tool_calls.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"]),
                },
            })

        history_msgs.append({
            "role": "assistant",
            "content": "",
            "tool_calls": all_tool_calls,
        })

        for tc in response.tool_calls:
            tool_name = tc["name"]
            arguments = tc["arguments"]

            # 搜索次数限制
            if tool_name == "tavily_search":
                search_count += 1
                if search_count > 2:
                    print(f"  ⚠️ 搜索次数超限，跳过")
                    history_msgs.append({
                        "role": "tool",
                        "content": "[搜索限制] 已达到最大搜索次数",
                        "tool_call_id": tc["id"],
                    })
                    continue

            print(f"  🔧 执行 {tool_name}({str(arguments)[:80]}...)")

            try:
                tool_result = await system._execute_tool(tool_name, arguments)
                preview = tool_result[:150].replace("\n", " ")
                if len(tool_result) > 150:
                    preview += "..."
                print(f"  ✅ 结果: {preview}")
            except Exception as e:
                tool_result = f"工具执行失败: {e}"
                print(f"  ❌ 失败: {e}")

            history_msgs.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tc["id"],
            })

    # 检查输出文件
    print("\n--- 检查输出文件 ---")
    output_dir = system.workspace.root / "output"
    if output_dir.exists():
        files = list(output_dir.iterdir())
        if files:
            print(f"✅ output/ 目录中有 {len(files)} 个文件:")
            for f in files:
                size = f.stat().st_size
                print(f"  - {f.name} ({size} bytes)")
        else:
            print("⚠️ output/ 目录为空")
    else:
        print("❌ output/ 目录不存在")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(test_full_weather_flow())
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
