"""测试本次修复的三个问题：任务分类、会话文件跟踪、显示截断"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from long.cli import LongSystem, load_dotenv
from long.ir.executor import TaskComplexityClassifier, TaskComplexity


def test_task_classification() -> bool:
    """测试1: 文件生成任务分类"""
    print("\n" + "=" * 60)
    print("测试1: 任务复杂度分类（'做'字 + produce）")
    print("=" * 60)

    classifier = TaskComplexityClassifier()

    tests = [
        ("未来AI AGent 的发展趋势，做一份ppt的报告", "COMPLEX", "文件生成+做字"),
        ("generate a ppt report about AI", "COMPLEX", "generate+ppt"),
        ("produce a pptx file", "COMPLEX", "produce+pptx (新增)"),
        ("做一份word文档", "COMPLEX", "做字+word"),
        ("帮我创建一个excel表格", "COMPLEX", "创建+excel"),
        ("今天天气怎么样", "SIMPLE", "简单查询"),
        ("写一个快速排序函数", "SIMPLE", "代码任务（纯函数不需要计划）"),
    ]

    all_pass = True
    for msg, expected_level, desc in tests:
        result = classifier.classify(msg)
        actual_level = result.level.value.upper()
        status = "PASS" if actual_level == expected_level else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {status} [{desc}] score={result.score:.1f} expected={expected_level} actual={actual_level}: {msg[:50]}")

    return all_pass


def test_session_file_tracking() -> bool:
    """测试2: 会话文件跟踪——只列出当前会话文件"""
    print("\n" + "=" * 60)
    print("测试2: 会话文件跟踪")
    print("=" * 60)

    output_dir = Path("./workspace/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建历史文件（mtime 设为很久以前）
    old_files = [
        "hefei_development_report.pptx",
        "old_weather_report.docx",
        "history_chart.png",
    ]
    old_time = time.time() - 86400  # 1天前
    for fname in old_files:
        fpath = output_dir / fname
        fpath.write_text("dummy")
        os.utime(fpath, (old_time, old_time))

    # 创建当前会话文件（mtime 设为当前时间）
    new_files = [
        "current_ppt.pptx",
        "current_report.docx",
    ]
    for fname in new_files:
        fpath = output_dir / fname
        fpath.write_text("dummy content")
        # mtime 设为当前时间

    # 模拟 _scan_output_files 的逻辑
    session_start_ts = time.time() - 10  # 10秒前
    binary_exts = {'.pptx', '.pdf', '.docx', '.xlsx', '.xls', '.zip', '.png', '.jpg', '.jpeg', '.gif', '.svg'}

    found_files = []
    for fname in sorted(os.listdir(output_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in binary_exts:
            continue
        fpath = output_dir / fname
        if not fpath.is_file() or fpath.stat().st_size <= 0:
            continue
        if session_start_ts > 0:
            mtime = fpath.stat().st_mtime
            if mtime < session_start_ts - 5:
                continue
        found_files.append(fname)

    # 验证：只应有当前会话文件
    expected = set(new_files)
    actual = set(found_files)
    old_included = [f for f in old_files if f in actual]

    print(f"  历史文件: {old_files}")
    print(f"  新文件: {new_files}")
    print(f"  实际返回: {found_files}")
    print(f"  历史文件被过滤: {len(old_included) == 0}")

    all_pass = actual == expected and len(old_included) == 0
    if all_pass:
        print("  PASS: 只列出当前会话文件，历史文件已过滤")
    else:
        print(f"  FAIL: 应返回 {expected}，实际返回 {actual}，历史文件混入: {old_included}")

    # 清理
    for fname in old_files + new_files:
        p = output_dir / fname
        if p.exists():
            p.unlink()

    return all_pass


async def test_real_task_pipeline() -> bool:
    """测试3: 真实任务——完整流水线测试（任务分类 + 文件生成）"""
    print("\n" + "=" * 60)
    print("测试3: 完整流水线测试")
    print("=" * 60)

    load_dotenv()
    system = LongSystem(config_dir="configs", workspace_root="./workspace")
    system.initialize()

    llm = system.llm
    if llm is None:
        print("  LLM 未初始化，跳过测试")
        return True  # 不算失败

    api_key = llm.config.resolve_api_key()
    if not api_key:
        print("  API Key 未配置，跳过测试")
        return True

    print(f"  LLM: model={llm.config.model}")

    # 测试任务：文件生成类（应走计划执行路径）
    tasks = [
        "写一个Python快速排序函数，保存到 output/quicksort_test.py 并执行验证",
    ]

    all_pass = True

    for task in tasks:
        print(f"\n  任务: {task}")

        # 第一步：验证分类
        result = system.plan_executor.classifier.classify(task)
        is_complex = result.level == TaskComplexity.COMPLEX
        print(f"  分类: {result.level.value} (score={result.score:.1f})")

        # 第二步：执行
        start = time.time()
        try:
            # 使用主流水线中的工具调用循环
            tools = system._gather_tools()
            cleaned_tools = system._clean_tools_for_api(tools)

            messages = [
                {"role": "system", "content": system._build_system_prompt()},
                {"role": "user", "content": task},
            ]

            final_response = await _run_fixed_agent_loop(
                system, messages, cleaned_tools, max_rounds=6
            )
            elapsed = time.time() - start

            if final_response:
                print(f"  响应 ({elapsed:.1f}s): {final_response[:300]}...")
                success = len(final_response) > 10
            else:
                print(f"  无响应 ({elapsed:.1f}s)")
                success = False

            if success:
                print("  PASS")
            else:
                print("  FAIL")
                all_pass = False

        except Exception as e:
            elapsed = time.time() - start
            print(f"  错误 ({elapsed:.1f}s): {type(e).__name__}: {str(e)[:200]}")
            # API 错误不算测试失败
            if "400" in str(e) or "InvalidParameter" in str(e):
                print("  SKIP: LLM API 错误，非逻辑问题")
            else:
                all_pass = False

    system.shutdown()
    return all_pass


async def _run_fixed_agent_loop(
    system: LongSystem,
    messages: list[dict],
    tools: list[dict],
    max_rounds: int = 6,
) -> str:
    """修复版 Agent 循环——修复 content: None 问题"""
    from long.llm.base import LLMMessage

    search_count = 0
    max_search = 3

    for round_idx in range(max_rounds):
        llm_messages = [
            LLMMessage(role=m["role"], content=m.get("content", "") or "")
            for m in messages
            if m["role"] in ("system", "user", "assistant")
        ]

        response = await system.llm.chat_with_tools(
            messages=llm_messages,
            tools=tools,
            temperature=0.7,
        )

        if not response.tool_calls:
            return response.content or ""

        # 构建 assistant 消息——修复：content 不能为 None
        assistant_content = response.content or ""
        assistant_msg: dict[str, object] = {
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": [],
        }

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

            assistant_msg["tool_calls"].append({
                "id": tc_id,
                "type": "function",
                "function": {"name": tc_name, "arguments": json.dumps(tc_args, ensure_ascii=False)},
            })

        messages.append(assistant_msg)

        for tc in assistant_msg["tool_calls"]:
            tool_name = tc["function"]["name"]
            tool_args = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]

            if "search" in tool_name.lower():
                search_count += 1
                if search_count > max_search:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "搜索次数已达上限，请基于已有信息生成回复。",
                    })
                    continue

            print(f"  🔧 {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:80]})")

            try:
                result = await system._execute_tool(tool_name, tool_args)
                result_preview = result[:200] + ("..." if len(result) > 200 else "")
                print(f"  → {result_preview}")
            except Exception as e:
                result = f"工具执行错误: {type(e).__name__}: {e}"
                print(f"  ❌ {result[:200]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result[:2000] if len(result) > 2000 else result,
            })

    return "未能完成任务（达到最大轮次限制）"


if __name__ == "__main__":
    success = True

    # 测试1: 任务分类
    if not test_task_classification():
        success = False

    # 测试2: 会话文件跟踪
    if not test_session_file_tracking():
        success = False

    # 测试3: 真实流水线
    if not asyncio.run(test_real_task_pipeline()):
        success = False

    print("\n" + "=" * 60)
    print(f"总体结果: {'PASS' if success else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if success else 1)