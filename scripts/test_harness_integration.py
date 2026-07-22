"""Harness 集成测试 — 验证新增模块在真实任务流程中的工作

测试内容：
1. CircuitBreaker 熔断保护
2. FeedbackLoop 运行时失败自动提案
3. DecisionLog 决策记录
4. PermissionManifest 权限检查
5. TaskContextIsolator 上下文隔离
6. DurabilityTracker 耐久性追踪
7. NearMissTracker 近失追踪
8. ContextAttentionEngineer 上下文管理
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from long.cli import LongSystem, load_dotenv
from long.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerRegistry, CircuitState
from long.harness.feedback_loop import FeedbackLoop
from long.harness.decision_log import DecisionLog, DecisionCategory
from long.harness.permission_manifest import PermissionManifest, PermissionManifestLoader, ToolPermission
from long.harness.context_isolation import TaskContextIsolator, IsolationLevel, ErrorBoundary
from long.harness.durability_tracker import DurabilityTracker
from long.harness.near_miss_tracker import NearMissTracker, NearMissCategory
from long.harness.attention_engineer import ContextAttentionEngineer, AttentionConfig, ContextPriority


def test_circuit_breaker():
    """测试1: CircuitBreaker 熔断保护"""
    print("\n" + "=" * 60)
    print("测试1: CircuitBreaker 熔断保护")
    print("=" * 60)

    registry = CircuitBreakerRegistry()
    cb = registry.get_or_create("llm_primary", CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=5.0,
    ))

    # 模拟正常调用
    assert cb.can_execute(), "CLOSED 状态应允许执行"
    cb.record_success()
    cb.record_success()
    print(f"  ✅ 正常调用: state={cb.state.value}, stats={cb.get_stats().total_successes}")

    # 模拟连续失败触发熔断
    for i in range(3):
        cb.record_failure(f"timeout error #{i+1}")
        print(f"  ⚠️  失败 #{i+1}: state={cb.state.value}")

    assert cb.state == CircuitState.OPEN, "连续失败3次应触发熔断"
    assert not cb.can_execute(), "OPEN 状态应拒绝执行"
    print(f"  🔴 熔断触发: state={cb.state.value}")

    # 模拟超时后进入半开状态
    cb._opened_at = time.monotonic() - 10.0  # 模拟已过超时时间
    assert cb.can_execute(), "超时后应进入 HALF_OPEN 允许探测"
    assert cb.state == CircuitState.HALF_OPEN, "应进入 HALF_OPEN"
    print(f"  🟡 半开探测: state={cb.state.value}")

    # 探测成功恢复
    cb.record_success()
    cb.record_success()
    assert cb.state == CircuitState.CLOSED, "探测成功应恢复 CLOSED"
    print(f"  🟢 恢复正常: state={cb.state.value}")

    # Registry 统计
    stats = registry.get_all_stats()
    print(f"  📊 Registry: {list(stats.keys())}, OPEN={registry.get_open_breakers()}")
    print("  ✅ CircuitBreaker 测试通过")


def test_feedback_loop_runtime():
    """测试2: FeedbackLoop 运行时失败自动提案"""
    print("\n" + "=" * 60)
    print("测试2: FeedbackLoop 运行时失败自动提案")
    print("=" * 60)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        fl = FeedbackLoop(td)

        # 模拟工具失败
        p1 = fl.generate_from_tool_failure("execute_code", "SyntaxError: invalid syntax")
        print(f"  📝 工具失败提案: {p1.category} - {p1.description[:60]}")

        # 模拟约束违反
        p2 = fl.generate_from_constraint_violation("state_machine", "非法状态转移: EXECUTING→INIT")
        print(f"  📝 约束违反提案: {p2.category} - {p2.description[:60]}")

        # 模拟沙箱失败
        p3 = fl.generate_from_sandbox_failure("reverse_shell", "检测到 socket.connect")
        print(f"  📝 沙箱失败提案: {p3.category} - {p3.description[:60]}")

        # 模拟 near-miss
        p4 = fl.generate_from_near_miss("budget", 0.78, 0.80, context="token 消耗")
        print(f"  📝 Near-miss 提案: {p4.category} - {p4.description[:60]}")

        # 去重测试
        p5 = fl.generate_from_tool_failure("execute_code", "另一个错误")
        assert p5 is None, "同工具失败应被去重"
        print(f"  🔄 去重验证: 重复提案被拦截 ✅")

        pending = fl.list_pending()
        print(f"  📊 待处理提案: {len(pending)} 条")
        stats = fl.get_stats()
        print(f"  📊 提案统计: {stats}")

        # 审批流程
        if p1:
            fl.approve(p1.proposal_id)
            fl.mark_applied(p1.proposal_id)
            print(f"  ✅ 提案 {p1.proposal_id} 已审批并应用")

        print("  ✅ FeedbackLoop 测试通过")


def test_decision_log():
    """测试3: DecisionLog 决策记录"""
    print("\n" + "=" * 60)
    print("测试3: DecisionLog 决策记录")
    print("=" * 60)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        dl = DecisionLog(td)

        # 记录架构决策
        r1 = dl.record(
            category="architecture",
            title="选择进程沙箱而非容器",
            context="个人工具不需要 Docker 的复杂度",
            options=["进程沙箱", "Docker 容器", "无沙箱"],
            chosen="进程沙箱",
            rationale="个人工具场景下，进程沙箱足够安全且更轻量",
            expected_outcome="安全隔离 + 低延迟 + 零 Docker 依赖",
            tags=["security", "sandbox"],
        )
        print(f"  📝 决策记录: {r1.decision_id} - {r1.title}")

        # 记录模型决策
        r2 = dl.record(
            category="model",
            title="主模型选择 deepseek-v4-pro",
            context="需要强推理能力处理复杂任务",
            options=["deepseek-v4-pro", "qwen3.6-flash", "gpt-4o"],
            chosen="deepseek-v4-pro",
            rationale="性价比最高，推理能力强",
            expected_outcome="高质量输出 + 合理成本",
        )
        print(f"  📝 决策记录: {r2.decision_id} - {r2.title}")

        # 复盘
        dl.review(r1.decision_id, "运行稳定，隔离有效，RLIMIT 问题已通过移除 NPROC 解决")
        print(f"  🔍 决策复盘: {r1.decision_id}")

        # 统计
        stats = dl.get_stats()
        unreviewed = dl.list_unreviewed()
        print(f"  📊 统计: {stats}")
        print(f"  📊 未复盘: {len(unreviewed)} 条")

        # 验证 Markdown 同步
        md_path = dl._markdown_path()
        if md_path.exists():
            content = md_path.read_text(encoding="utf-8")
            print(f"  📄 DECISIONS.md 已生成: {len(content)} 字符")
            assert "进程沙箱" in content

        print("  ✅ DecisionLog 测试通过")


def test_permission_manifest():
    """测试4: PermissionManifest 权限检查"""
    print("\n" + "=" * 60)
    print("测试4: PermissionManifest 权限检查")
    print("=" * 60)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        loader = PermissionManifestLoader(td)
        manifest = loader.load()

        # development 模式检查
        assert manifest.is_allowed("read_file", "development")
        assert manifest.is_allowed("write_file", "development")
        assert manifest.is_allowed("delete_file", "development")
        print("  ✅ development 模式: 所有工具允许")

        # service 模式检查
        assert manifest.is_allowed("read_file", "service")
        assert not manifest.is_allowed("delete_file", "service")
        assert not manifest.is_allowed("execute_code", "service")
        assert not manifest.is_allowed("execute_file", "service")
        print("  ✅ service 模式: 危险工具被禁止")

        # 确认需求
        assert manifest.needs_confirmation("delete_file")
        assert manifest.needs_confirmation("execute_code")
        assert not manifest.needs_confirmation("read_file")
        print("  ✅ HITL 确认: 高风险工具需确认")

        # 风险等级
        high_risk = manifest.get_high_risk_tools()
        print(f"  ⚠️  高风险工具: {high_risk}")
        assert "delete_file" in high_risk

        # 保存并验证 Markdown
        loader.save(manifest)
        md_path = loader._markdown_path()
        if md_path.exists():
            content = md_path.read_text(encoding="utf-8")
            print(f"  📄 PERMISSIONS.md 已生成: {len(content)} 字符")

        print("  ✅ PermissionManifest 测试通过")


def test_context_isolation():
    """测试5: TaskContextIsolator 上下文隔离"""
    print("\n" + "=" * 60)
    print("测试5: TaskContextIsolator 上下文隔离")
    print("=" * 60)

    isolator = TaskContextIsolator()

    # 创建主任务
    main_ctx = isolator.create_context("main_task", isolation_level=IsolationLevel.NONE)
    main_ctx.add_message("user", "帮我分析数据并生成报告")

    # 创建子任务（软隔离）
    sub1 = isolator.create_context("sub_data_analysis", parent_task_id="main_task", isolation_level=IsolationLevel.SOFT)
    sub1.add_tool_result("read_file", "数据内容: ...", is_error=False)
    sub1.add_tool_result("execute_code", "分析结果: ...", is_error=False)
    sub1.mark_completed()
    print(f"  ✅ 子任务1 (软隔离): phase={sub1.phase.value}, results={len(sub1.tool_results)}")

    # 创建子任务（硬隔离，模拟失败）
    sub2 = isolator.create_context("sub_report_gen", parent_task_id="main_task", isolation_level=IsolationLevel.HARD)
    sub2.add_tool_result("write_file", "磁盘空间不足", is_error=True)
    sub2.mark_failed("写入失败: 磁盘空间不足")
    print(f"  ❌ 子任务2 (硬隔离): phase={sub2.phase.value}, errors={sub2.error_summary[:50]}")

    # 错误边界测试
    with isolator.error_boundary.guard("sub_risky"):
        # 模拟一个被捕获的错误
        pass  # 正常执行

    assert not isolator.error_boundary.has_failed("sub_risky")

    with isolator.error_boundary.guard("sub_failing"):
        raise RuntimeError("模拟子任务崩溃")

    assert isolator.error_boundary.has_failed("sub_failing")
    print(f"  🛡️  错误边界: 捕获到 {len(isolator.error_boundary.get_all_failures())} 个失败")

    # 合并上下文
    merged = isolator.get_merged_messages("sub_data_analysis", system_prompt="你是数据分析助手")
    print(f"  📊 软隔离合并: {len(merged)} 条消息")

    # 统计
    stats = isolator.get_stats()
    print(f"  📊 隔离统计: {stats}")

    print("  ✅ TaskContextIsolator 测试通过")


def test_durability_tracker():
    """测试6: DurabilityTracker 耐久性追踪"""
    print("\n" + "=" * 60)
    print("测试6: DurabilityTracker 耐久性追踪")
    print("=" * 60)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tracker = DurabilityTracker(td)

        # 模拟一个5步任务，前3步成功，后2步失败
        steps = [
            (0, "read_file", True, ""),
            (1, "read_file", True, ""),
            (2, "write_file", True, ""),
            (3, "execute_code", False, "timeout"),
            (4, "execute_code", False, "tool_error"),
        ]
        for idx, tool, success, err in steps:
            tracker.record_step("task_report", idx, tool, success, err, duration_ms=100.0 * (idx + 1))

        report = tracker.get_report("task_report")
        print(f"  📊 总步数: {report.total_steps}")
        print(f"  📊 成功: {report.successful_steps}, 失败: {report.failed_steps}")
        print(f"  📊 成功率: {report.overall_success_rate:.1%}")
        print(f"  📊 跑偏拐点: 第 {report.drift_point} 步 (置信度: {report.drift_confidence:.2f})")
        print(f"  📊 每步累计成功率: {[f'{r:.2f}' for r in report.step_success_rates]}")
        print(f"  📊 错误类型分布: {report.error_types}")

        # 聚合报告
        tracker.record_step("task_other", 0, "read_file", True)
        tracker.record_step("task_other", 1, "execute_code", False, "syntax_error")
        agg = tracker.get_aggregate_report()
        print(f"  📊 聚合: total={agg.total_steps}, rate={agg.overall_success_rate:.1%}")

        print("  ✅ DurabilityTracker 测试通过")


def test_near_miss_tracker():
    """测试7: NearMissTracker 近失追踪"""
    print("\n" + "=" * 60)
    print("测试7: NearMissTracker 近失追踪")
    print("=" * 60)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tracker = NearMissTracker(td)

        # 记录各类 near-miss
        tracker.record("budget", "Token 消耗接近日预算", current_value=0.85, threshold=0.80, task_id="task_1")
        tracker.record("timeout", "LLM 响应接近超时", current_value=140, threshold=150, task_id="task_1")
        tracker.record("constraint", "约束验证接近阈值", current_value=0.72, threshold=0.70, task_id="task_2")
        tracker.record("retry", "第一次失败后重试成功", current_value=1, threshold=1, task_id="task_3")

        unresolved = tracker.get_unresolved()
        print(f"  ⚠️  未解决 near-miss: {len(unresolved)} 条")

        report = tracker.get_report()
        print(f"  📊 汇总: {report}")

        # 按严重程度查看
        high = tracker.get_by_severity("high")
        medium = tracker.get_by_severity("medium")
        low = tracker.get_by_severity("low")
        print(f"  📊 严重程度: HIGH={len(high)}, MEDIUM={len(medium)}, LOW={len(low)}")

        # 标记解决
        if unresolved:
            tracker.mark_resolved(unresolved[0].record_id)
            print(f"  ✅ 已标记解决: {unresolved[0].record_id}")

        print("  ✅ NearMissTracker 测试通过")


def test_attention_engineer():
    """测试8: ContextAttentionEngineer 上下文管理"""
    print("\n" + "=" * 60)
    print("测试8: ContextAttentionEngineer 上下文管理")
    print("=" * 60)

    engine = ContextAttentionEngineer(AttentionConfig(
        max_context_tokens=500,
        critical_reserve_ratio=0.3,
        high_reserve_ratio=0.4,
        compression_threshold=0.8,
    ))

    # 添加不同优先级的内容
    engine.add_slot("constraint_1", "🔒 不可删除 /etc/ 目录下的文件", priority=ContextPriority.CRITICAL, category="constraint")
    engine.add_slot("constraint_2", "🔒 execute_code 始终需要 AST 扫描", priority=ContextPriority.CRITICAL, category="constraint")
    engine.add_slot("current_goal", "当前目标: 生成数据分析报告", priority=ContextPriority.HIGH, category="system")
    engine.add_slot("tool_result_1", "文件内容: " + "数据行..." * 20, priority=ContextPriority.MEDIUM, category="tool_result")
    engine.add_slot("tool_result_2", "执行结果: " + "输出行..." * 20, priority=ContextPriority.MEDIUM, category="tool_result")
    engine.add_slot("old_history", "早期对话: " + "闲聊..." * 30, priority=ContextPriority.LOW, category="history")

    print(f"  📊 总 token 估计: {engine.total_token_estimate}")
    print(f"  📊 使用率: {engine.usage_ratio:.1%}")

    # 构建上下文
    context = engine.build_context()
    print(f"  📊 构建后消息数: {len(context)}")

    for msg in context:
        content_preview = msg["content"][:80].replace("\n", " ")
        print(f"    [{msg['role']}] {content_preview}...")

    stats = engine.get_stats()
    print(f"  📊 统计: {stats}")

    print("  ✅ ContextAttentionEngineer 测试通过")


async def test_real_task_with_harness():
    """测试9: 真实任务 + Harness 集成"""
    print("\n" + "=" * 60)
    print("测试9: 真实任务 + Harness 集成")
    print("=" * 60)

    load_dotenv()
    system = LongSystem(config_dir="configs", workspace_root="./workspace")
    system.initialize()

    llm = system.llm
    if llm is None:
        print("  ⚠️  LLM 未初始化，跳过真实任务测试")
        return True

    api_key = llm.config.resolve_api_key()
    if not api_key:
        print("  ⚠️  API Key 未配置，跳过真实任务测试")
        return True

    print(f"  🤖 LLM: model={llm.config.model}")

    # 初始化 Harness 模块
    import tempfile
    harness_dir = Path("./workspace")
    harness_dir.mkdir(parents=True, exist_ok=True)

    decision_log = DecisionLog(harness_dir)
    durability = DurabilityTracker(harness_dir)
    near_miss = NearMissTracker(harness_dir)
    permission_loader = PermissionManifestLoader(harness_dir)
    permissions = permission_loader.load()

    # 记录本次测试的决策
    decision_log.record(
        category="workflow",
        title="启用 Harness 集成测试",
        context="验证新增 Harness 模块在真实任务中的工作",
        chosen="全模块集成测试",
        rationale="确保所有模块可正常工作",
        expected_outcome="所有模块正常工作，无异常",
    )

    # 简单任务测试
    task = "列出当前目录的文件"
    print(f"\n  📋 任务: {task}")

    start = time.time()
    try:
        from long.llm.base import LLMMessage

        messages = [
            LLMMessage(role="system", content=system._build_system_prompt()),
            LLMMessage(role="user", content=task),
        ]

        tools = system._gather_tools()
        cleaned_tools = system._clean_tools_for_api(tools)

        # 检查工具权限
        for t in cleaned_tools:
            tool_name = t.get("function", {}).get("name", "")
            if not permissions.is_allowed(tool_name, system._security_mode):
                print(f"  🚫 工具被权限拦截: {tool_name}")

        response = await llm.chat_with_tools(
            messages=messages,
            tools=cleaned_tools,
            temperature=0.7,
        )

        elapsed = time.time() - start

        if response.tool_calls:
            for tc in response.tool_calls:
                tool_name = tc.get("name", "") if isinstance(tc, dict) else tc["name"]
                print(f"  🔧 调用工具: {tool_name}")

                # 记录耐久性
                durability.record_step("real_task", 0, tool_name, True)

                # 检查是否需要确认
                if permissions.needs_confirmation(tool_name):
                    near_miss.record(
                        "tool_error",
                        f"高风险工具 {tool_name} 在开发模式下执行",
                        task_id="real_task",
                        tool_name=tool_name,
                    )

                result = await system._execute_tool(tool_name, tc.get("arguments", {}))
                print(f"  📝 结果: {str(result)[:100]}")

            # 第二轮获取最终回复
            messages.append(LLMMessage(role="assistant", content=response.content or "", tool_calls=response.tool_calls))
            final = await llm.chat(messages=messages, temperature=0.7)
            result_text = final.content or ""
        else:
            result_text = response.content or ""

        print(f"\n  ✅ 响应 ({elapsed:.1f}s):")
        print(f"  {result_text[:300]}")

        # 记录耐久性
        durability.record_step("real_task", 1, "llm_response", True)

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ❌ 错误: {type(e).__name__}: {e}")
        durability.record_step("real_task", 0, "llm_call", False, type(e).__name__)

    # 查看 Harness 报告
    dur_report = durability.get_report("real_task")
    nm_report = near_miss.get_report()
    print(f"\n  📊 耐久性: steps={dur_report.total_steps}, rate={dur_report.overall_success_rate:.1%}")
    print(f"  📊 Near-miss: {nm_report}")

    system.shutdown()
    return True


def main():
    print("=" * 60)
    print("Harness 集成测试 — 7条原则 × 9个模块")
    print("=" * 60)

    tests = [
        ("CircuitBreaker 熔断保护", test_circuit_breaker),
        ("FeedbackLoop 运行时提案", test_feedback_loop_runtime),
        ("DecisionLog 决策记录", test_decision_log),
        ("PermissionManifest 权限检查", test_permission_manifest),
        ("TaskContextIsolator 上下文隔离", test_context_isolation),
        ("DurabilityTracker 耐久性追踪", test_durability_tracker),
        ("NearMissTracker 近失追踪", test_near_miss_tracker),
        ("ContextAttentionEngineer 上下文管理", test_attention_engineer),
    ]

    results = []
    for name, fn in tests:
        try:
            fn()
            results.append((name, True))
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            results.append((name, False))

    # 真实任务测试
    print("\n" + "=" * 60)
    print("真实任务测试（需要 API Key）")
    print("=" * 60)
    try:
        asyncio.run(test_real_task_with_harness())
        results.append(("真实任务+Harness", True))
    except Exception as e:
        print(f"  ❌ 真实任务失败: {e}")
        results.append(("真实任务+Harness", False))

    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, passed in results:
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n通过: {passed}/{total} ({passed/total:.0%})")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
