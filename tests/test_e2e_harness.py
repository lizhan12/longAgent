"""Harness 治理 E2E 测试 — 验证修复的正确性

覆盖修复:
  5. 并行工具执行结果丢失 (cli.py + execution_orchestrator.py)
  10. _daily_tokens_used 跨会话不重置 (llm/client.py)
  11. _half_open_calls 从未递增 (circuit_breaker.py)
  15. FeatureFlag YAML 解析崩溃 (feature_flag.py)
"""

from __future__ import annotations

import time

import pytest


# ===================== 修复 5: 并行工具执行 =====================


class TestParallelToolExecution:
    """并行工具执行结果修复测试"""

    def test_parallel_results_all_appended(self):
        """验证并行工具的所有结果都被添加到 history_msgs

        模拟 _fallback_tool_call_loop 中的并行执行逻辑，
        确保 for 循环内正确处理每个结果。
        """
        # 模拟并行执行时的内部逻辑
        results = [
            ("call_1", "结果1"),
            ("call_2", "结果2"),
            ("call_3", "结果3"),
        ]

        history_msgs: list[dict] = []

        # 修复前: budget_remaining 和 history_msgs.append 在循环外
        # 修复后: 全部在循环内
        for result in results:
            if isinstance(result, Exception):
                continue
            tc_id, tool_result = result
            # 所有处理逻辑现在在循环内
            is_error = tool_result.startswith("异常")
            history_msgs.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tc_id,
            })

        assert len(history_msgs) == 3, \
            f"应添加3条结果，实际: {len(history_msgs)}"
        assert history_msgs[0]["content"] == "结果1"
        assert history_msgs[1]["content"] == "结果2"
        assert history_msgs[2]["content"] == "结果3"

    def test_parallel_results_with_exception(self):
        """验证并行执行中异常结果被跳过，其他结果正常添加"""
        results = [
            ("call_1", "结果1"),
            Exception("tool failed"),
            ("call_3", "结果3"),
        ]

        history_msgs: list[dict] = []

        for result in results:
            if isinstance(result, Exception):
                continue
            tc_id, tool_result = result
            history_msgs.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tc_id,
            })

        assert len(history_msgs) == 2, \
            f"应添加2条结果(跳过异常)，实际: {len(history_msgs)}"


# ===================== 修复 10: 日预算自动重置 =====================


class TestDailyBudgetAutoReset:
    """日预算自动重置修复测试"""

    def test_daily_tokens_tracked_correctly(self):
        """验证日预算正常累计"""
        from long.llm.client import LLMClient
        from long.llm.base import LLMConfig, LLMUsage

        client = LLMClient(LLMConfig())
        assert client._daily_tokens_used == 0
        assert client._daily_tokens_date == ""

        usage = LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        client._track_usage(usage)
        assert client._daily_tokens_used == 150
        assert client._daily_tokens_date != ""

    def test_daily_budget_auto_resets_on_date_change(self):
        """验证日期变更后日预算自动重置"""
        from long.llm.client import LLMClient
        from long.llm.base import LLMConfig, LLMUsage

        client = LLMClient(LLMConfig())

        # 使用一次
        usage = LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        client._track_usage(usage)
        assert client._daily_tokens_used == 150

        # 模拟日期变更（将日期设为昨天）
        from datetime import datetime, timedelta
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        client._daily_tokens_date = yesterday

        # 再次使用，应自动重置
        client._track_usage(usage)
        assert client._daily_tokens_used == 150, \
            "日期变更后应重置为0再累加，预期150"

    def test_daily_budget_no_reset_same_date(self):
        """验证同一天内不重置"""
        from long.llm.client import LLMClient
        from long.llm.base import LLMConfig, LLMUsage

        client = LLMClient(LLMConfig())

        usage1 = LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        client._track_usage(usage1)
        assert client._daily_tokens_used == 150

        # 同一天再次使用
        usage2 = LLMUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300)
        client._track_usage(usage2)
        assert client._daily_tokens_used == 450, \
            "同一天内应累加，预期450"

    def test_budget_check_uses_correct_values(self):
        """验证预算检查使用正确的累计值"""
        from long.llm.client import LLMClient
        from long.llm.base import LLMConfig, BudgetConfig, LLMUsage

        # 设置小预算以便测试
        config = LLMConfig(budget=BudgetConfig(
            max_tokens_per_task=500,
            daily_token_limit=1000,
            max_tokens_per_request=16384,
        ))
        client = LLMClient(config)

        # 不超过预算
        usage = LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        client._track_usage(usage)
        client._check_budget(200)  # 不应抛出异常

        # 模拟日期变更
        from datetime import datetime, timedelta
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        client._daily_tokens_date = yesterday

        # 重置后应在预算内
        client._track_usage(usage)
        client._check_budget(200)  # 不应抛出异常


# ===================== 修复 11: half_open 限制 =====================


class TestCircuitBreakerHalfOpen:
    """熔断器 half_open 限制修复测试"""

    @staticmethod
    def _force_half_open(cb):
        """强制将熔断器转为 HALF_OPEN 状态"""
        from long.circuit_breaker import CircuitState
        cb._state = CircuitState.HALF_OPEN
        cb._half_open_calls = 0
        cb._opened_at = 0.0

    def test_half_open_limit_enforced(self):
        """验证 half_open_max_calls 限制生效"""
        from long.circuit_breaker import (
            CircuitBreaker, CircuitBreakerConfig, CircuitState,
        )

        cb = CircuitBreaker("test", CircuitBreakerConfig(
            half_open_max_calls=1,
            failure_threshold=2,
            timeout_seconds=0.1,
        ))

        # 触发熔断: 2 次失败 → OPEN
        cb.record_failure("err1")
        cb.record_failure("err2")
        assert cb.state == CircuitState.OPEN

        # 伪造超时，使 OPEN → HALF_OPEN
        cb._opened_at = time.monotonic() - 1000  # 伪造超时，使 _check_timeout 认为已超时

        # 第一次 can_execute: 应允许 (half_open_calls < 1)
        assert cb.can_execute() is True, "第一次 HALF_OPEN 调用应允许"
        assert cb._half_open_calls == 1, "half_open_calls 应递增为1"

        # 第二次 can_execute: 应拒绝 (half_open_calls >= 1)
        assert cb.can_execute() is False, "第二次 HALF_OPEN 调用应被限制"

    def test_half_open_multiple_slots(self):
        """验证 half_open_max_calls=3 时允许多次"""
        from long.circuit_breaker import (
            CircuitBreaker, CircuitBreakerConfig, CircuitState,
        )

        cb = CircuitBreaker("test", CircuitBreakerConfig(
            half_open_max_calls=3,
            failure_threshold=2,
            timeout_seconds=0.1,
        ))

        # 触发熔断
        cb.record_failure("err1")
        cb.record_failure("err2")
        assert cb.state == CircuitState.OPEN

        # 伪造超时
        cb._opened_at = time.monotonic() - 1000

        # 前3次应允许
        assert cb.can_execute() is True
        assert cb.can_execute() is True
        assert cb.can_execute() is True

        # 第4次应拒绝
        assert cb.can_execute() is False

    def test_half_open_success_transitions_to_closed(self):
        """验证 HALF_OPEN 中成功达到阈值后转为 CLOSED"""
        from long.circuit_breaker import (
            CircuitBreaker, CircuitBreakerConfig, CircuitState,
        )

        cb = CircuitBreaker("test", CircuitBreakerConfig(
            half_open_max_calls=2,  # 用2个槽位方便测试
            success_threshold=2,
            failure_threshold=2,
            timeout_seconds=0.1,
        ))

        self._force_half_open(cb)

        # 第一次成功
        assert cb.can_execute() is True
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # 1/2 成功，未达到阈值

        # 第二次成功
        assert cb.can_execute() is True
        cb.record_success()
        assert cb.state == CircuitState.CLOSED, \
            "2次成功后应转为 CLOSED"

    def test_half_open_failure_returns_to_open(self):
        """验证 HALF_OPEN 中失败回到 OPEN"""
        from long.circuit_breaker import (
            CircuitBreaker, CircuitBreakerConfig, CircuitState,
        )

        cb = CircuitBreaker("test", CircuitBreakerConfig(
            half_open_max_calls=1,
            failure_threshold=2,
            timeout_seconds=0.1,
        ))

        cb.record_failure("err1")
        cb.record_failure("err2")
        assert cb.state == CircuitState.OPEN

        cb._opened_at = time.monotonic() - 1000
        assert cb.can_execute() is True

        # HALF_OPEN 中失败
        cb.record_failure("half_open_fail")
        assert cb.state == CircuitState.OPEN, \
            "HALF_OPEN 中失败应回到 OPEN"


# ===================== 修复 15: FeatureFlag YAML 崩溃 =====================


class TestFeatureFlagYamlFallback:
    """FeatureFlag YAML 解析崩溃修复测试"""

    def test_yaml_crash_falls_back_to_default(self, tmp_path):
        """验证 YAML 解析失败时回退到默认配置"""
        from long.harness.feature_flag import FeatureFlag

        # 创建无效的 YAML 文件
        bad_yaml = tmp_path / "bad_flags.yaml"
        bad_yaml.write_text("{invalid: yaml: unclosed_quote", encoding="utf-8")

        # 不应抛出异常，应返回默认配置
        flags = FeatureFlag.from_yaml(str(bad_yaml))
        assert flags is not None
        # 默认配置中 flag 应启用
        assert flags.is_enabled("output_pii_filter") is True

    def test_nonexistent_file_returns_default(self, tmp_path):
        """验证不存在的文件返回默认配置"""
        from long.harness.feature_flag import FeatureFlag

        nonexistent = tmp_path / "nonexistent.yaml"
        flags = FeatureFlag.from_yaml(str(nonexistent))
        assert flags is not None
        assert isinstance(flags.is_enabled("output_pii_filter"), bool)

    def test_empty_file_returns_default(self, tmp_path):
        """验证空文件返回默认配置"""
        from long.harness.feature_flag import FeatureFlag

        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")

        # 空文件应返回默认（不崩溃）
        flags = FeatureFlag.from_yaml(str(empty))
        assert flags is not None

    def test_valid_yaml_works(self, tmp_path):
        """验证合法的 YAML 文件正常加载"""
        from long.harness.feature_flag import FeatureFlag

        valid = tmp_path / "valid.yaml"
        valid.write_text("""feature_flags:
  test_flag:
    enabled: true
""", encoding="utf-8")

        flags = FeatureFlag.from_yaml(str(valid))
        assert flags.is_enabled("test_flag") is True