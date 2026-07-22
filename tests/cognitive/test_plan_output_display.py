"""回归测试：多步计划执行成功后，output_text 必须被打印到控制台

背景 bug：`_try_plan_execution` 成功分支曾调用 `cli_adapter.console.print()`
（无参数，只输出空行），把真实的 `exec_result.output_text`（如天气查询结果）
吞掉，导致用户看到"计划执行完成"但没有任何内容输出。

本测试通过 `__new__` 绕过 LongSystem 的重量级初始化，只桩接
`_try_plan_execution` 成功分支所触及的最小依赖，确认 output_text
真正被 print 出来。
"""

import asyncio
from contextlib import contextmanager
from typing import Any

import pytest

from long.cli import LongSystem
from long.ir.executor import PlanExecutionResult


class _RecordingConsole:
    def __init__(self) -> None:
        self.printed: list[str] = []

    def print(self, *args: Any, **kwargs: Any) -> None:
        self.printed.append(args[0] if args else "")

    @contextmanager
    def status(self, *args: Any, **kwargs: Any):
        yield self


class _RecordingCLIAdapter:
    def __init__(self) -> None:
        self.console = _RecordingConsole()


class _FakeSession:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append((role, content))


class _FakePlanStep:
    def __init__(self) -> None:
        self.action = "call_tool"
        self.args: dict[str, Any] = {"tool_name": "query_weather"}


class _FakePlan:
    def __init__(self) -> None:
        self.goal = "查询苏州今天的天气情况并输出结果"
        # 两步：确保走多步分支（len > 1）
        self.steps = [_FakePlanStep(), _FakePlanStep()]


class _FakePlanExecutor:
    def __init__(self, output_text: str) -> None:
        self._output_text = output_text
        self.constraint_validator = self

    async def generate_plan(self, **kwargs: Any) -> _FakePlan:
        return _FakePlan()

    def validate_plan(self, plan: Any) -> Any:
        class _V:
            valid = True
            errors: list[str] = []

        return _V()

    async def execute_plan(self, **kwargs: Any) -> PlanExecutionResult:
        return PlanExecutionResult(
            plan_id="plan_test",
            success=True,
            output_text=self._output_text,
        )


def _make_system(output_text: str) -> tuple[LongSystem, _RecordingCLIAdapter]:
    system = LongSystem.__new__(LongSystem)
    system.plan_executor = _FakePlanExecutor(output_text)
    system.active_session = _FakeSession()
    system.memory = None
    system._llm_call_total = 0
    system._save_session = lambda: None
    system._schedule_auto_eval = lambda: None
    return system, _RecordingCLIAdapter()


@pytest.mark.asyncio
async def test_multistep_plan_output_text_is_printed():
    """成功的多步计划：output_text 必须出现在控制台输出里"""
    weather = "【苏州（江苏）】\n  实况: 晴，气温28℃，湿度60%"
    system, cli_adapter = _make_system(weather)

    history = [{"role": "user", "content": "今天的苏州的天气怎么样"}]
    ok = await system._try_plan_execution(cli_adapter, history, tools=[])

    assert ok is True
    # 核心断言：真实结果被打印，而不是被空 print() 吞掉
    assert weather in cli_adapter.console.printed
    # 同时应写入会话
    assert ("assistant", weather) in system.active_session.messages


@pytest.mark.asyncio
async def test_multistep_plan_output_text_saved_to_session():
    """output_text 同时进入会话历史，供后续上下文使用"""
    text = "结果内容 ABC"
    system, cli_adapter = _make_system(text)

    history = [{"role": "user", "content": "随便问点啥"}]
    await system._try_plan_execution(cli_adapter, history, tools=[])

    assert ("assistant", text) in system.active_session.messages
