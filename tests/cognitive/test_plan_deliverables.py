"""回归测试：多步计划成功后必须走「交付物验证 → 自动修复 → 输出生成文件」链路

背景 bug：`_try_plan_execution` 中曾存在一处无条件 `return True`，
使其后约 250 行（交付物验证 `_verify_plan_deliverables`、自动修复
`_repair_plan_deliverables`、生成文件输出 `_print_generated_files`）
成为死代码。结果：生成 PPTX/PDF/报告的任务只会显示"任务已完成。"，
用户拿不到任何文件下载链接。

本测试锁定修复后的行为：
1. 计划成功且生成了文件 → 输出文件内容/链接，并存入会话
2. 交付物验证失败且修复失败 → 返回 False（降级），不谎报成功
3. 空计划 → 安全降级，不抛 IndexError
"""

import os
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

    def all_text(self) -> str:
        return "\n".join(str(p) for p in self.printed)


class _RecordingCLIAdapter:
    def __init__(self) -> None:
        self.console = _RecordingConsole()


class _FakeSession:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append((role, content))


class _FakeWorkspace:
    def __init__(self, root: str) -> None:
        self.root = root


class _FakeStep:
    def __init__(self, action: str, args: dict[str, Any], description: str = "") -> None:
        self.action = action
        self.args = args
        self.description = description
        self.depends_on: list[str] = []
        self.step_id = "step_x"


class _FakePlan:
    def __init__(self, steps: list[_FakeStep], goal: str = "生成报告") -> None:
        self.goal = goal
        self.steps = steps


class _FakePlanExecutor:
    def __init__(self, plan: _FakePlan, result: PlanExecutionResult) -> None:
        self._plan = plan
        self._result = result
        self.constraint_validator = self

    async def generate_plan(self, **kwargs: Any) -> _FakePlan:
        return self._plan

    def validate_plan(self, plan: Any) -> Any:
        class _V:
            valid = True
            errors: list[str] = []

        return _V()

    async def execute_plan(self, **kwargs: Any) -> PlanExecutionResult:
        return self._result


def _make_system(
    plan: _FakePlan, result: PlanExecutionResult, workspace_root: str | None = None
) -> tuple[LongSystem, _RecordingCLIAdapter]:
    system = LongSystem.__new__(LongSystem)
    system.plan_executor = _FakePlanExecutor(plan, result)
    system.active_session = _FakeSession()
    system.memory = None
    system.workspace = _FakeWorkspace(workspace_root) if workspace_root else None
    system._llm_call_total = 0
    system._session_start_ts = 0
    system._save_session = lambda: None
    system._schedule_auto_eval = lambda: None
    return system, _RecordingCLIAdapter()


@pytest.mark.asyncio
async def test_generated_pptx_link_is_emitted(tmp_path):
    """计划生成了 PPTX 时，必须输出下载链接而不是只说"任务已完成。" """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pptx = output_dir / "report.pptx"
    pptx.write_bytes(b"fake pptx payload")

    steps = [
        _FakeStep("call_tool", {"tool_name": "write_file", "parameters": {
            "path": "output/gen.py", "content": "from pptx import Presentation\n"}}),
        _FakeStep("call_tool", {"tool_name": "write_file", "parameters": {
            "path": "output/report.pptx"}}),
    ]
    plan = _FakePlan(steps, goal="生成 PPT 报告")
    # execute_plan 未产出 output_text —— 正是旧代码只显示"任务已完成。"的场景
    result = PlanExecutionResult(plan_id="p1", success=True, output_text="")

    (tmp_path / "output" / "gen.py").write_text(
        "from pptx import Presentation\nprs = Presentation()\n", encoding="utf-8"
    )

    system, cli = _make_system(plan, result, workspace_root=str(tmp_path))
    ok = await system._try_plan_execution(cli, [{"role": "user", "content": "做个PPT"}], tools=[])

    assert ok is True
    printed = cli.console.all_text()
    # 核心断言：PPTX 下载链接出现在输出中
    assert "report.pptx" in printed
    assert "/output/report.pptx" in printed
    # 且不能退化为无内容的"任务已完成。"
    assert system.active_session.messages, "最终结果必须写入会话"
    assert system.active_session.messages[-1][1] != "任务已完成。"


@pytest.mark.asyncio
async def test_missing_deliverable_degrades_instead_of_claiming_success(tmp_path):
    """计划声称写了文件但文件不存在 → 验证失败、修复失败 → 降级返回 False"""
    (tmp_path / "output").mkdir()

    steps = [
        _FakeStep("call_tool", {"tool_name": "write_file", "parameters": {
            "path": "output/missing_a.md", "content": "# 报告"}}, description="生成报告"),
        _FakeStep("call_tool", {"tool_name": "write_file", "parameters": {
            "path": "output/missing_b.md", "content": "# 报告2"}}, description="生成报告"),
    ]
    plan = _FakePlan(steps, goal="生成报告")
    result = PlanExecutionResult(plan_id="p2", success=True, output_text="")
    result.step_results = []

    system, cli = _make_system(plan, result, workspace_root=str(tmp_path))
    # 修复路径会调用 LLM；此处让其失败，验证不会谎报成功
    system.llm = None

    ok = await system._try_plan_execution(cli, [{"role": "user", "content": "写报告"}], tools=[])

    assert ok is False, "交付物缺失且修复失败时必须降级，而不是报告成功"
    assert "未找到" in cli.console.all_text() or "验证失败" in cli.console.all_text()


@pytest.mark.asyncio
async def test_empty_plan_degrades_without_indexerror():
    """空计划必须安全降级，不能因 plan.steps[0] 抛 IndexError"""
    plan = _FakePlan([], goal="空计划")
    result = PlanExecutionResult(plan_id="p3", success=True, output_text="")

    system, cli = _make_system(plan, result)
    ok = await system._try_plan_execution(cli, [{"role": "user", "content": "你好"}], tools=[])

    assert ok is False
