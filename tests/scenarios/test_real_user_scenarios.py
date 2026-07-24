"""基于真实用户请求场景的端到端测试

测试覆盖真实用户请求的全链路：
1. 解析器能否正确解析该场景的 PlanIR 结构
2. 约束验证器能否通过该场景的计划
3. 计划中的步骤是否使用了正确的工具
4. 计划是否生成了预期的输出文件

每个场景模拟一个真实的用户请求，如：
- "杭州和苏州的未来一周的天气的对比，要有图表，要word格式"
- "写一篇关于ai发展报告的历史报告，要有图的ppt"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from long.ir.ir_parser import IRParseStatus, IRParser
from long.ir.plan_ir import PlanIR, StepIR
from long.ir.type_checker import TypeChecker
from long.ir.state_machine import AgentStateMachine
from long.ir.ltl import LTLValidator
from long.ir.constraint_validator import ConstraintValidator

_SCENARIO_FILE = Path(__file__).resolve().parent / "real_user_requests.json"


def _load_scenarios() -> list[dict]:
    """加载场景定义文件"""
    if not _SCENARIO_FILE.exists():
        return []
    data = json.loads(_SCENARIO_FILE.read_text(encoding="utf-8"))
    return data.get("scenarios", [])


SCENARIOS = _load_scenarios()


def _scenario_id(scenario: dict) -> str:
    """生成场景的测试 ID"""
    return f"{scenario['name']}[{scenario['user_request'][:30]}...]"


# ======================== 测试 1: 解析器能正确解析各场景的 PlanIR ========================


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_scenario_id)
def test_parser_handles_scenario_plan(scenario):
    """验证解析器能正确解析该场景的 PlanIR 结构"""
    parser = IRParser()
    plan_json = json.dumps(scenario["plan_template"])
    result = parser.parse(plan_json)

    assert result.status in (
        IRParseStatus.SUCCESS, IRParseStatus.REPAIRABLE
    ), (
        f"[{scenario['name']}] 解析失败: {result.errors}\n"
        f"  用户请求: {scenario['user_request']}"
    )
    assert result.plan is not None
    assert result.plan.plan_id, f"[{scenario['name']}] plan_id 为空"
    assert result.plan.goal, f"[{scenario['name']}] goal 为空"


# ======================== 测试 2: 约束验证器能通过各场景的计划 ========================


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_scenario_id)
def test_constraint_validator_passes_scenario(scenario):
    """验证约束验证器能通过该场景的计划（状态机 + 类型检查 + LTL）"""
    plan_json = json.dumps(scenario["plan_template"])
    parser = IRParser()
    result = parser.parse(plan_json)
    assert result.plan is not None, f"[{scenario['name']}] 解析失败"

    validator = ConstraintValidator(
        state_machine=AgentStateMachine(),
        type_checker=TypeChecker(),
        ltl_validator=LTLValidator(),
    )
    validation = validator.validate_plan(result.plan)
    assert validation.valid, (
        f"[{scenario['name']}] 约束验证失败: {validation.errors}\n"
        f"  用户请求: {scenario['user_request']}"
    )


# ======================== 测试 3: 计划使用了正确的工具 ========================


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_scenario_id)
def test_scenario_uses_required_tools(scenario):
    """验证该场景的计划使用了所有必需的 tool"""
    plan_json = json.dumps(scenario["plan_template"])
    parser = IRParser()
    result = parser.parse(plan_json)
    assert result.plan is not None

    # 收集计划中使用的所有工具（从 tool_name 或 action 中提取）
    used_tools: set[str] = set()
    for step in result.plan.steps:
        tool_name = ""
        if step.args:
            tool_name = step.args.get("tool_name", "")
        if not tool_name:
            tool_name = step.action if isinstance(step.action, str) else step.action.value if hasattr(step.action, 'value') else str(step.action)
        if tool_name:
            used_tools.add(tool_name)

    for required_tool in scenario["required_tools"]:
        assert required_tool in used_tools, (
            f"[{scenario['name']}] 缺少必需工具 '{required_tool}'\n"
            f"  已用工具: {used_tools}\n"
            f"  用户请求: {scenario['user_request']}"
        )


# ======================== 测试 4: 步骤数量在合理范围内 ========================


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_scenario_id)
def test_scenario_step_count_in_range(scenario):
    """验证该场景的步骤数量在合理范围内"""
    plan_json = json.dumps(scenario["plan_template"])
    parser = IRParser()
    result = parser.parse(plan_json)
    assert result.plan is not None

    step_count = len(result.plan.steps)
    assert scenario["min_steps"] <= step_count <= scenario["max_steps"], (
        f"[{scenario['name']}] 步骤数量 {step_count} 不在范围 "
        f"[{scenario['min_steps']}, {scenario['max_steps']}] 内\n"
        f"  用户请求: {scenario['user_request']}"
    )


# ======================== 测试 5: 计划能生成预期输出文件 ========================


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_scenario_id)
def test_scenario_produces_expected_outputs(scenario):
    """验证该场景的步骤中包含了预期输出文件的保存操作"""
    plan_json = json.dumps(scenario["plan_template"])
    parser = IRParser()
    result = parser.parse(plan_json)
    assert result.plan is not None

    # 从步骤中提取所有文件路径引用
    file_paths: list[str] = []
    for step in result.plan.steps:
        args = step.args or {}
        # 检查 args 顶层字段
        if "path" in args:
            file_paths.append(args["path"])
        # 检查 parameters 子字典
        params = args.get("parameters", {})
        if isinstance(params, dict):
            if "path" in params:
                file_paths.append(params["path"])
            code = params.get("code", "")
            if code:
                import re
                file_paths.extend(re.findall(r"\.(?:save|savefig|output)\([\"']([^\"']+)[\"']", code))
            content = params.get("content", "")
            if content:
                import re
                file_paths.extend(re.findall(r"\.(?:save|output)\([\"']([^\"']+)[\"']", content))
        # 检查 args 顶层 code/content
        code = args.get("code", "")
        if code:
            import re
            file_paths.extend(re.findall(r"\.(?:save|savefig|output)\([\"']([^\"']+)[\"']", code))
        content = args.get("content", "")
        if content:
            import re
            file_paths.extend(re.findall(r"\.(?:save|output)\([\"']([^\"']+)[\"']", content))

    for expected_ext in scenario["expected_outputs"]:
        found = any(expected_ext in fp for fp in file_paths)
        assert found, (
            f"[{scenario['name']}] 未找到预期输出文件类型 '{expected_ext}'\n"
            f"  引用的文件路径: {file_paths}\n"
            f"  用户请求: {scenario['user_request']}"
        )
    # runtime_outputs 是运行时生成的，不强制在计划参数中出现
    for runtime_ext in scenario.get("runtime_outputs", []):
        found = any(runtime_ext in fp for fp in file_paths)
        if not found:
            logger = logging.getLogger(__name__)
            logger.info(
                "运行时输出文件 '%s' 未在计划参数中显式引用（由脚本执行时生成），"
                "需运行时验证", runtime_ext
            )


# ======================== 测试 6: 依赖关系正确 ========================


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_scenario_id)
def test_scenario_dependencies_valid(scenario):
    """验证该场景的步骤依赖关系是有效的"""
    plan_json = json.dumps(scenario["plan_template"])
    parser = IRParser()
    result = parser.parse(plan_json)
    assert result.plan is not None

    step_ids = {s.step_id for s in result.plan.steps if s.step_id}

    for step in result.plan.steps:
        for dep in (step.depends_on or []):
            assert dep in step_ids, (
                f"[{scenario['name']}] 步骤 {step.step_id} 依赖了不存在的步骤 {dep}\n"
                f"  有效步骤: {step_ids}"
            )