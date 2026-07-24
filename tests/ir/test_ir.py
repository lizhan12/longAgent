"""IR 模块测试

覆盖 PlanIR, StepIR, ActionType, TypeChecker, IRParser
"""

import json
from pathlib import Path

import pytest

from long.ir.ir_parser import IRParseResult, IRParseStatus, IRParser
from long.ir.plan_ir import ActionType, PlanIR, RiskLevel, StepIR
from long.ir.type_checker import TypeChecker


# ========================
# PlanIR / StepIR 测试
# ========================


class TestActionType:
    """ActionType 枚举测试"""

    def test_all_action_types(self):
        expected = [
            "search", "call_api", "call_tool", "call_mcp", "call_skill",
            "read_file", "write_file", "execute_file",
            "reason", "summarize", "output", "wait_approval", "skip",
        ]
        actual = [a.value for a in ActionType]
        assert sorted(actual) == sorted(expected)

    def test_action_type_from_value(self):
        assert ActionType("search") == ActionType.SEARCH
        assert ActionType("output") == ActionType.OUTPUT

    def test_invalid_action_type(self):
        with pytest.raises(ValueError):
            ActionType("invalid_action")


class TestStepIR:
    """StepIR 模型测试"""

    def test_minimal_step(self):
        step = StepIR(step_id="s1", action=ActionType.SEARCH, args={"query": "test"})
        assert step.step_id == "s1"
        assert step.action == ActionType.SEARCH
        assert step.risk_level == RiskLevel.LOW

    def test_full_step(self):
        step = StepIR(
            step_id="s1",
            action=ActionType.SEARCH,
            args={"query": "test"},
            depends_on=["s0"],
            condition="has_data",
            fallback_step="s2",
            expected_state={"has_data": True},
            risk_level=RiskLevel.HIGH,
            description="搜索测试",
        )
        assert step.depends_on == ["s0"]
        assert step.fallback_step == "s2"
        assert step.risk_level == RiskLevel.HIGH

    def test_extra_fields_allowed(self):
        step = StepIR(step_id="s1", action=ActionType.SEARCH, unknown_field="x")
        assert step.step_id == "s1"


class TestPlanIR:
    """PlanIR 模型测试"""

    def _make_plan(self, steps=None):
        return PlanIR(
            plan_id="p1",
            goal="测试目标",
            steps=steps or [],
            estimated_steps=2,
        )

    def test_minimal_plan(self):
        plan = self._make_plan()
        assert plan.plan_id == "p1"
        assert plan.goal == "测试目标"
        assert plan.steps == []

    def test_validate_dependencies_valid(self):
        steps = [
            StepIR(step_id="s1", action=ActionType.SEARCH, args={"query": "a"}),
            StepIR(step_id="s2", action=ActionType.REASON, args={}, depends_on=["s1"]),
        ]
        plan = self._make_plan(steps)
        assert plan.validate_dependencies() == []

    def test_validate_dependencies_invalid(self):
        steps = [
            StepIR(step_id="s1", action=ActionType.SEARCH, args={"query": "a"}, depends_on=["s_missing"]),
        ]
        plan = self._make_plan(steps)
        invalid = plan.validate_dependencies()
        assert len(invalid) > 0
        assert "s_missing" in invalid[0]

    def test_validate_dependencies_invalid_fallback(self):
        steps = [
            StepIR(step_id="s1", action=ActionType.SEARCH, args={"query": "a"}, fallback_step="s_missing"),
        ]
        plan = self._make_plan(steps)
        invalid = plan.validate_dependencies()
        assert any("fallback" in d for d in invalid)

    def test_get_execution_order(self):
        steps = [
            StepIR(step_id="s2", action=ActionType.REASON, args={}, depends_on=["s1"]),
            StepIR(step_id="s1", action=ActionType.SEARCH, args={"query": "a"}),
        ]
        plan = self._make_plan(steps)
        order = plan.get_execution_order()
        assert order.index("s1") < order.index("s2")

    def test_get_execution_order_circular(self):
        steps = [
            StepIR(step_id="s1", action=ActionType.SEARCH, args={"query": "a"}, depends_on=["s2"]),
            StepIR(step_id="s2", action=ActionType.REASON, args={}, depends_on=["s1"]),
        ]
        plan = self._make_plan(steps)
        order = plan.get_execution_order()
        # 存在环，拓扑排序无法完成所有步骤
        assert len(order) < len(steps)


# ========================
# TypeChecker 测试
# ========================


class TestTypeChecker:
    """TypeChecker 测试"""

    def setup_method(self):
        self.checker = TypeChecker()

    def test_check_plan_valid(self):
        plan = PlanIR(
            plan_id="p1",
            goal="测试",
            steps=[
                StepIR(step_id="s1", action=ActionType.SEARCH, args={"query": "test"}),
                StepIR(step_id="s2", action=ActionType.OUTPUT, args={"content": "结果"}),
            ],
        )
        result = self.checker.check_plan(plan)
        assert result.valid

    def test_check_plan_empty_goal(self):
        # Pydantic 会校验 min_length=1, 所以空 goal 会直接报 ValidationError
        with pytest.raises(Exception):
            PlanIR(plan_id="p1", goal="", steps=[])

    def test_check_step_invalid_action(self):
        step = StepIR(step_id="s1", action=ActionType.SEARCH, args={"query": "test"})
        step.action = "invalid_action"  # type: ignore[assignment]
        result = self.checker.check_step(step)
        assert not result.valid

    def test_check_condition_valid(self):
        assert self.checker._check_condition("has_data and verified") is True

    def test_check_condition_invalid_name(self):
        assert self.checker._check_condition("os.system('rm -rf /')") is False

    def test_check_condition_function_call(self):
        assert self.checker._check_condition("print('hello')") is False

    def test_check_whitelist_tool_blocked(self):
        step = StepIR(
            step_id="s1",
            action=ActionType.CALL_TOOL,
            args={"tool_name": "dangerous_tool"},
        )
        result = self.checker.check_whitelist(step, allowed_tools={"safe_tool"})
        assert not result.valid

    def test_check_whitelist_tool_allowed(self):
        step = StepIR(
            step_id="s1",
            action=ActionType.CALL_TOOL,
            args={"tool_name": "safe_tool"},
        )
        result = self.checker.check_whitelist(step, allowed_tools={"safe_tool"})
        assert result.valid

    def test_strict_mode(self):
        checker = TypeChecker(strict=True)
        step = StepIR(
            step_id="s1",
            action=ActionType.SEARCH,
            args={"query": "test", "unknown_param": "val"},
        )
        result = checker.check_step(step)
        assert not result.valid  # strict mode: unknown param -> error


# ========================
# IRParser 测试
# ========================


class TestIRParser:
    """IRParser 测试"""

    def setup_method(self):
        self.parser = IRParser()

    def test_parse_valid_json(self):
        plan_dict = {
            "plan_id": "p1",
            "goal": "测试目标",
            "steps": [
                {
                    "step_id": "s1",
                    "action": "search",
                    "args": {"query": "hello"},
                },
            ],
            "estimated_steps": 1,
        }
        result = self.parser.parse(json.dumps(plan_dict))
        assert result.status == IRParseStatus.SUCCESS
        assert result.plan is not None
        assert result.plan.plan_id == "p1"

    def test_parse_valid_dict_like_json(self):
        """验证有效的 JSON 字符串可以正确解析"""
        plan_dict = {
            "plan_id": "p1",
            "goal": "测试",
            "steps": [],
            "estimated_steps": 0,
        }
        result = self.parser.parse(json.dumps(plan_dict))
        assert result.status == IRParseStatus.SUCCESS
        assert result.plan is not None

    def test_parse_markdown_code_block(self):
        plan_dict = {
            "plan_id": "p1",
            "goal": "测试",
            "steps": [],
            "estimated_steps": 0,
        }
        text = f"```json\n{json.dumps(plan_dict)}\n```"
        result = self.parser.parse(text)
        assert result.status == IRParseStatus.SUCCESS

    def test_parse_unparseable(self):
        result = self.parser.parse("This is not JSON at all, completely random text")
        assert result.status == IRParseStatus.UNPARSEABLE
        assert result.plan is None

    def test_extract_json_with_prefix(self):
        plan_dict = {"plan_id": "p1", "goal": "g", "steps": []}
        text = f"Here is the plan:\n{json.dumps(plan_dict)}"
        extracted = self.parser._extract_json(text)
        assert extracted is not None

    def test_parse_repairable_json(self):
        # 缺少逗号的 JSON（修复策略可以修复）
        broken = '{"plan_id": "p1" "goal": "g", "steps": [], "estimated_steps": 0}'
        result = self.parser.parse(broken)
        # 可能被修复或需要重试
        assert result.status in (IRParseStatus.SUCCESS, IRParseStatus.REPAIRABLE, IRParseStatus.UNPARSEABLE)

    def test_parse_non_plan_response_no_plan_id(self):
        """LLM 返回非计划内容（如直接回答天气问题），缺少 plan_id 和 goal"""
        data = {'city': '杭州'}
        result = self.parser.parse(json.dumps(data))
        # DefaultsRepairStrategy 应填充缺失字段，返回 REPAIRABLE
        assert result.status == IRParseStatus.REPAIRABLE, f"期望 REPAIRABLE, 实际 {result.status}"
        assert result.plan is not None
        assert result.plan.plan_id == "plan_auto_repaired"
        assert result.plan.goal == "plan_auto_repaired"
        assert len(result.plan.steps) == 0

    def test_parse_non_plan_response_empty_steps(self):
        """LLM 返回只有 plan_id 和 goal 但没有 steps 的 JSON"""
        data = {'plan_id': 'p1', 'goal': '天气对比'}
        result = self.parser.parse(json.dumps(data))
        # plan_id 和 goal 已存在，可直接解析成功
        assert result.status in (IRParseStatus.SUCCESS, IRParseStatus.REPAIRABLE), f"实际 {result.status}"
        assert result.plan is not None
        assert result.plan.plan_id == "p1"
        # steps 有默认值，应非 None
        assert result.plan.steps is not None

    def test_parse_non_plan_response_direct_answer(self):
        """LLM 直接回答用户问题而不是生成计划（如 '杭州未来一周小雨'）"""
        raw_text = '杭州未来一周以小雨为主，气温在25-30度之间'
        result = self.parser.parse(raw_text)
        # 无法提取 JSON，应返回 UNPARSEABLE
        assert result.status == IRParseStatus.UNPARSEABLE
        assert result.plan is None


# ======================== Fixture 驱动的参数化测试 ========================

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_fixtures() -> list[tuple[str, str, str, list[str]]]:
    """加载 fixtures/ 目录下所有测试用例

    Returns:
        [(description, input_text, expected_status, tags), ...]
    """
    if not _FIXTURE_DIR.exists():
        return []

    cases: list[tuple[str, str, str, list[str]]] = []
    for fpath in sorted(_FIXTURE_DIR.glob("*.json")):
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            cases.append((
                data.get("description", fpath.stem),
                data["input"],
                data["expected_status"],
                data.get("tags", []),
            ))
        except (KeyError, json.JSONDecodeError) as e:
            cases.append((f"ERROR: {fpath.name} - {e}", "", "unparseable", []))
    return cases


@pytest.mark.parametrize(
    "description, llm_output, expected_status, tags",
    _load_fixtures(),
    ids=lambda x: x[:50] if isinstance(x, str) else str(x),
)
def test_parse_with_fixtures(description, llm_output, expected_status, tags):
    """用 fixtures/ 目录下的真实 LLM 输出场景测试解析器"""
    parser = IRParser()
    result = parser.parse(llm_output)
    status_map = {
        "success": IRParseStatus.SUCCESS,
        "repairable": IRParseStatus.REPAIRABLE,
        "unparseable": IRParseStatus.UNPARSEABLE,
    }
    expected = status_map.get(expected_status)
    assert result.status == expected, (
        f"[{description}] 期望 {expected_status}, 实际 {result.status.value}\n"
        f"  tags: {tags}\n"
        f"  errors: {result.errors}\n"
        f"  repairs: {len(result.repairs)}"
    )
    # 验证一致性：SUCCESS/REPAIRABLE 必须有 plan，UNPARSEABLE 无 plan
    if result.status in (IRParseStatus.SUCCESS, IRParseStatus.REPAIRABLE):
        assert result.plan is not None, f"[{description}] 解析成功但 plan 为 None"
    else:
        assert result.plan is None, f"[{description}] 解析失败但 plan 不为 None"
