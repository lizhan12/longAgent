"""端到端 IR 链路测试

验证 IRParser → StateMachine → LTL → ConstraintValidator → Executor 全链路，
不依赖 LLM，直接构造 PlanIR 对象进行测试。

测试场景：
1. 合法计划路径：INIT→search→HAS_DATA→reason→VERIFIED→summarize→GENERATED→output→DONE
2. 非法计划：INIT→summarize（无数据不能总结，状态机拒绝）
3. LTL 规则拦截：不经过 VERIFIED 就 output（违反 G(output→verified)）
"""

import pytest

from long.ir.constraint_validator import ConstraintValidator, RuntimeCheckContext, ValidationResult
from long.ir.executor import PlanExecutor
from long.ir.ltl import ExecutionHistory, ExecutionStep, LTLValidator
from long.ir.plan_ir import ActionType, PlanIR, RiskLevel, StepIR
from long.ir.state_machine import AgentState, AgentStateMachine, TERMINAL_STATES


# ========================
# 测试场景 1: 合法计划路径
# ========================


def _make_legal_plan() -> PlanIR:
    """构造一个合法的计划：搜索 → 推理 → 总结 → 输出"""
    return PlanIR(
        plan_id="test_legal",
        goal="测试合法路径",
        steps=[
            StepIR(
                step_id="s1", action=ActionType.SEARCH.value,
                args={"query": "test"}, risk_level=RiskLevel.LOW,
                description="搜索数据",
            ),
            StepIR(
                step_id="s2", action=ActionType.REASON.value,
                args={}, risk_level=RiskLevel.LOW,
                depends_on=["s1"], description="推理验证",
            ),
            StepIR(
                step_id="s3", action=ActionType.SUMMARIZE.value,
                args={}, risk_level=RiskLevel.LOW,
                depends_on=["s2"], description="总结结果",
            ),
            StepIR(
                step_id="s4", action=ActionType.OUTPUT.value,
                args={}, risk_level=RiskLevel.LOW,
                depends_on=["s3"], description="输出结果",
            ),
        ],
        constraints=[],
        estimated_steps=4,
    )


def _make_bad_state_machine_plan() -> PlanIR:
    """构造一个非法计划：INIT→summarize（无数据不能总结）"""
    return PlanIR(
        plan_id="test_bad_sm",
        goal="测试状态机拒绝",
        steps=[
            StepIR(
                step_id="s1", action=ActionType.SUMMARIZE.value,
                args={}, risk_level=RiskLevel.LOW,
                description="直接总结（无数据）",
            ),
        ],
        constraints=[],
        estimated_steps=1,
    )


def _make_ltl_violation_plan() -> PlanIR:
    """构造一个违反 LTL 的计划：output 前没有经过 VERIFIED"""
    return PlanIR(
        plan_id="test_ltl_violation",
        goal="测试 LTL 拦截",
        steps=[
            StepIR(
                step_id="s1", action=ActionType.SEARCH.value,
                args={"query": "data"}, risk_level=RiskLevel.LOW,
                description="搜索数据",
            ),
            StepIR(
                step_id="s2", action=ActionType.OUTPUT.value,
                args={}, risk_level=RiskLevel.LOW,
                depends_on=["s1"], description="直接输出（未验证）",
            ),
        ],
        constraints=[],
        estimated_steps=2,
    )


class TestConstraintValidatorChain:
    """ConstraintValidator 全链路编译时验证"""

    def setup_method(self) -> None:
        self.validator = ConstraintValidator()

    def test_legal_plan_passes_compile_time(self):
        """合法计划通过编译时验证"""
        plan = _make_legal_plan()
        result = self.validator.validate_plan(plan)
        assert result.valid, f"合法计划应通过编译时验证，但返回错误: {result.errors}"
        assert len(result.errors) == 0

    def test_bad_state_machine_rejected_at_compile_time(self):
        """非法状态机路径在编译时被拒绝"""
        plan = _make_bad_state_machine_plan()
        result = self.validator.validate_plan(plan)
        assert not result.valid, "INIT→summarize 应被状态机拒绝"
        sm_errors = [e for e in result.errors if "[状态机]" in e]
        assert len(sm_errors) > 0, f"应包含状态机错误，实际: {result.errors}"

    def test_ltl_violation_passes_compile_time_but_fails_runtime(self):
        """LTL 违规在编译时通过，但在运行时被拦截"""
        plan = _make_ltl_violation_plan()

        # 编译时：状态机路径检查通过（search→output 在动作组层面合法）
        compile_result = self.validator.validate_plan(plan)
        assert compile_result.valid, (
            f"LTL 违规计划应在编译时通过（状态机检查合法），"
            f"但返回错误: {compile_result.errors}"
        )

        # 运行时：执行到 output 时，LTL 检查发现未经过 VERIFIED
        context = RuntimeCheckContext(current_state=AgentState.INIT)
        execution_history = context.history

        # 第1步：search → HAS_DATA
        step1 = StepIR(
            step_id="s1", action=ActionType.SEARCH.value,
            args={"query": "data"}, risk_level=RiskLevel.LOW,
        )
        step1_result = self.validator.validate_step_runtime(step1, context)
        assert step1_result.valid, f"search 应在运行时通过: {step1_result.errors}"
        self.validator.update_state(step1, "ok", context)
        execution_history.add_step(ExecutionStep(
            step_id="s1", action="search",
            state_before=AgentState.INIT.value, state_after=AgentState.HAS_DATA.value,
        ))

        # 第2步：output → LTL 检查发现未经过 VERIFIED
        step2 = StepIR(
            step_id="s2", action=ActionType.OUTPUT.value,
            args={}, risk_level=RiskLevel.LOW,
        )
        step2_result = self.validator.validate_step_runtime(step2, context)
        assert step2_result.valid, (
            f"output 的状态机转移应通过（HAS_DATA→DONE 合法），"
            f"但返回错误: {step2_result.errors}"
        )

        # 更新状态到 DONE
        self.validator.update_state(step2, "output done", context)
        execution_history.add_step(ExecutionStep(
            step_id="s2", action="output",
            state_before=AgentState.HAS_DATA.value, state_after=AgentState.DONE.value,
        ))

        # 终态检查：LTL 应该发现 DONE 但未经过 VERIFIED
        final_result = self.validator.validate_final(context)
        ltl_errors = [e for e in final_result.errors if "[LTL终态]" in e]
        assert len(ltl_errors) > 0, (
            f"终态检查应报告 LTL 违规（DONE 但未经过 VERIFIED），"
            f"实际错误: {final_result.errors}"
        )


class TestStateMachineChain:
    """状态机运行时逐步骤验证"""

    def test_legal_path_execution(self):
        """模拟完整合法路径执行，验证每一步状态转移"""
        sm = AgentStateMachine()
        state = AgentState.INIT
        path = [
            ("search", AgentState.HAS_DATA),
            ("reason", AgentState.VERIFIED),
            ("summarize", AgentState.GENERATED),
            ("output", AgentState.DONE),
        ]

        for action, expected_state in path:
            valid, transition, _ = sm.check_transition(state, action)
            assert valid, f"{state.value}--[{action}]-->? 应合法"
            if transition:
                state = transition.to_state
            assert state == expected_state, (
                f"执行 {action} 后预期状态 {expected_state.value}，"
                f"实际 {state.value}"
            )

        assert sm.is_terminal(state), f"DONE 应为终态"

    def test_illegal_action_rejected(self):
        """INIT→summarize 被状态机拒绝"""
        sm = AgentStateMachine()
        valid, _, reason = sm.check_transition(AgentState.INIT, "summarize")
        assert not valid, f"INIT→summarize 应被拒绝，但返回合法"
        assert reason is not None, "应返回拒绝原因"

    def test_terminal_states(self):
        """验证所有终态都被正确识别"""
        for state in AgentState:
            sm = AgentStateMachine()
            if state in TERMINAL_STATES:
                assert sm.is_terminal(state), f"{state.value} 应是终态"
            else:
                assert not sm.is_terminal(state), f"{state.value} 不应是终态"


class TestLTLChain:
    """LTL 验证器运行时时序验证"""

    def test_output_requires_verified_violation(self):
        """output 但未经过 VERIFIED → LTL 违规"""
        ltl = LTLValidator()
        history = ExecutionHistory()

        # 执行 search → HAS_DATA → output → DONE（跳过 VERIFIED）
        history.add_step(ExecutionStep(
            step_id="s1", action="search",
            state_before="INIT", state_after="HAS_DATA",
        ))
        history.add_step(ExecutionStep(
            step_id="s2", action="output",
            state_before="HAS_DATA", state_after="DONE",
        ))

        # 终态检查
        valid, errors = ltl.check_final(history)
        # 应该至少违反 output_requires_verified 和 done_requires_verified
        rule_names = {e.rule_name for e in errors}
        assert "output_requires_verified" in rule_names, (
            f"应报告 output_requires_verified 违规，"
            f"实际违规规则: {rule_names}"
        )
        assert "done_requires_verified" in rule_names, (
            f"应报告 done_requires_verified 违规，"
            f"实际违规规则: {rule_names}"
        )

    def test_legal_path_passes_ltl(self):
        """合法路径通过所有 LTL 检查"""
        ltl = LTLValidator()
        history = ExecutionHistory()

        # 完整合法路径
        for action, state in [
            ("search", "HAS_DATA"),
            ("reason", "VERIFIED"),
            ("summarize", "GENERATED"),
            ("output", "DONE"),
        ]:
            history.add_step(ExecutionStep(
                step_id=f"s_{action}", action=action,
                state_before="", state_after=state,
            ))

        # 运行时检查（跳过 Eventually 规则）
        valid, errors = ltl.check_runtime(history)
        assert valid, f"合法路径应通过运行时检查，但报告错误: {[(e.rule_name, e.message) for e in errors]}"

        # 终态检查
        final_valid, final_errors = ltl.check_final(history)
        assert final_valid, f"合法路径应通过终态检查，但报告错误: {[(e.rule_name, e.message) for e in final_errors]}"

    def test_abort_is_terminal(self):
        """ABORTED 被 LTL 承认为合法终态"""
        ltl = LTLValidator()
        history = ExecutionHistory()

        # 执行到 ABORTED
        history.add_step(ExecutionStep(
            step_id="s1", action="search",
            state_before="INIT", state_after="HAS_DATA",
        ))
        history.add_step(ExecutionStep(
            step_id="s2", action="abort",
            state_before="HAS_DATA", state_after="ABORTED",
        ))

        # 终态检查：ABORTED 是合法终态，不应违反 must_reach_terminal
        valid, errors = ltl.check_final(history)
        # 可能违反 output_requires_verified（没有 output），但不应违反 must_reach_terminal
        must_reach_errors = [e for e in errors if e.rule_name == "must_reach_terminal"]
        assert len(must_reach_errors) == 0, (
            f"ABORTED 是合法终态，不应违反 must_reach_terminal"
        )


class TestPlanExecutorChain:
    """PlanExecutor 端到端链路（模拟执行，不调 LLM）"""

    @pytest.mark.asyncio
    async def test_execute_legal_plan_with_mock_tools(self):
        """使用 mock tool_executor 执行合法计划"""
        executor = PlanExecutor()

        plan = _make_legal_plan()

        # 模拟 CLI adapter 和 tool_executor
        class MockCLI:
            class Console:
                def print(self, *args, **kwargs) -> None: pass
                def status(self, *args, **kwargs):
                    class CM:
                        def __enter__(self): return self
                        def __exit__(self, exc_type, exc_val, exc_tb): pass
                    return CM()
            console = Console()
            prompt_session = None

        tool_results: dict[str, str] = {}
        async def mock_tool_executor(tool_name: str, args: dict) -> str:
            tool_results[tool_name] = "ok"
            return "ok"

        result = await executor.execute_plan(
            plan=plan,
            cli_adapter=MockCLI(),
            tool_executor=mock_tool_executor,
            history_msgs=[],
        )

        assert result.success, f"合法计划应执行成功，但失败: {result.errors}"
        assert result.final_state == AgentState.DONE, (
            f"最终状态应为 DONE，实际: {result.final_state}"
        )
        assert len(result.step_results) == len(plan.steps), (
            f"应执行 {len(plan.steps)} 步，实际 {len(result.step_results)}"
        )
        # 所有步骤应成功
        for sr in result.step_results:
            assert sr.success, f"步骤 {sr.step_id} 应成功，但失败: {sr.error}"

    @pytest.mark.asyncio
    async def test_illegal_plan_rejected_at_compile_time(self):
        """非法计划在编译时被拒绝，不进入执行"""
        executor = PlanExecutor()
        plan = _make_bad_state_machine_plan()

        # validate_plan 应该失败
        validation = executor.constraint_validator.validate_plan(plan)
        assert not validation.valid, "INIT→summarize 应在编译时被拒绝"
        sm_errors = [e for e in validation.errors if "[状态机]" in e]
        assert len(sm_errors) > 0