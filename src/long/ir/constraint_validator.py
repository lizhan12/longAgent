"""ConstraintValidator - 约束验证器

整合三层防御：状态机 + LTL + Runtime Check，
提供编译时和运行时的统一约束验证。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ltl import ExecutionHistory, LTLValidator
from .plan_ir import ActionType, PlanIR, StepIR
from .state_machine import AgentState, AgentStateMachine, InvalidTransitionError
from .type_checker import TypeChecker, TypeCheckResult


@dataclass
class ValidationResult:
    """验证结果"""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """添加错误"""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str) -> None:
        """添加警告"""
        self.warnings.append(warning)

    def merge(self, other: ValidationResult) -> None:
        """合并另一个验证结果"""
        if not other.valid:
            self.valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


@dataclass
class RuntimeCheckContext:
    """运行时检查上下文"""

    current_state: AgentState = AgentState.INIT
    history: ExecutionHistory = field(default_factory=ExecutionHistory)
    budget_remaining: int = 100
    step_results: dict[str, Any] = field(default_factory=dict)
    allowed_tools: set[str] | None = None
    allowed_apis: set[str] | None = None
    allowed_mcps: set[str] | None = None
    allowed_skills: set[str] | None = None


class ConstraintValidator:
    """约束验证器

    整合三层防御体系：
    - 第1层: 状态机 (AgentStateMachine) — 编译时路径检查 + 运行时转移检查
    - 第2层: LTL (LTLValidator) — 运行时时序检查 + 终态检查
    - 第3层: Runtime Check — Schema + 权限 + 预算 + 白名单

    Attributes:
        state_machine: Agent 状态机
        ltl_validator: LTL 验证器
        type_checker: 类型检查器
        max_steps: 最大允许步数（预算约束）
    """

    def __init__(
        self,
        state_machine: AgentStateMachine | None = None,
        ltl_validator: LTLValidator | None = None,
        type_checker: TypeChecker | None = None,
        max_steps: int = 100,
    ) -> None:
        """初始化约束验证器

        Args:
            state_machine: Agent 状态机，默认创建实例
            ltl_validator: LTL 验证器，默认创建实例
            type_checker: 类型检查器，默认创建实例
            max_steps: 最大允许步数
        """
        self.state_machine = state_machine or AgentStateMachine()
        self.ltl_validator = ltl_validator or LTLValidator()
        self.type_checker = type_checker or TypeChecker()
        self.max_steps = max_steps

    def validate_plan(self, plan: PlanIR) -> ValidationResult:
        """编译时验证 — 计划提交后的静态检查

        执行：
        1. 状态机路径检查（validate_plan_path）
        2. 类型检查（Schema + 依赖 + 白名单）
        3. 安全约束（高风险操作需 wait_approval）
        4. 预算约束（预估步数不超过 max_steps）
        5. DAG 环检测（依赖关系不能成环）

        Args:
            plan: 计划 IR

        Returns:
            验证结果
        """
        result = ValidationResult(valid=True)

        # 第1层: 状态机路径检查
        sm_valid, sm_errors = self.state_machine.validate_plan_path(plan.steps)
        if not sm_valid:
            for error in sm_errors:
                result.add_error(f"[状态机] {error}")

        # 第2层: 类型检查（Schema + 依赖 + 参数）
        type_result: TypeCheckResult = self.type_checker.check_plan(plan)
        if not type_result.valid:
            result.valid = False
            for error in type_result.errors:
                result.add_error(f"[类型] {error}")
        for warning in type_result.warnings:
            result.add_warning(f"[类型] {warning}")

        has_wait_approval = any(
            step.action == ActionType.WAIT_APPROVAL.value for step in plan.steps
        )
        critical_steps = [
            step for step in plan.steps
            if step.risk_level == "critical"
        ]
        high_risk_steps = [
            step for step in plan.steps
            if step.risk_level == "high"
        ]
        if critical_steps and not has_wait_approval:
            step_ids = ", ".join(s.step_id for s in critical_steps)
            result.add_error(
                f"[安全] 极高风险步骤 [{step_ids}] 需要 wait_approval 步骤"
            )
        if high_risk_steps and not has_wait_approval:
            step_ids = ", ".join(s.step_id for s in high_risk_steps)
            result.add_warning(
                f"[安全] 高风险步骤 [{step_ids}] 建议添加 wait_approval 步骤"
            )

        # 预算约束
        if plan.estimated_steps > self.max_steps:
            result.add_error(
                f"[预算] 预估步数 {plan.estimated_steps} 超过最大允许 {self.max_steps}"
            )

        # DAG 环检测
        execution_order = plan.get_execution_order()
        if len(execution_order) < len(plan.steps):
            result.add_error("[DAG] 步骤依赖关系存在环，无法拓扑排序")

        # write_file content 质量检查：不能是描述文字而非代码
        _CODE_INDICATORS = ("import ", "def ", "class ", "from ", "if __name__", "#!", "plt.", "print(", "open(", "doc.save", ".append(", "return ")
        _DESC_INDICATORS = ("脚本：", "脚本:", "Python脚本", "代码逻辑", "该脚本", "此脚本", "用于", "功能：", "功能:")
        for step in plan.steps:
            action = step.action.lower() if hasattr(step, 'action') else ""
            args = step.args or {}
            content = ""

            # 提取 content 参数
            if action == "write_file":
                content = args.get("content", "")
            elif action == "call_tool" and args.get("tool_name") == "write_file":
                params = args.get("parameters", {})
                if isinstance(params, dict):
                    content = params.get("content", "")
                    if not content:
                        content = args.get("content", "")

            if content and len(content) > 10:
                has_code = any(ind in content for ind in _CODE_INDICATORS)
                has_desc = any(ind in content for ind in _DESC_INDICATORS)
                if has_desc and not has_code:
                    result.add_error(
                        f"[内容] 步骤 {step.step_id} 的 write_file content 是描述文字而非可执行代码，"
                        f"content 参数必须包含完整的可执行代码（如 Python/Shell 脚本），不能只写功能描述"
                    )

        return result

    def validate_step_runtime(
        self,
        step: StepIR,
        context: RuntimeCheckContext,
    ) -> ValidationResult:
        """运行时验证 — 每步执行前的动态检查

        执行：
        1. 状态机转移检查
        2. LTL 时序检查
        3. 类型/Schema 检查
        4. 白名单检查
        5. 预算检查

        Args:
            step: 当前步骤
            context: 运行时上下文

        Returns:
            验证结果
        """
        result = ValidationResult(valid=True)

        # 第1层: 状态机转移检查
        valid, transition, reason = self.state_machine.check_transition(
            context.current_state, step.action
        )
        if not valid:
            result.add_error(
                f"[状态机] {reason or '非法状态转移'}"
            )

        # 第2层: LTL 时序检查（已降级为终态检查，运行时跳过以避免 O(N²) 开销）
        # LTL 验证在 validate_final() 中一次性执行

        # 第3层: Runtime Check

        # 3a. 类型/Schema 检查
        type_result = self.type_checker.check_step(step)
        if not type_result.valid:
            result.valid = False
            for error in type_result.errors:
                result.add_error(f"[Schema] {error}")

        # 3b. 白名单检查
        whitelist_result = self.type_checker.check_whitelist(
            step,
            allowed_tools=context.allowed_tools,
            allowed_apis=context.allowed_apis,
            allowed_mcps=context.allowed_mcps,
            allowed_skills=context.allowed_skills,
        )
        if not whitelist_result.valid:
            result.valid = False
            for error in whitelist_result.errors:
                result.add_error(f"[白名单] {error}")

        # 3c. 预算检查
        if context.budget_remaining <= 0:
            result.add_error("[预算] 执行预算已耗尽")

        return result

    def update_state(
        self,
        step: StepIR,
        step_result: Any,
        context: RuntimeCheckContext,
    ) -> AgentState:
        """执行后更新状态

        根据步骤的动作和结果，更新状态机当前状态。

        Args:
            step: 已执行的步骤
            step_result: 步骤执行结果
            context: 运行时上下文

        Returns:
            新状态
        """
        # 记录步骤结果
        context.step_results[step.step_id] = step_result

        # 状态机转移
        valid, transition, _ = self.state_machine.check_transition(
            context.current_state, step.action
        )
        if valid and transition is not None:
            context.current_state = transition.to_state

        # 扣减预算
        context.budget_remaining -= 1

        return context.current_state

    def validate_final(self, context: RuntimeCheckContext) -> ValidationResult:
        """终态验证 — 任务完成后调用

        检查：
        1. 终态是否合法
        2. LTL 终态规则是否满足
        3. 预算是否超限

        Args:
            context: 运行时上下文

        Returns:
            验证结果
        """
        result = ValidationResult(valid=True)

        # 检查当前是否为终态
        if not self.state_machine.is_terminal(context.current_state):
            result.add_warning(
                f"任务结束于非终态 {context.current_state.value}"
            )

        # LTL 终态检查（包括 Eventually 规则）
        ltl_valid, ltl_errors = self.ltl_validator.check_final(context.history)
        if not ltl_valid:
            for error in ltl_errors:
                result.add_error(f"[LTL终态] {error.message} (规则: {error.rule_name})")

        # 预算超限检查
        if context.budget_remaining <= 0:
            result.add_warning("[预算] 执行预算已耗尽")

        return result
