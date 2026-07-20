"""形式化状态机

定义 Agent 的状态和状态转换规则。
使用条件路由替代硬编码转移表，支持语义化决策。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentState(str, Enum):
    """Agent 状态枚举"""

    INIT = "INIT"
    HAS_DATA = "HAS_DATA"
    VERIFIED = "VERIFIED"
    GENERATED = "GENERATED"
    APPROVED = "APPROVED"
    DONE = "DONE"
    ABORTED = "ABORTED"
    CANCELLED = "CANCELLED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"


class StateTransition(BaseModel):
    """状态转换定义（兼容旧接口）"""

    action: str = Field(..., description="触发动作")
    from_state: AgentState | str = Field(..., description="源状态")
    to_state: AgentState = Field(..., description="目标状态")
    preconditions: list[str] = Field(default_factory=list, description="前置条件")
    side_effects: dict[str, Any] = Field(default_factory=dict, description="副作用")

    model_config = {"extra": "forbid"}


class InvalidTransitionError(Exception):
    """无效状态转换异常"""

    def __init__(
        self,
        from_state: AgentState,
        action: str,
        reason: str | None = None,
    ) -> None:
        self.from_state = from_state
        self.action = action
        self.reason = reason
        message = f"无效的状态转换: {from_state.value} --[{action}]--> ?"
        if reason:
            message += f" ({reason})"
        super().__init__(message)


TERMINAL_STATES = {
    AgentState.DONE,
    AgentState.ABORTED,
    AgentState.CANCELLED,
    AgentState.BUDGET_EXCEEDED,
}

DATA_ACQUISITION_ACTIONS = {"search", "call_api", "call_tool", "call_mcp", "call_skill"}
DATA_PROCESSING_ACTIONS = {"reason", "summarize"}
OUTPUT_ACTIONS = {"output"}
CONTROL_ACTIONS = {"abort", "cancel", "budget_exceeded"}
APPROVAL_ACTIONS = {"wait_approval"}

_DATA_STATES = {AgentState.INIT, AgentState.HAS_DATA, AgentState.VERIFIED}
_VERIFY_STATES = {AgentState.HAS_DATA, AgentState.VERIFIED, AgentState.GENERATED}
_OUTPUT_STATES = {AgentState.HAS_DATA, AgentState.VERIFIED, AgentState.GENERATED, AgentState.APPROVED}


def _resolve_target_state(from_state: AgentState, action: str) -> AgentState | None:
    """根据当前状态和动作，解析目标状态

    Returns:
        目标状态，如果转移不合法则返回 None
    """
    if action in CONTROL_ACTIONS:
        control_targets = {
            "abort": AgentState.ABORTED,
            "cancel": AgentState.CANCELLED,
            "budget_exceeded": AgentState.BUDGET_EXCEEDED,
        }
        return control_targets[action]

    if action in DATA_ACQUISITION_ACTIONS:
        if from_state in _DATA_STATES:
            return AgentState.HAS_DATA
        return None

    if action == "reason":
        if from_state == AgentState.INIT:
            return AgentState.VERIFIED
        if from_state == AgentState.HAS_DATA:
            return AgentState.VERIFIED
        if from_state == AgentState.VERIFIED:
            return AgentState.VERIFIED
        if from_state == AgentState.GENERATED:
            return AgentState.VERIFIED
        return None

    if action == "summarize":
        if from_state in {AgentState.HAS_DATA, AgentState.VERIFIED}:
            return AgentState.GENERATED
        return None

    if action == "output":
        if from_state in _OUTPUT_STATES:
            return AgentState.DONE
        return None

    if action == "wait_approval":
        if from_state in {AgentState.VERIFIED, AgentState.GENERATED}:
            return AgentState.APPROVED
        return None

    return None


DEFAULT_TRANSITIONS: list[StateTransition] = []


class AgentStateMachine:
    """Agent 状态机

    使用条件路由替代硬编码转移表。
    动作分组驱动转移决策，支持语义化路由。

    Attributes:
        transitions: 兼容旧接口的自定义转换配置
    """

    def __init__(
        self,
        config: list[StateTransition] | None = None,
    ) -> None:
        self.transitions = config or []
        self._custom_table: dict[tuple[AgentState, str], StateTransition] = {}
        self._custom_wildcards: dict[str, StateTransition] = {}
        if config:
            self._build_from_config(config)

    def _build_from_config(self, config: list[StateTransition]) -> None:
        self._custom_table.clear()
        self._custom_wildcards.clear()
        for transition in config:
            if transition.from_state == "*":
                self._custom_wildcards[transition.action] = transition
            else:
                from_state = (
                    transition.from_state
                    if isinstance(transition.from_state, AgentState)
                    else AgentState(transition.from_state)
                )
                self._custom_table[(from_state, transition.action)] = transition

    def can_transition(self, state: AgentState, action: str, context: dict | None = None) -> bool:
        """条件路由：判断状态转移是否合法

        Args:
            state: 当前状态
            action: 要执行的动作
            context: 可选的语义路由上下文

        Returns:
            是否允许转移
        """
        if state in TERMINAL_STATES:
            return False
        if action in CONTROL_ACTIONS:
            return True
        target = _resolve_target_state(state, action)
        if target is not None:
            return True
        if context and context.get("semantic_routing"):
            return self._semantic_check(state, action, context)
        if self._custom_table and (state, action) in self._custom_table:
            return True
        if self._custom_wildcards and action in self._custom_wildcards:
            return True
        return False

    def _semantic_check(self, state: AgentState, action: str, context: dict) -> bool:
        """语义路由检查：允许模型根据 context 自主决策"""
        if state in TERMINAL_STATES:
            return False
        if action in CONTROL_ACTIONS:
            return True
        return bool(context.get("force_allow", False))

    def get_allowed_actions(self, state: AgentState) -> list[str]:
        """获取当前状态允许的动作

        Args:
            state: 当前状态

        Returns:
            允许的动作列表
        """
        actions: set[str] = set()

        actions.update(CONTROL_ACTIONS)

        if state in _DATA_STATES:
            actions.update(DATA_ACQUISITION_ACTIONS)

        if state in _VERIFY_STATES:
            actions.add("reason")

        if state in {AgentState.HAS_DATA, AgentState.VERIFIED}:
            actions.add("summarize")

        if state in _OUTPUT_STATES:
            actions.add("output")

        if state in {AgentState.VERIFIED, AgentState.GENERATED}:
            actions.add("wait_approval")

        for (s, action) in self._custom_table:
            if s == state:
                actions.add(action)
        actions.update(self._custom_wildcards.keys())

        return list(actions)

    def get_illegal_actions(self, state: AgentState) -> list[str]:
        """获取当前状态禁止的动作

        Args:
            state: 当前状态

        Returns:
            禁止的动作列表
        """
        all_known = (
            DATA_ACQUISITION_ACTIONS
            | DATA_PROCESSING_ACTIONS
            | OUTPUT_ACTIONS
            | CONTROL_ACTIONS
            | APPROVAL_ACTIONS
        )
        allowed = set(self.get_allowed_actions(state))
        return list(all_known - allowed)

    def check_transition(
        self,
        state: AgentState,
        action: str,
    ) -> tuple[bool, StateTransition | None, str | None]:
        """检查状态转换是否合法

        Args:
            state: 当前状态
            action: 要执行的动作

        Returns:
            (是否合法, 转换规则, 错误原因)
        """
        if state in TERMINAL_STATES:
            return False, None, f"状态 {state.value} 是终态，不允许转换"

        if (state, action) in self._custom_table:
            return True, self._custom_table[(state, action)], None

        if action in self._custom_wildcards:
            return True, self._custom_wildcards[action], None

        target = _resolve_target_state(state, action)
        if target is not None:
            transition = StateTransition(
                action=action,
                from_state=state,
                to_state=target,
            )
            return True, transition, None

        allowed = self.get_allowed_actions(state)
        return False, None, f"动作 '{action}' 在状态 {state.value} 下不允许。允许的动作: {allowed}"

    def validate_plan_path(
        self,
        steps: list[Any],
        initial_state: AgentState = AgentState.INIT,
    ) -> tuple[bool, list[str]]:
        """静态验证计划路径

        Args:
            steps: 步骤列表（需要有 action 属性）
            initial_state: 初始状态

        Returns:
            (是否有效, 错误列表)
        """
        errors: list[str] = []
        current = initial_state

        for i, step in enumerate(steps):
            action = getattr(step, "action", None) or step.get("action", "")
            valid, transition, reason = self.check_transition(current, action)

            if not valid:
                errors.append(f"步骤 {i + 1} ({action}): {reason}")

            if valid and transition is not None:
                current = transition.to_state

        return len(errors) == 0, errors

    def is_terminal(self, state: AgentState) -> bool:
        """检查是否为终态

        Args:
            state: 状态

        Returns:
            是否为终态
        """
        return state in TERMINAL_STATES

    def get_transition(
        self,
        state: AgentState,
        action: str,
    ) -> StateTransition | None:
        """获取转换规则

        Args:
            state: 当前状态
            action: 动作

        Returns:
            转换规则，如果不存在则返回 None
        """
        if (state, action) in self._custom_table:
            return self._custom_table[(state, action)]
        if action in self._custom_wildcards:
            return self._custom_wildcards[action]
        target = _resolve_target_state(state, action)
        if target is not None:
            return StateTransition(
                action=action,
                from_state=state,
                to_state=target,
            )
        return None
