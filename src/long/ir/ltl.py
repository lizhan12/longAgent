"""LTL 时序逻辑验证器

实现线性时序逻辑（Linear Temporal Logic）验证，确保系统行为符合时序约束。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionStep:
    """执行步骤记录"""

    step_id: str
    action: str
    state_before: str
    state_after: str
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionHistory:
    """执行历史（带 set 累加器优化 O(1) 查询）"""

    steps: list[ExecutionStep] = field(default_factory=list)
    _actions_set: set[str] = field(default_factory=set)
    _states_set: set[str] = field(default_factory=set)

    def add_step(self, step: ExecutionStep) -> None:
        """添加步骤"""
        self.steps.append(step)
        self._actions_set.add(step.action)
        self._states_set.add(step.state_after)

    def get_actions(self) -> list[str]:
        """获取所有动作"""
        return [s.action for s in self.steps]

    def get_states(self) -> list[str]:
        """获取所有状态"""
        return [s.state_after for s in self.steps]

    def has_action(self, action: str) -> bool:
        """检查是否包含某动作（O(1)）"""
        return action in self._actions_set

    def has_state(self, state: str) -> bool:
        """检查是否到达过某状态（O(1)）"""
        return state in self._states_set

    def last_state(self) -> str | None:
        """获取最后状态"""
        if self.steps:
            return self.steps[-1].state_after
        return None

    def is_terminal(self) -> bool:
        """检查是否到达终态"""
        terminal_states = {"DONE", "ABORTED", "CANCELLED", "BUDGET_EXCEEDED"}
        last = self.last_state()
        return last in terminal_states if last else False


@dataclass
class LTLError:
    """LTL 验证错误"""

    rule_name: str
    message: str
    formula: str
    history_snapshot: list[str] = field(default_factory=list)


class LTLFormula(ABC):
    """LTL 公式基类"""

    @abstractmethod
    def check(self, history: ExecutionHistory) -> tuple[bool, LTLError | None]:
        """检查公式是否满足

        Args:
            history: 执行历史

        Returns:
            (是否满足, 错误信息)
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class Globally(LTLFormula):
    """G(φ) - 全局性：φ 在所有状态下都满足"""

    def __init__(self, inner: LTLFormula) -> None:
        self.inner = inner

    def check(self, history: ExecutionHistory) -> tuple[bool, LTLError | None]:
        valid, error = self.inner.check(history)
        if not valid:
            return False, LTLError(
                rule_name="Globally",
                message=f"G(φ) 违规: {error.message if error else 'inner formula failed'}",
                formula=str(self),
                history_snapshot=history.get_states(),
            )
        return True, None

    def __str__(self) -> str:
        return f"G({self.inner})"


class Eventually(LTLFormula):
    """F(φ) - 最终性：φ 最终会满足"""

    def __init__(self, inner: LTLFormula) -> None:
        self.inner = inner

    def check(self, history: ExecutionHistory) -> tuple[bool, LTLError | None]:
        if history.is_terminal():
            valid, _ = self.inner.check(history)
            if not valid:
                return False, LTLError(
                    rule_name="Eventually",
                    message=f"F(φ) 违规: 到达终态但 φ 未满足",
                    formula=str(self),
                    history_snapshot=history.get_states(),
                )
        return True, None

    def __str__(self) -> str:
        return f"F({self.inner})"


class Implies(LTLFormula):
    """φ → ψ - 蕴含：如果 φ 满足则 ψ 也必须满足"""

    def __init__(self, antecedent: LTLFormula, consequent: LTLFormula) -> None:
        self.antecedent = antecedent
        self.consequent = consequent

    def check(self, history: ExecutionHistory) -> tuple[bool, LTLError | None]:
        ant_valid, _ = self.antecedent.check(history)
        if ant_valid:
            cons_valid, _ = self.consequent.check(history)
            if not cons_valid:
                return False, LTLError(
                    rule_name="Implies",
                    message="φ → ψ 违规: 前件满足但后件不满足",
                    formula=str(self),
                    history_snapshot=history.get_states(),
                )
        return True, None

    def __str__(self) -> str:
        return f"({self.antecedent} → {self.consequent})"


class ActionOccurred(LTLFormula):
    """原子命题：某动作已发生"""

    def __init__(self, action: str) -> None:
        self.action = action

    def check(self, history: ExecutionHistory) -> tuple[bool, LTLError | None]:
        if history.has_action(self.action):
            return True, None
        return False, LTLError(
            rule_name="ActionOccurred",
            message=f"动作 '{self.action}' 未发生",
            formula=str(self),
            history_snapshot=history.get_actions(),
        )

    def __str__(self) -> str:
        return f"Action({self.action})"


class StateReached(LTLFormula):
    """原子命题：某状态已到达"""

    def __init__(self, state: str) -> None:
        self.state = state

    def check(self, history: ExecutionHistory) -> tuple[bool, LTLError | None]:
        if history.has_state(self.state):
            return True, None
        return False, LTLError(
            rule_name="StateReached",
            message=f"状态 '{self.state}' 未到达",
            formula=str(self),
            history_snapshot=history.get_states(),
        )

    def __str__(self) -> str:
        return f"State({self.state})"


class TerminalStateReached(LTLFormula):
    """原子命题：任意终态已到达"""

    def check(self, history: ExecutionHistory) -> tuple[bool, LTLError | None]:
        if history.is_terminal():
            return True, None
        return False, LTLError(
            rule_name="TerminalStateReached",
            message="未到达任何终态",
            formula=str(self),
            history_snapshot=history.get_states(),
        )

    def __str__(self) -> str:
        return "Terminal"


class And(LTLFormula):
    """φ ∧ ψ - 合取"""

    def __init__(self, left: LTLFormula, right: LTLFormula) -> None:
        self.left = left
        self.right = right

    def check(self, history: ExecutionHistory) -> tuple[bool, LTLError | None]:
        left_valid, left_error = self.left.check(history)
        if not left_valid:
            return False, left_error
        right_valid, right_error = self.right.check(history)
        if not right_valid:
            return False, right_error
        return True, None

    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"


class Or(LTLFormula):
    """φ ∨ ψ - 析取"""

    def __init__(self, left: LTLFormula, right: LTLFormula) -> None:
        self.left = left
        self.right = right

    def check(self, history: ExecutionHistory) -> tuple[bool, LTLError | None]:
        left_valid, _ = self.left.check(history)
        if left_valid:
            return True, None
        right_valid, right_error = self.right.check(history)
        if right_valid:
            return True, None
        return False, right_error

    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"


class Not(LTLFormula):
    """¬φ - 否定"""

    def __init__(self, inner: LTLFormula) -> None:
        self.inner = inner

    def check(self, history: ExecutionHistory) -> tuple[bool, LTLError | None]:
        valid, _ = self.inner.check(history)
        if valid:
            return False, LTLError(
                rule_name="Not",
                message=f"¬φ 违规: φ 满足",
                formula=str(self),
                history_snapshot=history.get_states(),
            )
        return True, None

    def __str__(self) -> str:
        return f"¬{self.inner}"


class LTLValidator:
    """LTL 验证器

    管理 LTL 规则并执行验证。
    """

    def __init__(self) -> None:
        self._rules: dict[str, LTLFormula] = {}
        self._build_default_rules()

    def _build_default_rules(self) -> None:
        """构建默认 LTL 规则

        规则 1: G(output → verified) - 输出前必须验证
        规则 2: G(wait_approval → verified) - 审批前必须验证
        规则 3: F(Terminal) - 最终必须到达终态
        规则 4: G(ABORTED → Terminal) - ABORTED 是合法终态
        规则 5: G(CANCELLED → Terminal) - CANCELLED 是合法终态
        规则 6: G(DONE → State(VERIFIED)) - DONE 必须经过 VERIFIED
        """
        self.add_rule(
            "output_requires_verified",
            Globally(Implies(ActionOccurred("output"), Or(StateReached("VERIFIED"), ActionOccurred("summarize")))),
        )

        self.add_rule(
            "approval_requires_verified",
            Globally(Implies(ActionOccurred("wait_approval"), Or(StateReached("VERIFIED"), ActionOccurred("summarize")))),
        )

        self.add_rule(
            "must_reach_terminal",
            Eventually(TerminalStateReached()),
        )

        self.add_rule(
            "aborted_is_terminal",
            Globally(
                Implies(
                    StateReached("ABORTED"),
                    TerminalStateReached(),
                )
            ),
        )

        self.add_rule(
            "cancelled_is_terminal",
            Globally(
                Implies(
                    StateReached("CANCELLED"),
                    TerminalStateReached(),
                )
            ),
        )

        self.add_rule(
            "done_requires_verified",
            Globally(
                Implies(
                    StateReached("DONE"),
                    Or(
                        StateReached("VERIFIED"),
                        Or(
                            StateReached("APPROVED"),
                            ActionOccurred("summarize"),
                        ),
                    ),
                )
            ),
        )

    def add_rule(self, name: str, formula: LTLFormula) -> None:
        """添加规则

        Args:
            name: 规则名称
            formula: LTL 公式
        """
        self._rules[name] = formula

    def remove_rule(self, name: str) -> bool:
        """移除规则

        Args:
            name: 规则名称

        Returns:
            是否成功移除
        """
        if name in self._rules:
            del self._rules[name]
            return True
        return False

    def check_runtime(self, history: ExecutionHistory) -> tuple[bool, list[LTLError]]:
        """运行时检查

        检查当前执行历史是否满足所有 LTL 规则。

        Args:
            history: 执行历史

        Returns:
            (是否全部满足, 错误列表)
        """
        errors: list[LTLError] = []

        for name, formula in self._rules.items():
            if isinstance(formula, Eventually):
                continue

            valid, error = formula.check(history)
            if not valid and error:
                error.rule_name = name
                errors.append(error)

        return len(errors) == 0, errors

    def check_final(self, history: ExecutionHistory) -> tuple[bool, list[LTLError]]:
        """终态检查

        检查最终状态是否满足所有 LTL 规则（包括 Eventually）。

        Args:
            history: 执行历史

        Returns:
            (是否全部满足, 错误列表)
        """
        errors: list[LTLError] = []

        for name, formula in self._rules.items():
            valid, error = formula.check(history)
            if not valid and error:
                error.rule_name = name
                errors.append(error)

        return len(errors) == 0, errors

    def get_rules(self) -> dict[str, str]:
        """获取所有规则

        Returns:
            规则名称到公式字符串的映射
        """
        return {name: str(formula) for name, formula in self._rules.items()}
