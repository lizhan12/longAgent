"""TypeChecker - 类型检查器

对 PlanIR 和 StepIR 进行类型和约束检查。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .plan_ir import ActionType, PlanIR, StepIR


def validate_args(action: str, args: dict[str, Any]) -> tuple[bool, list[str]]:
    """校验动作参数类型。

    对已知 action 进行基本类型检查，未知 action 放行。
    """
    errors: list[str] = []
    # 已知参数的类型约束
    type_constraints: dict[str, dict[str, type]] = {
        "search": {"query": str, "top_k": int},
        "call_tool": {"tool_name": str, "parameters": dict},
        "call_api": {"endpoint": str, "method": str, "body": dict},
        "call_mcp": {"server_name": str, "tool_name": str, "arguments": dict},
        "call_skill": {"skill_name": str, "tool_name": str, "arguments": dict},
    }
    constraints = type_constraints.get(action)
    if constraints is None:
        return True, []
    for key, expected_type in constraints.items():
        if key in args:
            if not isinstance(args[key], expected_type):
                errors.append(
                    f"参数 '{key}' 类型应为 {expected_type.__name__}，"
                    f"实际为 {type(args[key]).__name__}"
                )
    return len(errors) == 0, errors


@dataclass
class TypeCheckResult:
    """类型检查结果"""

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

    def merge(self, other: TypeCheckResult) -> None:
        """合并另一个检查结果"""
        if not other.valid:
            self.valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


class TypeChecker:
    """类型检查器

    执行以下检查：
    1. ActionType 枚举完整性
    2. 参数类型 Pydantic 校验
    3. 参数白名单校验
    4. 依赖关系完整性
    """

    ALLOWED_PARAM_KEYS: set[str] = {
        "query",
        "top_k",
        "filters",
        "search_type",
        "tool_name",
        "parameters",
        "timeout",
        "endpoint",
        "method",
        "headers",
        "body",
        "query_params",
        "server_name",
        "arguments",
        "skill_name",
        "reasoning_type",
        "premises",
        "max_steps",
        "content",
        "max_length",
        "style",
        "format",
        "destination",
        "request_type",
        "message",
        "options",
        "timeout_seconds",
    }

    def __init__(self, strict: bool = False) -> None:
        """初始化类型检查器

        Args:
            strict: 是否启用严格模式（未知参数报错而非警告）
        """
        self.strict = strict

    def check_plan(self, plan: PlanIR) -> TypeCheckResult:
        """检查整个计划

        Args:
            plan: 计划 IR

        Returns:
            类型检查结果
        """
        result = TypeCheckResult(valid=True)

        if not plan.plan_id:
            result.add_error("plan_id 不能为空")

        if not plan.goal:
            result.add_error("goal 不能为空")

        if not plan.steps:
            result.add_warning("计划没有步骤")

        invalid_deps = plan.validate_dependencies()
        if invalid_deps:
            repairs = plan.auto_fix_dependencies()
            if repairs:
                for repair in repairs:
                    result.add_warning(f"[自动修复] {repair}")
                still_invalid = plan.validate_dependencies()
                if still_invalid:
                    for dep in still_invalid:
                        result.add_error(f"无效的依赖关系: {dep}")
                else:
                    result.add_warning(
                        f"自动修复了 {len(repairs)} 个依赖关系问题，"
                        f"原始问题: {', '.join(invalid_deps)}"
                    )
            else:
                for dep in invalid_deps:
                    result.add_error(f"无效的依赖关系: {dep}")

        step_ids = set()
        for step in plan.steps:
            if step.step_id in step_ids:
                result.add_error(f"重复的 step_id: {step.step_id}")
            step_ids.add(step.step_id)

            step_result = self.check_step(step)
            result.merge(step_result)

        return result

    def check_step(self, step: StepIR) -> TypeCheckResult:
        """检查单个步骤

        Args:
            step: 步骤 IR

        Returns:
            类型检查结果
        """
        result = TypeCheckResult(valid=True)

        if not step.step_id:
            result.add_error("step_id 不能为空")

        try:
            ActionType(step.action)
        except ValueError:
            result.add_error(f"无效的 action 类型: {step.action}")

        args_result = self._check_args(step.action, step.args)
        result.merge(args_result)

        if step.condition:
            cond_result = self._check_condition(step.condition)
            if not cond_result:
                result.add_warning(f"条件表达式可能无效: {step.condition}")

        return result

    def _check_args(self, action: str, args: dict[str, Any]) -> TypeCheckResult:
        """检查参数

        Args:
            action: 动作类型
            args: 参数字典

        Returns:
            类型检查结果
        """
        result = TypeCheckResult(valid=True)

        if not isinstance(args, dict):
            result.add_error(f"args 必须是字典，实际类型: {type(args).__name__}")
            return result

        for key in args:
            if key not in self.ALLOWED_PARAM_KEYS:
                msg = f"未知参数 '{key}' (action: {action})"
                if self.strict:
                    result.add_error(msg)
                else:
                    result.add_warning(msg)

        valid, errors = validate_args(action, args)
        if not valid:
            for error in errors:
                result.add_error(f"参数校验失败: {error}")

        return result

    def _check_condition(self, condition: str) -> bool:
        """检查条件表达式语法

        Args:
            condition: 条件表达式

        Returns:
            表达式是否有效
        """
        allowed_names = {
            "has_data",
            "verified",
            "approved",
            "error_count",
            "tokens_used",
            "True",
            "False",
            "and",
            "or",
            "not",
        }

        try:
            import ast
            tree = ast.parse(condition, mode="eval")

            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if node.id not in allowed_names:
                        return False
                elif isinstance(node, ast.Call):
                    return False
                elif isinstance(node, ast.Attribute):
                    return False
            return True
        except SyntaxError:
            return False

    def check_whitelist(
        self,
        step: StepIR,
        allowed_tools: set[str] | None = None,
        allowed_apis: set[str] | None = None,
        allowed_mcps: set[str] | None = None,
        allowed_skills: set[str] | None = None,
    ) -> TypeCheckResult:
        """检查白名单

        Args:
            step: 步骤 IR
            allowed_tools: 允许的工具列表
            allowed_apis: 允许的 API 列表
            allowed_mcps: 允许的 MCP 服务器列表
            allowed_skills: 允许的 Skill 列表

        Returns:
            类型检查结果
        """
        result = TypeCheckResult(valid=True)

        if step.action == ActionType.CALL_TOOL.value:
            if allowed_tools is not None:
                tool_name = step.args.get("tool_name", "")
                if not tool_name:
                    result.add_error("call_tool 缺少 tool_name 参数")
                elif tool_name not in allowed_tools:
                    result.add_error(f"工具 '{tool_name}' 不在白名单中")

        elif step.action == ActionType.CALL_API.value:
            if allowed_apis is not None:
                endpoint = step.args.get("endpoint", "")
                if endpoint not in allowed_apis:
                    result.add_error(f"API '{endpoint}' 不在白名单中")

        elif step.action == ActionType.CALL_MCP.value:
            if allowed_mcps is not None:
                server_name = step.args.get("server_name", "")
                if server_name not in allowed_mcps:
                    result.add_error(f"MCP 服务器 '{server_name}' 不在白名单中")

        elif step.action == ActionType.CALL_SKILL.value:
            if allowed_skills is not None:
                skill_name = step.args.get("skill_name", "")
                if skill_name not in allowed_skills:
                    result.add_error(f"Skill '{skill_name}' 不在白名单中")

        return result
