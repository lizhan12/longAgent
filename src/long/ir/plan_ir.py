"""PlanIR - 计划中间表示

定义 AI Agent 执行计划的结构化表示。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field, model_validator


class ActionType(str, Enum):
    """动作类型枚举"""

    SEARCH = "search"
    CALL_API = "call_api"
    CALL_TOOL = "call_tool"
    CALL_MCP = "call_mcp"
    CALL_SKILL = "call_skill"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EXECUTE_FILE = "execute_file"
    REASON = "reason"
    SUMMARIZE = "summarize"
    OUTPUT = "output"
    WAIT_APPROVAL = "wait_approval"
    SKIP = "skip"


class RiskLevel(str, Enum):
    """风险等级"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StepIR(BaseModel):
    """步骤中间表示

    表示计划中的单个执行步骤。

    Attributes:
        step_id: 步骤唯一标识符
        action: 动作类型
        args: 动作参数
        depends_on: 依赖的步骤 ID 列表
        condition: 执行条件表达式
        fallback_step: 失败时的回退步骤 ID
        expected_state: 预期状态变更
        risk_level: 风险等级
        description: 步骤描述
    """

    step_id: str = Field(..., min_length=1, description="步骤唯一标识符")
    action: ActionType = Field(..., description="动作类型")
    args: dict[str, Any] = Field(default_factory=dict, description="动作参数")
    depends_on: list[str] = Field(default_factory=list, description="依赖的步骤 ID")
    condition: str | None = Field(default=None, description="执行条件表达式")
    fallback_step: str | None = Field(default=None, description="失败时的回退步骤")
    expected_state: dict[str, Any] = Field(
        default_factory=dict, description="预期状态变更"
    )
    risk_level: RiskLevel = Field(default=RiskLevel.LOW, description="风险等级")
    description: str | None = Field(default=None, description="步骤描述")

    model_config = {
        "extra": "allow",
        "use_enum_values": True,
    }


class Constraint(BaseModel):
    """约束定义"""

    type: str = Field(default="custom", description="约束类型")
    value: Any = Field(default=None, description="约束值")
    description: str | None = Field(default=None, description="约束描述")

    model_config = {
        "extra": "allow",
    }


class PlanIR(BaseModel):
    """计划中间表示

    表示完整的执行计划。

    Attributes:
        plan_id: 计划唯一标识符
        goal: 计划目标描述
        steps: 步骤列表
        constraints: 约束条件列表
        estimated_steps: 预估步骤数
        metadata: 元数据
    """

    plan_id: str = Field(..., min_length=1, description="计划唯一标识符")
    goal: str = Field(..., min_length=1, description="计划目标描述")
    steps: list[StepIR] = Field(default_factory=list, description="步骤列表")
    constraints: list[Constraint] = Field(
        default_factory=list, description="约束条件"
    )
    estimated_steps: int = Field(default=0, ge=0, description="预估步骤数")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")

    model_config = {
        "extra": "allow",
    }

    ACTION_TYPE_ALIASES: ClassVar[dict[str, str]] = {
        "search": "search",
        "query": "search",
        "tavily_search": "search",
        "web_search": "search",
        "web_search_query": "search",
        "internet_search": "search",
        "api": "call_api",
        "call_api": "call_api",
        "http_request": "call_api",
        "tool": "call_tool",
        "call_tool": "call_tool",
        "execute_code": "call_tool",
        "execute_file": "call_tool",
        "run_code": "call_tool",
        "run_script": "call_tool",
        "write_file": "call_tool",
        "read_file": "call_tool",
        "delete_file": "call_tool",
        "mcp": "call_mcp",
        "call_mcp": "call_mcp",
        "skill": "call_skill",
        "call_skill": "call_skill",
        "reason": "reason",
        "think": "reason",
        "analyze": "reason",
        "plan": "reason",
        "verify": "reason",
        "summarize": "summarize",
        "summary": "summarize",
        "output": "output",
        "print": "output",
        "respond": "output",
        "reply": "output",
        "wait": "wait_approval",
        "approve": "wait_approval",
        "wait_approval": "wait_approval",
        "human_review": "wait_approval",
    }

    ARGS_FIELD_ALIASES: ClassVar[dict[str, str]] = {
        "tool": "tool_name",
        "name": "tool_name",
        "search_query": "query",
        "q": "query",
        "source": "code",
        "script": "code",
        "file_path": "path",
        "filepath": "path",
        "body": "content",
        "text": "content",
        "lang": "language",
    }

    @model_validator(mode="before")
    @classmethod
    def _coerce_and_normalize(cls, data: Any) -> Any:
        if isinstance(data, dict) and "constraints" in data:
            coerced = []
            for c in data["constraints"]:
                if isinstance(c, str):
                    coerced.append({"type": "custom", "value": c, "description": c})
                else:
                    coerced.append(c)
            data["constraints"] = coerced

        if isinstance(data, dict) and "steps" in data:
            for step in data["steps"]:
                if not isinstance(step, dict):
                    continue

                if "action" in step:
                    original = step["action"]
                    normalized = original.lower().replace("-", "_").replace(" ", "_")
                    if normalized in cls.ACTION_TYPE_ALIASES:
                        mapped = cls.ACTION_TYPE_ALIASES[normalized]
                        if mapped != original:
                            step["action"] = mapped
                            # 当 action 被映射为 call_tool 时，自动补充 tool_name
                            if mapped == "call_tool" and "args" in step and isinstance(step["args"], dict):
                                args = step["args"]
                                if "tool_name" not in args:
                                    args["tool_name"] = normalized
                                # 将 parameters 子字典中的参数提升到 args 顶层
                                if "parameters" in args and isinstance(args["parameters"], dict):
                                    params = args.pop("parameters")
                                    for key, value in params.items():
                                        if key not in args:
                                            args[key] = value

                if "args" in step and isinstance(step["args"], dict):
                    args = step["args"]
                    renamed = {}
                    for key, value in args.items():
                        mapped = cls.ARGS_FIELD_ALIASES.get(key, key)
                        renamed[mapped] = value
                    step["args"] = renamed

        return data

    @classmethod
    def build_structured_output_schema(cls) -> dict[str, Any]:
        """构建适用于 OpenAI Structured Outputs 的 JSON Schema

        生成扁平化的、无 $ref 的 Schema，兼容 json_schema 模式。
        """
        return {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "计划唯一标识符"},
                "goal": {"type": "string", "description": "计划目标描述"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_id": {"type": "string", "description": "步骤唯一标识符"},
                            "action": {
                                "type": "string",
                                "enum": [e.value for e in ActionType],
                                "description": "动作类型",
                            },
                            "args": {
                                "type": "object",
                                "description": "动作参数",
                                "additionalProperties": True,
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "依赖的步骤 ID",
                            },
                            "condition": {
                                "type": "string",
                                "description": "执行条件表达式",
                                "nullable": True,
                            },
                            "fallback_step": {
                                "type": "string",
                                "description": "失败时的回退步骤",
                                "nullable": True,
                            },
                            "risk_level": {
                                "type": "string",
                                "enum": [e.value for e in RiskLevel],
                                "description": "风险等级",
                            },
                            "description": {
                                "type": "string",
                                "description": "步骤描述",
                                "nullable": True,
                            },
                        },
                        "required": ["step_id", "action"],
                    },
                    "description": "步骤列表",
                },
                "constraints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "description": "约束类型"},
                            "value": {"description": "约束值"},
                            "description": {
                                "type": "string",
                                "description": "约束描述",
                                "nullable": True,
                            },
                        },
                        "required": ["type"],
                    },
                    "description": "约束条件",
                },
                "estimated_steps": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "预估步骤数",
                },
                "metadata": {
                    "type": "object",
                    "description": "元数据",
                    "additionalProperties": True,
                },
            },
            "required": ["plan_id", "goal", "steps"],
        }

    def get_step(self, step_id: str) -> StepIR | None:
        """根据 ID 获取步骤"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_step_index(self, step_id: str) -> int | None:
        """获取步骤索引"""
        for i, step in enumerate(self.steps):
            if step.step_id == step_id:
                return i
        return None

    def validate_dependencies(self) -> list[str]:
        """验证依赖关系，返回无效的依赖 ID 列表"""
        valid_ids = {step.step_id for step in self.steps}
        invalid_deps = []

        for step in self.steps:
            for dep_id in step.depends_on:
                if dep_id not in valid_ids:
                    invalid_deps.append(f"{step.step_id} -> {dep_id}")

            if step.fallback_step and step.fallback_step not in valid_ids:
                invalid_deps.append(f"{step.step_id} (fallback) -> {step.fallback_step}")

        return invalid_deps

    def auto_fix_dependencies(self) -> list[str]:
        """自动修复无效的依赖关系

        移除引用不存在 step_id 的 depends_on 和 fallback_step。
        返回修复描述列表。
        """
        valid_ids = {step.step_id for step in self.steps}
        repairs: list[str] = []

        for step in self.steps:
            original_deps = list(step.depends_on)
            step.depends_on = [d for d in step.depends_on if d in valid_ids]
            removed = set(original_deps) - set(step.depends_on)
            for r_dep in removed:
                repairs.append(f"{step.step_id}.depends_on 移除无效引用 {r_dep!r}")

            if step.fallback_step and step.fallback_step not in valid_ids:
                repairs.append(
                    f"{step.step_id}.fallback_step 移除无效引用 {step.fallback_step!r}"
                )
                step.fallback_step = None

        return repairs

    def get_execution_order(self) -> list[str]:
        """获取拓扑排序后的执行顺序"""
        in_degree: dict[str, int] = {step.step_id: 0 for step in self.steps}
        graph: dict[str, list[str]] = {step.step_id: [] for step in self.steps}

        for step in self.steps:
            for dep_id in step.depends_on:
                if dep_id in graph:
                    graph[dep_id].append(step.step_id)
                    in_degree[step.step_id] += 1

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result
