"""IR 修复策略

定义从 LLM 输出中修复 PlanIR 解析错误的策略。
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from .plan_ir import ActionType


class IRRepairStrategy(ABC):
    """修复策略基类"""

    @abstractmethod
    def can_repair(self, data: dict[str, Any], error: str) -> bool:
        """判断是否可以修复"""
        ...

    @abstractmethod
    def repair(self, data: dict[str, Any]) -> dict[str, Any]:
        """执行修复"""
        ...


class JSONRepairStrategy(IRRepairStrategy):
    """修复 JSON 语法错误"""

    name = "json_repair"

    def can_repair(self, data: dict[str, Any], error: str) -> bool:
        return any(kw in error.lower() for kw in [
            "expecting", "unexpected", "invalid", "extra data",
        ])

    def repair(self, data: dict[str, Any]) -> dict[str, Any]:
        return data  # JSON 修复在解析层处理


class SchemaRepairStrategy(IRRepairStrategy):
    """修复 Schema 不匹配（如 ActionType 枚举值错误）"""

    name = "schema_repair"

    VALID_ACTIONS = {a.value for a in ActionType}

    def can_repair(self, data: dict[str, Any], error: str) -> bool:
        return "action" in error.lower() or "actiontype" in error.lower()

    def repair(self, data: dict[str, Any]) -> dict[str, Any]:
        if "steps" not in data:
            return data

        for step in data["steps"]:
            action = step.get("action", "")
            # 尝试匹配最接近的合法 ActionType
            if action not in self.VALID_ACTIONS:
                best_match = self._find_best_match(action)
                if best_match:
                    step["action"] = best_match
        return data

    def _find_best_match(self, action: str) -> str | None:
        """查找最接近的合法 ActionType"""
        action_lower = action.lower()
        for valid in self.VALID_ACTIONS:
            if valid in action_lower or action_lower in valid:
                return valid
        return None


class DependencyRepairStrategy(IRRepairStrategy):
    """修复 depends_on 引用无效 step_id"""

    name = "dependency_repair"

    def can_repair(self, data: dict[str, Any], error: str) -> bool:
        return "depends_on" in error.lower() or "dependency" in error.lower()

    def repair(self, data: dict[str, Any]) -> dict[str, Any]:
        if "steps" not in data:
            return data

        valid_ids = {step["step_id"] for step in data["steps"] if "step_id" in step}

        for step in data["steps"]:
            deps = step.get("depends_on", [])
            if isinstance(deps, list):
                step["depends_on"] = [d for d in deps if d in valid_ids]
        return data


class DefaultsRepairStrategy(IRRepairStrategy):
    """填充缺失的必需字段"""

    name = "defaults_repair"

    def can_repair(self, data: dict[str, Any], error: str) -> bool:
        return "field required" in error.lower() or "none" in error.lower()

    def repair(self, data: dict[str, Any]) -> dict[str, Any]:
        if "plan_id" not in data or not data["plan_id"]:
            data["plan_id"] = "plan_auto_repaired"
        if "goal" not in data or not data["goal"]:
            data["goal"] = data.get("plan_id", "auto_repaired")
        if "estimated_steps" not in data:
            data["estimated_steps"] = len(data.get("steps", []))
        if "steps" not in data:
            data["steps"] = []

        for step in data.get("steps", []):
            if "step_id" not in step:
                step["step_id"] = f"step_{len(data['steps'])}"
            if "risk_level" not in step:
                step["risk_level"] = "low"
            if "args" not in step:
                step["args"] = {}
            if "depends_on" not in step:
                step["depends_on"] = []

        return data


# 默认修复策略链（按优先级排序）
DEFAULT_REPAIR_STRATEGIES: list[IRRepairStrategy] = [
    DefaultsRepairStrategy(),
    DependencyRepairStrategy(),
    SchemaRepairStrategy(),
    JSONRepairStrategy(),
]