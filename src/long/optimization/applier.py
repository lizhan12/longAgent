"""变更应用

应用优化变更，检测回归，支持回滚。
真正修改系统配置，维护配置版本栈。
"""

from __future__ import annotations

import copy
import json
import logging
import time
from typing import Any

from .base import OptimizationProposal, OptimizationTarget

logger = logging.getLogger(__name__)


class ConfigSnapshot:
    """配置快照，用于回滚"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.timestamp = time.time()
        self.config = copy.deepcopy(config)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "config": self.config,
        }


class ChangeApplier:
    """变更应用器

    应用优化变更，检测回归，支持回滚。
    维护配置版本栈，支持回退到任意版本。

    Attributes:
        _applied: 已应用的变更
        _snapshots: 应用前的配置快照
        _config_stack: 配置版本栈
        _system_config: 当前系统配置引用
    """

    def __init__(self, system_config: dict[str, Any] | None = None) -> None:
        self._applied: dict[str, OptimizationProposal] = {}
        self._snapshots: dict[str, ConfigSnapshot] = {}
        self._config_stack: list[ConfigSnapshot] = []
        self._system_config: dict[str, Any] = system_config or {}

    def set_system_config(self, config: dict[str, Any]) -> None:
        self._system_config = config

    def apply(
        self,
        proposal: OptimizationProposal,
    ) -> dict[str, Any]:
        """应用变更

        根据优化目标类型，修改对应的系统配置。

        Args:
            proposal: 优化提案

        Returns:
            应用结果
        """
        snapshot = ConfigSnapshot(self._system_config)
        self._snapshots[proposal.change] = snapshot
        self._config_stack.append(snapshot)

        try:
            changes_made = self._apply_by_target(proposal)

            self._applied[proposal.change] = proposal

            logger.info(
                "Applied optimization: target=%s, change=%s, changes=%s",
                proposal.target.value,
                proposal.change[:50],
                changes_made,
            )

            return {
                "success": True,
                "target": proposal.target.value,
                "change": proposal.change,
                "changes_made": changes_made,
                "metrics_after": {},
            }

        except Exception as e:
            logger.error("Failed to apply optimization: %s", e)
            self._rollback_to_snapshot(snapshot)
            return {
                "success": False,
                "error": str(e),
            }

    def _apply_by_target(self, proposal: OptimizationProposal) -> dict[str, Any]:
        """根据目标类型应用变更

        Returns:
            实际修改的配置项
        """
        changes: dict[str, Any] = {}

        if proposal.target == OptimizationTarget.PROMPT:
            changes = self._apply_prompt_change(proposal)

        elif proposal.target == OptimizationTarget.ROUTING:
            changes = self._apply_routing_change(proposal)

        elif proposal.target == OptimizationTarget.BUDGET:
            changes = self._apply_budget_change(proposal)

        elif proposal.target == OptimizationTarget.TOOL:
            changes = self._apply_tool_change(proposal)

        return changes

    def _apply_prompt_change(self, proposal: OptimizationProposal) -> dict[str, Any]:
        """应用 Prompt 优化变更"""
        llm_config = self._system_config.setdefault("llm", {})
        models_config = llm_config.setdefault("models", {})

        change_data = self._parse_change_data(proposal.change)
        changes: dict[str, Any] = {}

        if "temperature" in change_data:
            for purpose, model_cfg in models_config.items():
                old_temp = model_cfg.get("temperature", 0.7)
                model_cfg["temperature"] = change_data["temperature"]
                changes[f"models.{purpose}.temperature"] = {
                    "old": old_temp,
                    "new": change_data["temperature"],
                }

        if "max_tokens" in change_data:
            for purpose, model_cfg in models_config.items():
                old_tokens = model_cfg.get("max_tokens", 4096)
                model_cfg["max_tokens"] = change_data["max_tokens"]
                changes[f"models.{purpose}.max_tokens"] = {
                    "old": old_tokens,
                    "new": change_data["max_tokens"],
                }

        return changes

    def _apply_routing_change(self, proposal: OptimizationProposal) -> dict[str, Any]:
        """应用路由优化变更"""
        ir_config = self._system_config.setdefault("ir", {})
        routing = ir_config.setdefault("routing", {})

        change_data = self._parse_change_data(proposal.change)
        changes: dict[str, Any] = {}

        if "complexity_threshold" in change_data:
            old_val = routing.get("complexity_threshold", {})
            routing["complexity_threshold"] = change_data["complexity_threshold"]
            changes["ir.routing.complexity_threshold"] = {
                "old": old_val,
                "new": change_data["complexity_threshold"],
            }

        if "max_plan_retries" in change_data:
            old_val = ir_config.get("max_plan_retries", 2)
            ir_config["max_plan_retries"] = change_data["max_plan_retries"]
            changes["ir.max_plan_retries"] = {
                "old": old_val,
                "new": change_data["max_plan_retries"],
            }

        return changes

    def _apply_budget_change(self, proposal: OptimizationProposal) -> dict[str, Any]:
        """应用预算优化变更"""
        llm_config = self._system_config.setdefault("llm", {})
        budget = llm_config.setdefault("budget", {})

        change_data = self._parse_change_data(proposal.change)
        changes: dict[str, Any] = {}

        if "max_tokens_per_task" in change_data:
            old_val = budget.get("max_tokens_per_task", 100000)
            budget["max_tokens_per_task"] = change_data["max_tokens_per_task"]
            changes["llm.budget.max_tokens_per_task"] = {
                "old": old_val,
                "new": change_data["max_tokens_per_task"],
            }

        if "daily_token_limit" in change_data:
            old_val = budget.get("daily_token_limit", 1000000)
            budget["daily_token_limit"] = change_data["daily_token_limit"]
            changes["llm.budget.daily_token_limit"] = {
                "old": old_val,
                "new": change_data["daily_token_limit"],
            }

        if "max_tokens_per_request" in change_data:
            old_val = budget.get("max_tokens_per_request", 16384)
            budget["max_tokens_per_request"] = change_data["max_tokens_per_request"]
            changes["llm.budget.max_tokens_per_request"] = {
                "old": old_val,
                "new": change_data["max_tokens_per_request"],
            }

        return changes

    def _apply_tool_change(self, proposal: OptimizationProposal) -> dict[str, Any]:
        """应用工具优化变更"""
        tools_config = self._system_config.setdefault("tools", {})

        change_data = self._parse_change_data(proposal.change)
        changes: dict[str, Any] = {}

        if "priority" in change_data:
            priority = tools_config.setdefault("priority", {})
            for tool_name, weight in change_data["priority"].items():
                old_val = priority.get(tool_name, 1.0)
                priority[tool_name] = weight
                changes[f"tools.priority.{tool_name}"] = {
                    "old": old_val,
                    "new": weight,
                }

        if "disabled" in change_data:
            disabled = tools_config.setdefault("disabled", [])
            for tool_name in change_data["disabled"]:
                if tool_name not in disabled:
                    disabled.append(tool_name)
                    changes[f"tools.disabled.{tool_name}"] = {"old": False, "new": True}

        if "timeout" in change_data:
            timeout = tools_config.setdefault("timeout", {})
            for tool_name, timeout_val in change_data["timeout"].items():
                old_val = timeout.get(tool_name, 30)
                timeout[tool_name] = timeout_val
                changes[f"tools.timeout.{tool_name}"] = {
                    "old": old_val,
                    "new": timeout_val,
                }

        return changes

    def _parse_change_data(self, change_str: str) -> dict[str, Any]:
        """解析变更描述中的结构化数据

        尝试从 change 字符串中提取 JSON 格式的变更数据。
        如果不是 JSON，返回空字典。
        """
        try:
            if "{" in change_str and "}" in change_str:
                start = change_str.index("{")
                end = change_str.rindex("}") + 1
                return json.loads(change_str[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return {}

    def _rollback_to_snapshot(self, snapshot: ConfigSnapshot) -> None:
        """回滚到指定快照"""
        self._system_config.clear()
        self._system_config.update(copy.deepcopy(snapshot.config))

    def rollback(
        self,
        proposal: OptimizationProposal,
    ) -> bool:
        """回滚变更

        Args:
            proposal: 要回滚的优化提案

        Returns:
            是否成功回滚
        """
        snapshot = self._snapshots.get(proposal.change)
        if snapshot is None:
            logger.warning("No snapshot found for rollback: %s", proposal.change[:50])
            return False

        try:
            self._rollback_to_snapshot(snapshot)
            self._applied.pop(proposal.change, None)

            logger.info(
                "Rolled back optimization: target=%s, change=%s",
                proposal.target.value,
                proposal.change[:50],
            )

            return True

        except Exception as e:
            logger.error("Failed to rollback optimization: %s", e)
            return False

    def detect_regression(
        self,
        proposal: OptimizationProposal,
        current_metrics: dict[str, float],
    ) -> bool:
        """检测回归

        对比变更后的指标与变更前的指标，
        如果关键指标下降超过阈值，则检测到回归。
        """
        before = proposal.metrics_before
        regression_threshold = 0.1

        for key, before_value in before.items():
            current_value = current_metrics.get(key)
            if current_value is None:
                continue

            if "rate" in key or "score" in key:
                if before_value > 0 and current_value < before_value * (1 - regression_threshold):
                    logger.warning(
                        "Regression detected: %s dropped from %.3f to %.3f",
                        key, before_value, current_value,
                    )
                    return True

            elif "steps" in key or "duration" in key:
                if before_value > 0 and current_value > before_value * (1 + regression_threshold):
                    logger.warning(
                        "Regression detected: %s increased from %.3f to %.3f",
                        key, before_value, current_value,
                    )
                    return True

        return False

    def get_applied_changes(self) -> list[str]:
        """获取已应用的变更列表"""
        return list(self._applied.keys())

    def get_config_history(self) -> list[dict[str, Any]]:
        """获取配置变更历史"""
        return [s.to_dict() for s in self._config_stack]

    def get_current_config(self) -> dict[str, Any]:
        """获取当前配置"""
        return copy.deepcopy(self._system_config)
