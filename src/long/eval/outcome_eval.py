"""Outcome Eval - 结果层评估

评估输出结果的准确性、Schema 合法性和约束满足情况。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .report import EvalTask, OutcomeResult

logger = logging.getLogger(__name__)


class OutcomeEvaluator:
    """结果层评估器

    评估输出结果的准确性、Schema 合法性和约束满足情况。

    Attributes:
        strict_mode: 严格模式（Schema 必须完全匹配）
    """

    def __init__(self, strict_mode: bool = False) -> None:
        self.strict_mode = strict_mode

    def evaluate(
        self,
        task: EvalTask,
        output: str | dict[str, Any] | None,
    ) -> OutcomeResult:
        """评估输出结果

        Args:
            task: 评估任务
            output: 实际输出

        Returns:
            结果层评估结果
        """
        if output is None:
            return OutcomeResult(
                accuracy=0.0,
                schema_valid=False,
                constraint_satisfied=False,
                details={"error": "Output is None"},
            )

        accuracy = self._compute_accuracy(task, output)
        schema_valid = self._check_schema(task, output)
        constraint_satisfied = self._check_constraints(task, output)

        return OutcomeResult(
            accuracy=accuracy,
            schema_valid=schema_valid,
            constraint_satisfied=constraint_satisfied,
            details={
                "output_type": type(output).__name__,
                "expected_type": type(task.expected).__name__,
            },
        )

    def _compute_accuracy(
        self,
        task: EvalTask,
        output: str | dict[str, Any],
    ) -> float:
        """计算准确性分数"""
        if task.expected is None:
            return 1.0  # 没有期望输出，无法评估准确性

        if isinstance(task.expected, str) and isinstance(output, str):
            # 字符串精确匹配
            if task.expected.strip() == output.strip():
                return 1.0

            # 部分匹配
            expected_words = set(task.expected.lower().split())
            output_words = set(output.lower().split())

            if not expected_words:
                return 0.0

            overlap = expected_words & output_words
            return len(overlap) / len(expected_words)

        elif isinstance(task.expected, dict) and isinstance(output, dict):
            # 字典匹配
            return self._dict_accuracy(task.expected, output)

        return 0.0

    def _dict_accuracy(
        self,
        expected: dict[str, Any],
        output: dict[str, Any],
    ) -> float:
        """计算字典匹配分数"""
        if not expected:
            return 1.0

        correct = 0
        total = len(expected)

        for key, value in expected.items():
            if key in output:
                if output[key] == value:
                    correct += 1
                elif isinstance(value, str) and isinstance(output[key], str):
                    # 部分字符串匹配
                    if value.lower() in output[key].lower():
                        correct += 0.5

        return correct / total

    def _check_schema(
        self,
        task: EvalTask,
        output: str | dict[str, Any],
    ) -> bool:
        """检查 Schema 合法性"""
        if isinstance(output, dict):
            return True  # 字典输出视为 Schema 合法

        if isinstance(output, str):
            # 尝试解析为 JSON
            try:
                json.loads(output)
                return True
            except json.JSONDecodeError:
                if self.strict_mode:
                    return False
                return True  # 非严格模式下纯文本也视为合法

        return True

    def _check_constraints(
        self,
        task: EvalTask,
        output: str | dict[str, Any],
    ) -> bool:
        """检查约束满足"""
        # 基本约束检查
        if isinstance(output, str):
            # 检查输出是否为空
            if not output.strip():
                return False

        return True
