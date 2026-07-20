"""评估报告模型

定义评估任务、报告和各层评估结果的数据模型。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvalCategory(str, Enum):
    """评估分类"""

    NORMAL = "normal"
    ADVERSARIAL = "adversarial"
    BOUNDARY = "boundary"


class EvalTask(BaseModel):
    """评估任务

    Attributes:
        name: 任务名称
        input: 输入数据
        expected: 期望输出
        category: 评估分类
        difficulty: 难度 (0-1)
        tags: 标签
    """

    name: str
    input: str
    expected: str | dict[str, Any] | None = None
    category: EvalCategory = EvalCategory.NORMAL
    difficulty: float = 0.5
    tags: list[str] = Field(default_factory=list)


class OutcomeResult(BaseModel):
    """结果评估结果

    Attributes:
        accuracy: 准确性分数 (0-1)
        schema_valid: Schema 是否合法
        constraint_satisfied: 约束是否满足
        details: 详细信息
    """

    accuracy: float = 0.0
    schema_valid: bool = True
    constraint_satisfied: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


class ProcessResult(BaseModel):
    """过程评估结果

    Attributes:
        score: 过程分数 (0-1)
        rule_violations: 规则违反数
        ltl_violations: LTL 违反数
        efficiency: 效率分数 (0-1)
        details: 详细信息
    """

    score: float = 0.0
    rule_violations: int = 0
    ltl_violations: int = 0
    efficiency: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class SystemResult(BaseModel):
    """系统评估结果

    Attributes:
        stability: 稳定性分数 (0-1)
        convergence: 收敛性分数 (0-1)
        failure_modes: 失败模式列表
        details: 详细信息
    """

    stability: float = 0.0
    convergence: float = 0.0
    failure_modes: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class EvalReport(BaseModel):
    """评估报告

    Attributes:
        task: 评估任务
        outcome: 结果层评估
        process: 过程层评估
        system: 系统层评估
        needs_human_review: 是否需要人工审核
        score: 综合分数 (0-1)
        auto_reviewed: 是否自动审核
    """

    task: EvalTask
    outcome: OutcomeResult = Field(default_factory=OutcomeResult)
    process: ProcessResult = Field(default_factory=ProcessResult)
    system: SystemResult = Field(default_factory=SystemResult)
    needs_human_review: bool = False
    score: float = 0.0
    auto_reviewed: bool = True
