"""Eval 数据管理

管理评估数据集: 公开集、隐藏集和轮换集。
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from .report import EvalCategory, EvalTask

logger = logging.getLogger(__name__)


class EvalDatasetManager:
    """评估数据集管理器

    管理公开集、隐藏集和轮换集，防止数据污染。

    Attributes:
        _public_set: 公开测试集
        _hidden_set: 隐藏测试集（不入 git）
        _rotated_set: 当前轮换使用的测试集
        _last_rotation: 上次轮换时间
    """

    def __init__(
        self,
        public_set: list[EvalTask] | None = None,
        hidden_set: list[EvalTask] | None = None,
        rotation_period_days: int = 30,
    ) -> None:
        self._public_set = public_set or []
        self._hidden_set = hidden_set or []
        self._rotated_set: list[EvalTask] = []
        self._rotation_period_days = rotation_period_days
        self._last_rotation: float = 0.0

    @property
    def public_set(self) -> list[EvalTask]:
        return list(self._public_set)

    @property
    def hidden_set(self) -> list[EvalTask]:
        return list(self._hidden_set)

    def add_to_public(self, task: EvalTask) -> None:
        """添加任务到公开集"""
        self._public_set.append(task)

    def add_to_hidden(self, task: EvalTask) -> None:
        """添加任务到隐藏集"""
        task.category = EvalCategory.BOUNDARY  # 隐藏集标记为边界
        self._hidden_set.append(task)

    def get_eval_set(
        self,
        include_hidden: bool = False,
    ) -> list[EvalTask]:
        """获取测试集

        Args:
            include_hidden: 是否包含隐藏集

        Returns:
            测试任务列表
        """
        tasks = list(self._public_set)

        if include_hidden:
            tasks.extend(self._hidden_set)

        if self._rotated_set:
            tasks.extend(self._rotated_set)

        return tasks

    def rotate_dataset(self, period_days: int | None = None) -> int:
        """定期轮换测试集

        从隐藏集中选取部分任务加入轮换集。

        Args:
            period_days: 轮换周期（天），None 使用默认值

        Returns:
            轮换的任务数
        """
        days = period_days or self._rotation_period_days

        # 检查是否需要轮换
        now = time.time()
        elapsed_days = (now - self._last_rotation) / 86400

        if elapsed_days < days and self._rotated_set:
            return 0

        if not self._hidden_set:
            return 0

        # 基于时间戳选择任务
        seed = int(now / 86400) % len(self._hidden_set)
        num_to_rotate = max(1, len(self._hidden_set) // 3)

        self._rotated_set = []
        for i in range(num_to_rotate):
            idx = (seed + i) % len(self._hidden_set)
            self._rotated_set.append(self._hidden_set[idx])

        self._last_rotation = now
        logger.info("Rotated %d tasks into eval set", len(self._rotated_set))

        return len(self._rotated_set)

    def compute_task_hash(self, task: EvalTask) -> str:
        """计算任务哈希（用于去重和验证）

        Args:
            task: 评估任务

        Returns:
            哈希值
        """
        content = f"{task.name}:{task.input}:{task.expected}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
