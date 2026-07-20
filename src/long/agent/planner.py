"""PlannerAgent — 旗舰模型专司规划

LLM 驱动的 TaskIR 生成，不调用任何工具。
级联回退：LLM (FLAGSHIP) → LLM (FAST) → 规则兜底。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from long.cognitive.task_ir import (
    SubtaskIR,
    TaskIR,
    get_task_ir_generation_prompt,
    parse_task_ir_from_message,
)

logger = logging.getLogger(__name__)


class PlannerAgent:
    """规划 Agent — 旗舰模型，不调用工具

    职责：将模糊用户意图拆解为可执行的 TaskIR。
    优先使用 FLAGSHIP tier 模型 + Structured Output 生成规划，
    失败则级联回退到规则解析。

    用法:
        planner = PlannerAgent(llm_chat_fn=llm.chat, cascade_router=router)
        task_ir = await planner.plan("帮我分析最新的AI趋势并生成报告")
    """

    def __init__(
        self,
        llm_chat_fn: Any = None,
        cascade_router: Any = None,
    ) -> None:
        self._llm_chat = llm_chat_fn
        self._cascade_router = cascade_router

    async def plan(self, user_message: str, tools: list[dict[str, Any]] | None = None) -> TaskIR:
        """生成执行计划

        Args:
            user_message: 用户原始消息
            tools: 可用工具列表（仅用于上下文，不传给 LLM）

        Returns:
            TaskIR 实例
        """
        if self._cascade_router is not None:
            task_ir = await self._cascade_plan(user_message)
            if task_ir is not None:
                return task_ir

        if self._llm_chat is not None:
            task_ir = await self._llm_plan(user_message)
            if task_ir is not None:
                return task_ir

        return self._fallback_plan(user_message)

    async def _cascade_plan(self, user_message: str) -> TaskIR | None:
        """通过级联路由器规划（FLAGSHIP → FAST → EDGE）"""
        prompt = get_task_ir_generation_prompt(user_message)
        try:
            response = await self._cascade_router.route(
                [{"role": "user", "content": prompt}],
                purpose="plan",
                response_format={"type": "json_object"},
            )
            content = getattr(response, "content", "") or ""
            return self._parse_llm_response(content)
        except Exception as e:
            logger.warning("PlannerAgent 级联规划失败: %s", e)
            return None

    async def _llm_plan(self, user_message: str) -> TaskIR | None:
        """LLM 直接规划（默认模型，不传 tools）"""
        prompt = get_task_ir_generation_prompt(user_message)
        try:
            response = await self._llm_chat(
                [{"role": "user", "content": prompt}],
                purpose="plan",
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=2048,
            )
            content = getattr(response, "content", "") or ""
            return self._parse_llm_response(content)
        except Exception as e:
            logger.warning("PlannerAgent LLM 规划失败: %s", e)
            return None

    def _parse_llm_response(self, content: str) -> TaskIR | None:
        """解析 LLM 返回的 JSON 为 TaskIR"""
        if not content:
            return None

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        task = TaskIR(
            goal=data.get("goal", ""),
            constraints=data.get("constraints", []),
            deliverables=data.get("deliverables", []),
        )

        subtasks_data = data.get("subtasks", [])
        if not subtasks_data:
            return None

        sub_by_index: dict[int, SubtaskIR] = {}
        for idx, sub_data in enumerate(subtasks_data):
            subtask = SubtaskIR(
                description=sub_data.get("description", ""),
                tool_hint=sub_data.get("tool_hint"),
            )
            task.subtasks.append(subtask)
            sub_by_index[idx] = subtask

        for idx, sub_data in enumerate(subtasks_data):
            deps = sub_data.get("depends_on", [])
            for dep_idx in deps:
                if isinstance(dep_idx, int) and dep_idx < idx and dep_idx in sub_by_index:
                    dep_id = sub_by_index[dep_idx].id
                    if dep_id not in task.subtasks[idx].depends_on:
                        task.subtasks[idx].depends_on.append(dep_id)

        return task

    def _fallback_plan(self, user_message: str) -> TaskIR:
        """规则兜底：使用 parse_task_ir_from_message"""
        logger.info("PlannerAgent 回退到规则解析")
        return parse_task_ir_from_message(user_message)