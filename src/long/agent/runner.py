"""SubAgentRunner

连接 SubAgentSpec → WorkerAgent → EscalationController → CriticAgent → TaskOrchestrator。
当主 Agent 调用 delegate_subtask 时，SubAgentRunner 创建 WorkerAgent 实例并以异步任务方式执行。

HAR 架构关键改进：
  - CriticAgent 只评估 → 输出 CriticReport (PASS / FAIL + FailureType)
  - EscalationController 路由决策 → RETRY_LOCAL / RETRY_REFINE / REPLAN_DAG / HITL
  - HITL 只出现在 EscalationController 一层
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from . import DelegateTask, SubAgentRegistry, TaskOrchestrator
from .critic import CriticAgent, CriticReport, CriticVerdict
from .escalation import (
    EscalationAction,
    EscalationController,
    EscalationDecision,
    FailureSignal,
    FailureType,
)
from .worker import WorkerAgent, WorkerResult

logger = logging.getLogger(__name__)


class SubAgentRunner:
    """子 Agent 执行器 — HAR P-W-E 拓扑的桥梁

    连接 Spec → Worker → Critic → EscalationController → Orchestrator。
    失败处理遵循 HAR 专家建议的三层分类路由。

    用法:
        runner = SubAgentRunner(
            llm_chat_fn=llm.chat,
            llm_chat_with_tools_fn=llm.chat_with_tools,
            registry=SubAgentRegistry(),
            all_tools=gathered_tools,
            tool_handlers=tool_handler_map,
            orchestrator=TaskOrchestrator(),
            escalation=EscalationController(),
        )
        result = await runner.delegate("search_agent", "搜索AI最新进展")
    """

    def __init__(
        self,
        llm_chat_fn: Any,
        llm_chat_with_tools_fn: Any,
        registry: SubAgentRegistry,
        all_tools: list[dict[str, Any]],
        tool_handlers: dict[str, Any],
        orchestrator: TaskOrchestrator | None = None,
        critic: Any = None,
        escalation: EscalationController | None = None,
        on_llm_stats: Any = None,
        on_llm_timeout: Any = None,
        on_llm_fail: Any = None,
    ) -> None:
        self._chat_fn = llm_chat_fn
        self._chat_with_tools_fn = llm_chat_with_tools_fn
        self._registry = registry
        self._all_tools = all_tools
        self._tool_handlers = tool_handlers
        self._orchestrator = orchestrator or TaskOrchestrator()
        self._critic = critic
        self._escalation = escalation or EscalationController()
        self._on_llm_stats = on_llm_stats
        self._on_llm_timeout = on_llm_timeout
        self._on_llm_fail = on_llm_fail

    def _filter_tools(self, tool_names: list[str]) -> list[dict[str, Any]]:
        """从全部工具中筛选出指定名称的工具"""
        if not tool_names:
            return []
        return [t for t in self._all_tools if t.get("function", {}).get("name") in tool_names]

    @staticmethod
    def _build_time_suffix() -> str:
        """构建当前时间后缀 — 追加在 prompt 末尾，遵循前缀缓存原则

        LLM 前缀缓存: 静态 prefix 命中缓存，动态 suffix 仅使末尾失效。
        """
        from datetime import datetime, timedelta as td, timezone

        now = datetime.now(timezone(td(hours=8)))
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        current_date = f"{now.strftime('%Y年%m月%d日')} {weekdays[now.weekday()]} {now.strftime('%H:%M')}"
        iso_date = now.strftime("%Y-%m-%d")
        unix_ts = int(now.timestamp())
        return (
            f"\n\n## 当前时间\n"
            f"- 中文: {current_date}\n"
            f"- ISO: {iso_date}\n"
            f"- Unix: {unix_ts}\n"
            f"- 时区: UTC+08:00\n"
            f"以上时间是系统确认的当前真实时间。当用户提到'今天'、'现在'、'最近'等时间词时，"
            f"请使用上面的时间。搜索时请基于此时间构造 query。如果搜索不到当日实时数据，"
            f"最近的可用数据也可作为参考。"
        )

    def _build_worker(self, sub_agent_name: str, instruction: str, timeout: float) -> WorkerAgent | None:
        spec = self._registry.get(sub_agent_name)
        if spec is None:
            logger.warning("子 Agent '%s' 未注册", sub_agent_name)
            return None

        filtered_tools = self._filter_tools(spec.tools)
        if spec.tools and not filtered_tools:
            logger.warning(
                "子 Agent '%s' 声明的工具 %s 在当前工具集中未找到",
                sub_agent_name, spec.tools,
            )

        return WorkerAgent(
            name=spec.name,
            description=spec.description,
            system_prompt=spec.prompt + self._build_time_suffix(),
            tools=filtered_tools,
            tool_handlers=self._tool_handlers,
            llm_chat_fn=self._chat_fn,
            llm_chat_with_tools_fn=self._chat_with_tools_fn,
            model=spec.model or "",
            max_rounds=getattr(spec, "max_retries", 3) + 1,
            max_tokens=2048,
            timeout=min(timeout, spec.timeout),
        )

    async def delegate(self, sub_agent_name: str, instruction: str, timeout: float = 120.0) -> str:
        """委托子 Agent 执行任务

        HAR 失败路由策略：
          1. Worker 执行 → 成功则返回
          2. 失败 → CriticAgent.review() 纯评估 → CriticReport
          3. CriticReport.is_pass → 返回
          4. CriticReport.is_fail → EscalationController.decide()
             - RETRY_LOCAL: Worker 重试
             - RETRY_REFINE: Worker 带修复提示重试
             - REPLAN_DAG: 标记需要 Planner 重规划
             - HITL: 人工介入

        Args:
            sub_agent_name: 子 Agent 名称
            instruction: 任务指令
            timeout: 整体超时时间

        Returns:
            子 Agent 输出字符串
        """
        worker = self._build_worker(sub_agent_name, instruction, timeout)
        if worker is None:
            available = self._registry.list_names()
            return f"错误: 子 Agent '{sub_agent_name}' 未找到。可用: {', '.join(available)}"

        spec = self._registry.get(sub_agent_name)
        max_retries = getattr(spec, "max_retries", 1) if spec else 1
        retry_count = 0

        for attempt in range(max_retries + 1):
            result = await worker.execute(instruction)
            retry_count = attempt

            if self._on_llm_stats is not None:
                try:
                    self._on_llm_stats(result)
                except Exception:
                    pass

            if result.success:
                return result.output

            if self._critic is not None:
                report = await self._critic.review(result, {
                    "instruction": instruction,
                    "sub_agent_name": sub_agent_name,
                })

                if report.is_pass:
                    return result.output

                failure_type = report.dominant_failure_type or FailureType.EXECUTION
                pattern = self._escalation.detect_pattern(sub_agent_name)

                signal = FailureSignal(
                    failure_type=failure_type,
                    source=f"worker:{sub_agent_name}",
                    description=report.summary,
                    retry_count=retry_count,
                    failure_pattern=pattern,
                    context={
                        "sub_agent_name": sub_agent_name,
                        "instruction": instruction[:200],
                        "issues": [i.description for i in report.issues],
                    },
                )
                decision = self._escalation.decide(signal)

                logger.info(
                    "SubAgentRunner[%s] escalation: %s → %s (reason=%s)",
                    sub_agent_name,
                    failure_type.value,
                    decision.action.value,
                    decision.reason,
                )

                if decision.action == EscalationAction.RETRY_LOCAL:
                    continue

                if decision.action == EscalationAction.RETRY_REFINE:
                    instruction = self._augment_instruction(
                        instruction, decision.repair_hint
                    )
                    continue

                if decision.action == EscalationAction.HITL:
                    return (
                        f"[Worker {sub_agent_name}] 需要人工介入 (HITL): "
                        f"{decision.reason}\n"
                        f"最后输出: {result.output[:500]}"
                    )

                if decision.action in (
                    EscalationAction.REPLAN_NODE,
                    EscalationAction.REPLAN_DAG,
                ):
                    return (
                        f"[Worker {sub_agent_name}] 需要 Planner 重规划 "
                        f"({decision.action.value}): {decision.reason}"
                    )

                if attempt >= max_retries:
                    break

        return f"[Worker {sub_agent_name}] 所有重试已用尽 ({max_retries + 1} 次): {result.output}"

    def _augment_instruction(self, instruction: str, repair_hint: str) -> str:
        """用修复提示增强指令"""
        if not repair_hint:
            return instruction
        return f"{instruction}\n\n[注意] {repair_hint}"

    async def delegate_with_result(self, sub_agent_name: str, instruction: str, timeout: float = 120.0) -> WorkerResult:
        """委托子 Agent 执行任务，返回详细结果"""
        worker = self._build_worker(sub_agent_name, instruction, timeout)
        if worker is None:
            return WorkerResult(output=f"子 Agent '{sub_agent_name}' 未找到", success=False)

        return await worker.execute(instruction)