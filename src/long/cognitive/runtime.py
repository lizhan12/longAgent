"""Cognitive Runtime — 认知运行时

现代 Agent 的核心不是 LLM，而是 Runtime。
Runtime 负责控制流程、状态管理、记忆调度、工具路由。

架构：
                    ┌─────────────────┐
                    │      User       │
                    └────────┬────────┘
                             ↓
                  ┌────────────────────┐
                  │ Intent Understanding│
                  └────────┬───────────┘
                           ↓
                 ┌─────────────────────┐
                 │ Complexity Estimator│
                 └───────┬─────────────┘
                         ↓
        ┌─────────────────────────────────────┐
        │         Cognitive Runtime           │
        │                                     │
        │  StateGraph / TaskIR / Planner      │
        │  Memory / Reflection / Compression  │
        └───────────────┬─────────────────────┘
                        ↓
             ┌────────────────────┐
             │ Execution Runtime  │
             └─────────┬──────────┘
                       ↓
      ┌────────────────────────────────┐
      │ Tools / APIs / Browser / Code │
      └────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .compression import SemanticCompressor
from .planner import PlanResult, TaskPlanner
from .reflection import PlanRepair, StrategyCritique, StrategyCritiqueResult
from .task_ir import TaskIR, parse_task_ir_from_message
from ..observability.tracing import current_trace, SpanStatus

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Execution Mode — 执行模式
# ──────────────────────────────────────────────────────────

class ExecutionMode(str, enum.Enum):
    LIGHTWEIGHT = "lightweight"
    STANDARD = "standard"
    STRUCTURED = "structured"


# ──────────────────────────────────────────────────────────
# State Graph — 状态图（替代 while True 循环）
# ──────────────────────────────────────────────────────────

class NodeKind(str, enum.Enum):
    THINK = "think"
    ACT = "act"
    OBSERVE = "observe"
    REFLECT = "reflect"
    PLAN = "plan"
    OUTPUT = "output"
    ERROR = "error"


@dataclass
class GraphNode:
    name: str
    kind: NodeKind
    handler: Callable[..., Awaitable[dict[str, Any]]]
    transitions: dict[str, str] = field(default_factory=dict)
    max_visits: int = 10
    visit_count: int = 0


@dataclass
class GraphEdge:
    from_node: str
    to_node: str
    condition: str | None = None
    priority: int = 0


class StateGraph:

    def __init__(self, on_span_created: Callable[[dict[str, Any]], Awaitable[None]] | None = None) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._entry: str | None = None
        self._state: dict[str, Any] = {}
        self._history: list[str] = []
        self._on_span_created = on_span_created

    def add_node(self, name: str, kind: NodeKind,
                 handler: Callable[..., Awaitable[dict[str, Any]]],
                 max_visits: int = 10) -> StateGraph:
        self._nodes[name] = GraphNode(
            name=name, kind=kind, handler=handler, max_visits=max_visits,
        )
        return self

    def add_edge(self, from_node: str, to_node: str,
                 condition: str | None = None, priority: int = 0) -> StateGraph:
        self._edges.append(GraphEdge(
            from_node=from_node, to_node=to_node,
            condition=condition, priority=priority,
        ))
        return self

    def set_entry(self, name: str) -> StateGraph:
        self._entry = name
        return self

    def get_next_node(self, current: str, context: dict[str, Any]) -> str | None:
        edges = [e for e in self._edges if e.from_node == current]
        edges.sort(key=lambda e: e.priority, reverse=True)

        for edge in edges:
            if edge.condition is None:
                return edge.to_node
            if edge.condition in context:
                condition_value = context[edge.condition]
                if callable(condition_value):
                    if condition_value(context):
                        return edge.to_node
                elif condition_value:
                    return edge.to_node

        node = self._nodes.get(current)
        if node and node.transitions:
            for cond, target in node.transitions.items():
                if cond in context and context[cond]:
                    return target

        return None

    def checkpoint(self) -> dict[str, Any]:
        return {
            "state": dict(self._state),
            "history": list(self._history),
            "visit_counts": {n: v.visit_count for n, v in self._nodes.items()},
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self._state = snapshot.get("state", {})
        self._history = snapshot.get("history", [])
        for name, count in snapshot.get("visit_counts", {}).items():
            if name in self._nodes:
                self._nodes[name].visit_count = count

    async def run(self, initial_context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._entry:
            raise ValueError("StateGraph: 未设置入口节点")

        context = dict(initial_context or {})
        current = self._entry
        self._history = [current]

        while current:
            node = self._nodes.get(current)
            if node is None:
                logger.error("StateGraph: 节点不存在: %s", current)
                break

            node.visit_count += 1
            if node.visit_count > node.max_visits:
                logger.debug("StateGraph: 节点 %s 访问次数超限 (%d/%d)",
                               current, node.visit_count, node.max_visits)
                context["_loop_detected"] = True
                break

            logger.debug("StateGraph: 执行节点 %s (kind=%s, visit=%d)",
                         current, node.kind.value, node.visit_count)
            try:
                # 为每个节点创建 trace span
                trace = current_trace()
                span_ctx = None
                span = None
                if trace is not None:
                    span_ctx = trace.span(
                        f"cognitive.{node.kind.value}",
                        attributes={"node": current, "kind": node.kind.value, "visit": node.visit_count},
                    )
                    span = await span_ctx.__aenter__()

                try:
                    result = await node.handler(context)
                    context.update(result)
                finally:
                    if span_ctx is not None and span is not None:
                        await span_ctx.__aexit__(None, None, None)
                        # 增量推送：span 完成后通知前端
                        if self._on_span_created is not None:
                            try:
                                await self._on_span_created(span.to_dict())
                            except Exception:
                                pass
            except Exception as e:
                logger.error("StateGraph: 节点 %s 执行失败: %s", current, e)
                context["_error"] = str(e)
                context["_error_node"] = current
                error_node = self.get_next_node(current, {"_error": True})
                if error_node:
                    current = error_node
                    self._history.append(current)
                    continue
                break

            next_node = self.get_next_node(current, context)
            if next_node is None:
                break

            current = next_node
            self._history.append(current)

        return context


# ──────────────────────────────────────────────────────────
# Cognitive Context — 认知上下文
# ──────────────────────────────────────────────────────────

@dataclass
class CognitiveContext:
    """认知运行时的上下文对象

    包含完整的状态信息，替代裸的 history_msgs 列表。
    """

    user_message: str = ""
    intent: dict[str, Any] = field(default_factory=dict)

    current_phase: str = "think"
    round_count: int = 0
    max_rounds: int = 8

    tool_history: list[dict[str, Any]] = field(default_factory=list)
    search_count: int = 0
    max_search_count: int = 3
    last_action_was_search: bool = False

    # 重复工具调用检测
    consecutive_duplicate_calls: int = 0
    last_tool_call_key: str = ""
    consecutive_empty_responses: int = 0  # LLM 连续返回空响应的次数

    messages: list[dict[str, Any]] = field(default_factory=list)

    tool_results: dict[str, str] = field(default_factory=dict)

    reflections: list[str] = field(default_factory=list)
    needs_retry: bool = False
    retry_count: int = 0
    max_retries: int = 3

    final_output: str = ""
    is_complete: bool = False

    errors: list[str] = field(default_factory=list)

    checkpoint_data: dict[str, Any] = field(default_factory=dict)

    task_ir: TaskIR | None = None
    execution_mode: ExecutionMode = ExecutionMode.STANDARD
    plan_result: PlanResult | None = None
    strategy_critique: StrategyCritiqueResult | None = None
    llm_call_count: int = 0
    llm_timeout_count: int = 0
    autonomous_executed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_message": self.user_message,
            "current_phase": self.current_phase,
            "round_count": self.round_count,
            "search_count": self.search_count,
            "tool_history_count": len(self.tool_history),
            "is_complete": self.is_complete,
            "has_errors": len(self.errors) > 0,
            "needs_retry": self.needs_retry,
            "has_task_ir": self.task_ir is not None,
            "execution_mode": self.execution_mode.value,
            "task_progress": self.task_ir.progress_ratio() if self.task_ir else 0.0,
        }


# ──────────────────────────────────────────────────────────
# Reflection — 三层反思系统
# ──────────────────────────────────────────────────────────

class Reflector:
    """三层反思系统

    Layer 1: Execution Validation（工具执行成功/失败）
    Layer 2: Strategy Critique（策略是否合理）
    Layer 3: Plan Repair（是否需要调整 TaskIR）
    """

    def __init__(self, llm_chat_fn: Callable | None = None) -> None:
        self._llm_chat = llm_chat_fn
        self._strategy_critique = StrategyCritique()
        self._plan_repair = PlanRepair()

    async def reflect(self, context: CognitiveContext) -> dict[str, Any]:
        if not context.tool_history:
            return {"needs_retry": False, "reflection": "无工具执行历史"}

        last_tool = context.tool_history[-1]
        tool_name = last_tool.get("name", "")
        tool_result = last_tool.get("result", "")
        tool_error = last_tool.get("error", "")

        layer1 = self._execution_validation(tool_name, tool_result, tool_error, context)

        layer2 = self._strategy_critique.critique(context)
        context.strategy_critique = layer2

        if layer2.needs_plan_repair and context.task_ir:
            repairs = self._plan_repair.repair(context.task_ir, layer2)
            if repairs:
                logger.debug("计划修复: %s", "; ".join(repairs))

        reflection_text = layer1.get("reflection", "")
        if layer2.issues:
            issue_descs = [f"[{i.severity}] {i.description}" for i in layer2.issues[:3]]
            reflection_text += " | 策略问题: " + "; ".join(issue_descs)

        return {
            "needs_retry": layer1.get("needs_retry", False),
            "reflection": reflection_text,
            "retry_strategy": layer1.get("retry_strategy"),
            "needs_plan_repair": layer2.needs_plan_repair,
        }

    def _execution_validation(
        self, tool_name: str, result: str, error: str, context: CognitiveContext
    ) -> dict[str, Any]:
        if error:
            if context.retry_count < context.max_retries:
                return {
                    "needs_retry": True,
                    "reflection": f"工具 {tool_name} 执行失败: {error[:100]}",
                    "retry_strategy": "fix_and_retry",
                }
            return {
                "needs_retry": False,
                "reflection": f"工具 {tool_name} 重试次数已耗尽",
            }

        if tool_name == "tavily_search":
            if not result or len(result) < 20:
                return {
                    "needs_retry": False,
                    "reflection": "搜索无有效结果，应告知用户",
                }
            if result.startswith("[") and ("限制" in result or "约束" in result):
                return {
                    "needs_retry": False,
                    "reflection": "搜索被策略拦截",
                }
            return {
                "needs_retry": False,
                "reflection": "搜索执行成功",
            }

        if tool_name in ("execute_code", "execute_file"):
            if "失败" in result or "Error" in result or "Traceback" in result:
                if context.retry_count < context.max_retries:
                    return {
                        "needs_retry": True,
                        "reflection": "代码执行失败，需要修复",
                        "retry_strategy": "fix_code",
                    }
                return {
                    "needs_retry": False,
                    "reflection": "代码修复次数已耗尽",
                }
            return {
                "needs_retry": False,
                "reflection": "代码执行成功",
            }

        if tool_name == "write_file":
            if "成功" in result or "✅" in result:
                return {
                    "needs_retry": False,
                    "reflection": "文件写入成功",
                }
            if "失败" in result or "错误" in result:
                if context.retry_count < context.max_retries:
                    return {
                        "needs_retry": True,
                        "reflection": "文件写入失败",
                        "retry_strategy": "fix_and_retry",
                    }
                return {
                    "needs_retry": False,
                    "reflection": "文件写入重试耗尽",
                }

        if "成功" in result or "✅" in result:
            return {
                "needs_retry": False,
                "reflection": f"工具 {tool_name} 执行成功",
            }

        return {"needs_retry": False, "reflection": f"工具 {tool_name} 已完成"}


# ──────────────────────────────────────────────────────────
# Tool Router — 工具路由
# ──────────────────────────────────────────────────────────

class ToolRouter:

    def __init__(self) -> None:
        self._search_count = 0
        self._max_search = 2
        self._last_was_search = False
        self._executed_tools: list[str] = []

    def validate_tool_call(
        self, tool_name: str, arguments: dict[str, Any], context: CognitiveContext
    ) -> tuple[bool, str]:
        if tool_name == "tavily_search":
            if context.search_count >= context.max_search_count:
                return False, f"[搜索限制] 已达到最大搜索次数 ({context.max_search_count})"

            if context.last_action_was_search:
                return False, "[ReAct 约束] 搜索后必须先分析结果，不能连续搜索"

        if tool_name == "execute_file":
            # 已有脚本文件（如 skills/qweather/query_weather.py）可以直接执行，
            # 不需要先 write_file。仅当路径以 output/ 开头时才检查。
            path = arguments.get("path", "")
            if path and path.startswith("output/") and "write_file" not in self._executed_tools:
                has_write = any(
                    t["name"] == "write_file" for t in context.tool_history
                )
                if not has_write:
                    return False, "[依赖检查] execute_file 前需要先 write_file"

        if tool_name == "delete_file":
            return False, "[安全] delete_file 需要用户确认"

        return True, ""

    def record_execution(self, tool_name: str, _arguments: dict[str, Any]) -> None:
        self._executed_tools.append(tool_name)
        if tool_name == "tavily_search":
            self._search_count += 1
            self._last_was_search = True
        else:
            self._last_was_search = False


# ──────────────────────────────────────────────────────────
# Cognitive Runtime — 认知运行时（核心）
# ──────────────────────────────────────────────────────────

class CognitiveRuntime:

    def __init__(
        self,
        llm_chat_fn: Callable,
        llm_chat_with_tools_fn: Callable,
        tool_execute_fn: Callable,
        output_fn: Callable,
        memory_search_fn: Callable | None = None,
        memory_store_fn: Callable | None = None,
        memory_controller: Any | None = None,
        tool_capability_registry: Any | None = None,
        planner_agent: Any | None = None,
        on_span_created: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        plan_executor: Any | None = None,
    ) -> None:
        self._llm_chat = llm_chat_fn
        self._llm_chat_with_tools = llm_chat_with_tools_fn
        self._tool_execute = tool_execute_fn
        self._output = output_fn
        self._memory_search = memory_search_fn
        self._memory_store = memory_store_fn
        self._memory = memory_controller
        self._tool_capability = tool_capability_registry
        self._planner_agent = planner_agent
        self._on_span_created = on_span_created
        self._plan_executor = plan_executor

        self.reflector = Reflector(llm_chat_fn)
        self.tool_router = ToolRouter()
        self.compressor = SemanticCompressor(llm_chat_fn=llm_chat_fn)
        self.planner = TaskPlanner()

        self._graph = self._build_graph()
        self._graph._on_span_created = on_span_created

        # 工具调用缓存：key=f"{tool_name}:{json_args}" → value=(result, timestamp)
        self._tool_call_cache: dict[str, tuple[str, float]] = {}

    def _build_graph(self) -> StateGraph:
        graph = StateGraph()

        graph.add_node("think", NodeKind.THINK, self._handle_think)
        graph.add_node("act", NodeKind.ACT, self._handle_act)
        graph.add_node("observe", NodeKind.OBSERVE, self._handle_observe)
        graph.add_node("reflect", NodeKind.REFLECT, self._handle_reflect)
        graph.add_node("plan", NodeKind.PLAN, self._handle_plan)
        graph.add_node("output", NodeKind.OUTPUT, self._handle_output)
        graph.add_node("error", NodeKind.ERROR, self._handle_error, max_visits=3)

        graph.set_entry("think")

        graph.add_edge("think", "act", condition="has_tool_calls", priority=3)
        graph.add_edge("think", "output", condition="has_final_text", priority=2)
        graph.add_edge("think", "think", condition="_retry_think", priority=1)
        graph.add_edge("think", "error", condition="_error", priority=0)

        graph.add_edge("act", "observe")
        graph.add_edge("act", "error", condition="_error")

        graph.add_edge("observe", "reflect")

        graph.add_edge("reflect", "act", condition="needs_retry")
        graph.add_edge("reflect", "plan", condition="trigger_planir")
        graph.add_edge("reflect", "think", condition="should_continue")

        graph.add_edge("plan", "think", condition="should_continue")
        graph.add_edge("plan", "output", condition="is_complete")

        graph.add_edge("output", "think", condition="_retry_think")

        graph.add_edge("error", "think", condition="recovered")
        graph.add_edge("error", "output")

        return graph

    async def run(
        self, context: CognitiveContext, extra: dict[str, Any] | None = None
    ) -> CognitiveContext:
        ctx_dict = context.to_dict()
        ctx_dict["_cognitive_context"] = context
        if extra:
            ctx_dict.update(extra)
        result = await self._graph.run(ctx_dict)

        context.is_complete = result.get("is_complete", False)
        context.final_output = result.get("final_output", "")
        context.errors = result.get("errors", [])

        return context

    def _evaluate_search_sufficiency(self, context: CognitiveContext) -> bool:
        search_results = [
            t for t in context.tool_history
            if t["name"] == "tavily_search"
            and "error" not in t
            and not t.get("result", "").startswith("[")
        ]

        if not search_results:
            return False

        total_content = "".join(t.get("result", "") for t in search_results)

        if len(total_content) < 100:
            return False

        user_entities = re.findall(r'[\u4e00-\u9fff]{2,}', context.user_message)
        user_entities = [e for e in user_entities if e not in (
            "未来", "一周", "怎么样", "如何", "什么", "为什么", "怎么",
            "可以", "能够", "应该", "需要", "帮我", "请问", "查询",
            "搜索", "查找", "了解", "知道", "告诉", "显示", "生成",
            "带有", "包含", "制作", "创建", "写出", "输出",
        )]

        if user_entities:
            matched = sum(
                1 for e in user_entities
                if e in total_content
            )
            entity_coverage = matched / len(user_entities)
            if entity_coverage < 0.3:
                return False

        has_specific_data = bool(re.search(r'\d+[℃°%元万]|\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?|\d+\.\d+', total_content))

        return has_specific_data or len(total_content) > 500

    def _build_task_context_message(self, context: CognitiveContext) -> dict[str, str] | None:
        if not context.task_ir:
            return None

        task_text = context.task_ir.to_prompt_text()

        hint = ""
        if context.plan_result and context.plan_result.strategy_hint:
            hint = f"\n\n策略提示：{context.plan_result.strategy_hint}"

        return {
            "role": "system",
            "content": f"## 任务执行状态\n{task_text}{hint}",
        }

    async def _ensure_task_ir(self, context: CognitiveContext) -> None:
        if context.task_ir is not None:
            return

        if self._planner_agent is not None:
            try:
                context.task_ir = await self._planner_agent.plan(
                    context.user_message, tools=None
                )
            except Exception:
                logger.debug("PlannerAgent 规划异常，回退到规则解析", exc_info=True)
                context.task_ir = parse_task_ir_from_message(context.user_message)
        else:
            context.task_ir = parse_task_ir_from_message(context.user_message)

        if self._tool_capability and context.task_ir:
            for subtask in context.task_ir.subtasks:
                if not subtask.tool_hint:
                    subtask.tool_hint = self._tool_capability.infer_tool_hint(subtask.description)

        if self._memory and context.user_message:
            with contextlib.suppress(Exception):
                await self._memory.store(
                    content=f"TaskIR: {context.task_ir.goal}",
                    memory_type=self._get_memory_type("working"),
                    importance=0.8,
                    task_id="current",
                    tags=["task_ir"],
                )

    def _get_memory_type(self, type_name: str):
        try:
            from long.memory.base import MemoryType
            return getattr(MemoryType, type_name.upper(), MemoryType.SHORT_TERM)
        except ImportError:
            return None

    async def _inject_memory_context(self, context: CognitiveContext) -> None:
        if not self._memory or not context.user_message:
            return

        try:
            mem_items = await self._memory.search(
                context.user_message, limit=5, strategy="hybrid"
            )
            if not mem_items:
                return

            mem_lines = []
            for item in mem_items[:5]:
                content = getattr(item, "content", str(item))[:150].strip()
                if content:
                    mem_lines.append(f"- {content}")

            if mem_lines:
                mem_text = "## 相关记忆\n以下是历史积累的知识，请参考：\n" + "\n".join(mem_lines)
                for i, msg in enumerate(context.messages):
                    if msg.get("role") == "system":
                        context.messages[i] = {
                            "role": "system",
                            "content": msg.get("content", "") + "\n\n" + mem_text,
                        }
                        break
        except Exception:
            pass

    async def _store_tool_result_to_memory(self, context: CognitiveContext, tool_name: str, result: str, success: bool) -> None:
        if not self._memory:
            return

        try:
            from long.memory.base import MemoryType

            if tool_name == "tavily_search" and success and result:
                await self._memory.store(
                    content=result[:500],
                    memory_type=MemoryType.SEMANTIC,
                    importance=0.6,
                    tags=["search_result", context.user_message[:30]],
                )

            if not success:
                await self._memory.store(
                    content=f"工具 {tool_name} 失败: {result[:200]}",
                    memory_type=MemoryType.EPISODIC,
                    importance=0.7,
                    tags=["error", tool_name],
                )

            if success and context.round_count == 1:
                await self._memory.store(
                    content=f"任务 '{context.user_message[:50]}' 首轮使用 {tool_name} 成功",
                    memory_type=MemoryType.PROCEDURAL,
                    importance=0.5,
                    tags=["success_pattern", tool_name],
                )
        except Exception:
            pass

    def _update_task_ir_from_result(self, context: CognitiveContext, tool_name: str, result: str, success: bool) -> None:
        if not context.task_ir:
            return

        if tool_name == "tavily_search" and success and result:
            key_sentences = self.compressor._protector.extract_key_sentences(result, max_sentences=3)
            for s in key_sentences:
                context.task_ir.add_key_fact(s)

        for subtask in context.task_ir.subtasks:
            if subtask.status == "in_progress":
                if success:
                    context.task_ir.complete_subtask(subtask.id, result_summary=result[:100] if result else None)
                else:
                    context.task_ir.fail_subtask(subtask.id)
                break

    # ── 节点处理器 ──

    async def _handle_think(self, ctx: dict[str, Any]) -> dict[str, Any]:
        context: CognitiveContext = ctx.get("_cognitive_context")
        if context is None:
            return {"_error": "缺少认知上下文"}

        context.round_count += 1
        if context.round_count > context.max_rounds:
            exhausted_msg = (
                f"任务轮次已耗尽（{context.max_rounds}轮）。"
                f"已执行工具: {[t['name'] for t in context.tool_history]}"
            )
            return {
                "is_complete": True,
                "has_final_text": True,
                "has_tool_calls": False,
                "_retry_think": False,
                "_final_text": exhausted_msg,
                "final_output": exhausted_msg,
            }

        if context.round_count == 1:
            await self._ensure_task_ir(context)
            await self._inject_memory_context(context)

        # 第一轮：在用户消息末尾追加通用工具调用提示
        # 这不是针对特定关键词的硬编码，而是对所有带工具的请求都生效：
        # 让 LLM 在收到消息时就知道应该优先调用工具，而不是回复文字。
        if context.round_count == 1 and not context.tool_history:
            for i in range(len(context.messages) - 1, -1, -1):
                if context.messages[i].get("role") == "user":
                    context.messages[i]["content"] += (
                        "\n\n[工具调用提示] 你有工具可用。如果这个请求可以通过调用工具来回答，"
                        "请直接调用工具，不要先回复文字。"
                    )
                    break

        task_context_msg = self._build_task_context_message(context)
        if task_context_msg:
            has_system = any(m.get("role") == "system" for m in context.messages)
            if has_system:
                for i, msg in enumerate(context.messages):
                    if msg.get("role") == "system":
                        existing = msg.get("content", "")
                        if "任务执行状态" not in existing:
                            context.messages[i] = {
                                "role": "system",
                                "content": existing + "\n\n" + task_context_msg["content"],
                            }
                        break
            else:
                context.messages.insert(0, task_context_msg)

        if context.search_count >= context.max_search_count:
            search_hint = self._build_search_exhaustion_hint(context)
            if search_hint:
                last_msg = context.messages[-1] if context.messages else None
                if last_msg and last_msg.get("role") == "user":
                    context.messages[-1]["content"] += search_hint
                else:
                    context.messages.append({"role": "user", "content": search_hint})

        # 检测重复工具调用：连续多次调用同一工具，强制生成回复
        if context.consecutive_duplicate_calls >= 2:
            force_hint = (
                "\n\n## ⚠️ 重复工具调用检测\n"
                "你已经连续多次调用同一个工具并获得了相同的结果。"
                "请立即基于已有数据生成最终回答，不要再调用任何工具。"
                "如果确实需要再次查询，请先说明原因。"
            )
            last_msg = context.messages[-1] if context.messages else None
            if last_msg and last_msg.get("role") == "user":
                context.messages[-1]["content"] += force_hint
            else:
                context.messages.append({"role": "user", "content": force_hint})

        try:
            tools_list = ctx.get("_tools", [])
            if context.search_count >= context.max_search_count:
                tools_list = [
                    t for t in tools_list
                    if t.get("function", {}).get("name") != "tavily_search"
                ]

            context.llm_call_count += 1
            response = await asyncio.wait_for(
                self._llm_chat_with_tools(
                    context.messages,
                    tools_list,
                    purpose="chat",
                ),
                timeout=120.0,
            )
        except TimeoutError:
            context.llm_timeout_count += 1
            return {"_error": "LLM 超时"}
        except Exception as e:
            context.llm_timeout_count += 1
            return {"_error": f"LLM 调用失败: {e}"}

        logger.debug(
            "THINK round %d: messages=%d, tool_calls=%s, content=%s",
            context.round_count, len(context.messages),
            bool(response.tool_calls), bool(response.content),
        )

        if response.tool_calls:
            context.consecutive_empty_responses = 0
            if context.task_ir:
                for tc in response.tool_calls:
                    for subtask in context.task_ir.subtasks:
                        if subtask.status == "pending" and not context.task_ir.in_progress_subtasks():
                            hint = subtask.tool_hint
                            if hint is None or tc["name"].startswith(hint.split("_")[0]) or hint in tc["name"]:
                                context.task_ir.mark_subtask_in_progress(subtask.id)
                                break
                            break

            ctx["_pending_tool_calls"] = response.tool_calls
            ctx["_llm_response"] = response
            return {"has_tool_calls": True, "has_final_text": False, "_retry_think": False}

        if response.content:
            context.consecutive_empty_responses = 0
            extracted_code = self._extract_code_from_response(response.content)
            if extracted_code and len(extracted_code) > 50:
                needs_code_exec = any(
                    kw in context.user_message
                    for kw in ("图", "chart", "plot", "折线", "可视化", "报告", "report", "生成", "保存", "计算", "分析", "word", "docx", "文档")
                )
                has_code_tools = any(
                    t["name"] in ("write_file", "execute_code", "execute_file")
                    for t in context.tool_history
                    if "error" not in t
                )
                if needs_code_exec and not has_code_tools:
                    try:
                        exec_result = await asyncio.wait_for(
                            self._tool_execute("execute_code", {"code": extracted_code, "language": "python"}),
                            timeout=120.0,
                        )
                        context.tool_history.append({
                            "name": "execute_code",
                            "arguments": {"language": "python"},
                            "result": exec_result,
                        })
                        exec_success = "失败" not in exec_result and "Error" not in exec_result and "Traceback" not in exec_result
                        if exec_success:
                            context.messages.append({"role": "assistant", "content": response.content})
                            context.messages.append({"role": "tool", "content": exec_result, "tool_call_id": "auto_exec"})
                            return {"has_tool_calls": False, "has_final_text": False, "_retry_think": True}
                    except Exception as e:
                        logger.debug("自动执行代码块失败: %s", e)

            needs_chart = any(
                kw in context.user_message
                for kw in ("图", "chart", "plot", "折线", "可视化")
            )
            needs_report = any(
                kw in context.user_message
                for kw in ("报告", "report", "文档", "生成", "保存", "导出")
            )
            has_code_tools = any(
                t["name"] in ("write_file", "execute_code", "execute_file")
                for t in context.tool_history
                if "error" not in t
            )
            has_search = any(
                t["name"] == "tavily_search"
                for t in context.tool_history
                if "error" not in t and not t.get("result", "").startswith("[")
            )

            if (needs_chart or needs_report) and has_search and not has_code_tools and context.round_count < context.max_rounds:
                continuation_hint = (
                    "\n\n[系统提示] 你还没有完成用户的要求。用户要求生成"
                    + ("折线图" if needs_chart else "")
                    + ("和" if needs_chart and needs_report else "")
                    + ("报告" if needs_report else "")
                    + "，请使用 write_file 和 execute_file 工具来完成。不要只返回文本。"
                )
                context.messages.append({"role": "assistant", "content": response.content})
                context.messages.append({"role": "user", "content": continuation_hint})
                return {"has_tool_calls": False, "has_final_text": False, "_retry_think": True}

            search_results_sufficient = self._evaluate_search_sufficiency(context)
            content_gives_up = any(
                kw in response.content
                for kw in ("无法提供", "未能获取", "建议你", "无法获取", "未能直接获取", "没有找到", "找不到")
            )

            if content_gives_up and not search_results_sufficient and context.round_count < context.max_rounds:
                # 通用引导：提醒 LLM 使用可用工具而非放弃
                has_any_tool_result = len(context.tool_history) > 0
                if not has_any_tool_result:
                    retry_hint = (
                        "\n\n[系统提示] 你的回答表示无法获取用户需要的信息，但你还没有尝试调用任何工具。"
                        "请立即调用以下工具之一获取数据：\n"
                        "- execute_file: 执行已有脚本（如 skills/ 目录下的脚本），参数 path 和 args\n"
                        "- tavily_search: 搜索网络信息，参数 query\n"
                        "- read_skill_md: 读取 skill 文档，了解如何使用某个 skill\n"
                        "不要直接放弃。必须调用工具！"
                    )
                else:
                    retry_hint = (
                        "\n\n[系统提示] 你的回答表示无法获取用户需要的信息，但搜索结果可能不够充分。"
                        "请尝试用不同的关键词重新搜索，换一个搜索策略。不要放弃。"
                    )
                context.messages.append({"role": "assistant", "content": response.content})
                context.messages.append({"role": "user", "content": retry_hint})
                if has_any_tool_result and context.search_count < context.max_search_count:
                    context.max_search_count = context.search_count + 1
                return {"has_tool_calls": False, "has_final_text": False, "_retry_think": True}

            _CODE_TASK_KEYWORDS = (
                "排序", "算法", "写代码", "实现", "编程", "函数", "程序",
                "生成图表", "画图", "数据分析", "可视化", "折线图", "柱状图",
                "快速排序", "归并排序", "冒泡排序", "桶排序", "树排序",
                "二叉树", "链表", "哈希表", "栈", "队列",
            )
            # 这类任务必须调用工具获取数据，不能纯文本回答
            _TOOL_REQUIRED_KEYWORDS = (
                "天气", "气温", "温度", "下雨", "下雪", "weather",
                "汇率", "股价", "股票", "基金", "比特币",
                "新闻", "热点", "赛事", "比分",
            )
            _FABRICATED_PATTERNS = (
                "测试结果", "运行结果", "执行结果", "排序结果", "输出结果",
                "程序输出", "运行输出", "测试通过", "测试成功",
                "Output:", "Result:", "Test passed",
            )
            needs_code = any(kw in context.user_message for kw in _CODE_TASK_KEYWORDS)
            needs_tool_required = any(kw in context.user_message for kw in _TOOL_REQUIRED_KEYWORDS)
            has_tool_history = len(context.tool_history) > 0
            has_code_tools = any(
                t["name"] in ("write_file", "execute_code", "execute_file")
                for t in context.tool_history
                if "error" not in t
            )

            has_code_exec = any(
                t["name"] in ("execute_code", "execute_file")
                for t in context.tool_history
                if "error" not in t
            )

            logger.info(
                "THINK 防幻觉检查: needs_code=%s, needs_tool_required=%s, has_tool_history=%s, has_code_tools=%s, has_code_exec=%s, round=%d/%d, tool_history_names=%s",
                needs_code, needs_tool_required, has_tool_history, has_code_tools, has_code_exec,
                context.round_count, context.max_rounds,
                [t["name"] for t in context.tool_history],
            )

            can_retry = context.round_count < context.max_rounds

            # 需要工具获取数据的任务，LLM 没有调用任何工具就放弃了
            if needs_tool_required and not has_tool_history:
                if can_retry:
                    logger.info("THINK: 检测到需要工具的任务但 LLM 未调用任何工具，强制重试")

                    # 根据用户消息内容生成更具体的工具调用指引
                    _weather_keywords = ("天气", "气温", "温度", "下雨", "下雪", "weather", "预报")
                    _finance_keywords = ("汇率", "股价", "股票", "基金", "比特币")
                    _news_keywords = ("新闻", "热点", "赛事", "比分")

                    specific_hint = ""
                    if any(kw in context.user_message for kw in _weather_keywords):
                        # 从系统提示中提取城市名
                        import re as _re
                        city_patterns = _re.findall(
                            r'([\u4e00-\u9fff]{2,4})(?:和|与|及|、|,|的)?(?:天气|气温|温度|下雨|下雪)',
                            context.user_message,
                        )
                        # 也匹配 "天气怎么样" 前面的城市
                        if not city_patterns:
                            city_patterns = _re.findall(
                                r'([\u4e00-\u9fff]{2,4})(?:和|与|及|、|,)',
                                context.user_message,
                            )
                        # 处理 "首都" 等代称
                        city_aliases = {"首都": "北京", "魔都": "上海", "山城": "重庆", "羊城": "广州", "蓉城": "成都"}
                        resolved_cities = []
                        for c in city_patterns:
                            resolved_cities.append(city_aliases.get(c, c))
                        # 如果没提取到城市，尝试从整个消息中提取
                        if not resolved_cities:
                            all_cities = _re.findall(r'[\u4e00-\u9fff]{2,4}', context.user_message)
                            # 过滤常见非城市词
                            _non_city = {"怎么样", "天气", "气温", "温度", "请问", "查询", "现在", "今天", "明天", "首都", "和风"}
                            resolved_cities = [city_aliases.get(c, c) for c in all_cities if c not in _non_city and len(c) >= 2]

                        cities_arg = ",".join(resolved_cities) if resolved_cities else "杭州"
                        specific_hint = (
                            f"\n\n**天气查询的具体调用方式：**\n"
                            f"```\n"
                            f"query_weather(city='{cities_arg}')\n"
                            f"```\n"
                            f"多城市用英文逗号分隔城市名。请立即调用此工具！\n"
                        )
                    elif any(kw in context.user_message for kw in _finance_keywords):
                        specific_hint = "\n请使用 tavily_search 搜索实时金融数据。\n"
                    elif any(kw in context.user_message for kw in _news_keywords):
                        specific_hint = "\n请使用 tavily_search 搜索最新新闻。\n"

                    context.messages.append({"role": "assistant", "content": response.content})
                    context.messages.append({
                        "role": "user",
                        "content": (
                            "这个任务需要调用工具获取实时数据，你不能直接用文本回答。"
                            "请立即调用以下工具之一获取数据：\n"
                            "- query_weather: 查询天气数据，参数 city（城市名，多城市用逗号分隔）\n"
                            "- execute_file: 执行已有脚本（如 skills/ 目录下的脚本），参数 path 和 args\n"
                            "- tavily_search: 搜索网络信息，参数 query（禁止用于天气查询）\n"
                            "- read_skill_md: 读取 skill 文档，了解如何使用某个 skill\n"
                            "不要放弃，不要编造数据。必须调用工具！"
                            + specific_hint
                        ),
                    })
                    return {"has_tool_calls": False, "has_final_text": False, "_retry_think": True}

            if needs_code and not has_code_tools:
                if can_retry:
                    if not has_tool_history:
                        logger.info("THINK: 检测到需要代码的任务但 LLM 未调用工具，强制重试")
                    else:
                        logger.info("THINK: 检测到需要代码的任务但 LLM 未调用代码工具，强制重试")
                    context.messages.append({"role": "assistant", "content": response.content})
                    context.messages.append({
                        "role": "user",
                        "content": (
                            "这个任务需要写代码并执行，"
                            "使用 write_file 写入代码文件，然后用 execute_file 执行该文件（传入 path 参数，如 path='output/xxx.py'）。"
                            "不要只在文本中描述结果，必须实际执行代码。"
                            "不要调用 list_files 等无关工具。"
                        ),
                    })
                    return {"has_tool_calls": False, "has_final_text": False, "_retry_think": True}
                else:
                    logger.warning("THINK: 需要代码的任务但轮次已耗尽，走降级输出路径")
                    degraded = await self._generate_degraded_code_output(context, response.content)
                    if degraded:
                        return {
                            "has_final_text": True,
                            "has_tool_calls": False,
                            "_retry_think": False,
                            "_final_text": degraded,
                            "final_output": degraded,
                            "is_complete": True,
                        }

            if needs_code and has_code_tools and not has_code_exec:
                if can_retry:
                    logger.info("THINK: 检测到需要代码的任务：代码已写入但未执行，强制重试")
                    context.messages.append({"role": "assistant", "content": response.content})
                    context.messages.append({
                        "role": "user",
                        "content": (
                            "你已经用 write_file 写入了代码文件，但还没有执行它。"
                            "请立即调用 execute_file 工具执行该文件（传入 path 参数）。"
                            "不要在文本中编造运行结果。"
                        ),
                    })
                    return {"has_tool_calls": False, "has_final_text": False, "_retry_think": True}
                else:
                    logger.warning("THINK: 代码已写入但未执行且轮次已耗尽，走降级输出路径")
                    degraded = await self._generate_degraded_code_output(context, response.content)
                    if degraded:
                        return {
                            "has_final_text": True,
                            "has_tool_calls": False,
                            "_retry_think": False,
                            "_final_text": degraded,
                            "final_output": degraded,
                            "is_complete": True,
                        }

            if not has_code_exec and response.content:
                has_fabricated = any(p in response.content for p in _FABRICATED_PATTERNS)
                has_code_block = "```" in response.content and (
                    "python" in response.content.lower()
                    or "def " in response.content
                    or "import " in response.content
                )
                if has_fabricated and has_code_block:
                    if can_retry:
                        logger.info("THINK: 检测到 LLM 编造了不存在的执行结果，强制重试")
                        context.messages.append({"role": "assistant", "content": response.content})
                        context.messages.append({
                            "role": "user",
                            "content": (
                                "⚠️ 你刚才的回复包含了编造的测试/执行结果。"
                                "你并没有调用 execute_code/execute_file 工具，所以不可能有真实的执行结果。"
                                "请使用 write_file 写入代码文件，然后用 execute_file 执行该文件（传入 path 参数），"
                                "展示真实的工具返回结果。禁止编造执行结果。"
                            ),
                        })
                        return {"has_tool_calls": False, "has_final_text": False, "_retry_think": True}
                    else:
                        logger.warning("THINK: 检测到编造的执行结果且轮次已耗尽，剥离幻觉内容")
                        cleaned = self._strip_fabricated_content(response.content)
                        return {
                            "has_final_text": True,
                            "has_tool_calls": False,
                            "_retry_think": False,
                            "_final_text": cleaned,
                            "final_output": cleaned,
                            "is_complete": True,
                        }

            if context.task_ir:
                context.task_ir.add_conclusion(response.content[:200])

            return {
                "has_final_text": True,
                "has_tool_calls": False,
                "_retry_think": False,
                "_final_text": response.content,
            }

        # LLM 返回空响应（无 tool_calls 也无 content）
        context.consecutive_empty_responses += 1
        logger.warning(
            "THINK: LLM 返回空响应（第%d次），%s",
            context.consecutive_empty_responses,
            "强制生成回复" if context.consecutive_empty_responses >= 2 else "重试",
        )

        if context.consecutive_empty_responses >= 2:
            # 连续 2 次空响应，使用已有工具结果生成回复
            fallback = self._build_fallback_output(context)
            return {
                "has_final_text": True,
                "has_tool_calls": False,
                "_retry_think": False,
                "_final_text": fallback,
                "final_output": fallback,
                "is_complete": True,
            }

        context.messages.append({"role": "assistant", "content": ""})
        context.messages.append({
            "role": "user",
            "content": (
                "你刚才没有返回任何内容。请重新思考并回复。"
                "如果需要查询信息，请调用相应的工具。"
                "如果可以直接回答，请给出你的回答。"
            ),
        })
        return {"has_tool_calls": False, "has_final_text": False, "_retry_think": True}

    def _build_fallback_output(self, context: CognitiveContext) -> str:
        """当 LLM 连续返回空响应时，用已有工具结果生成降级回复"""
        lines = ["以下是根据已获取的信息生成的回复：\n"]
        for t in context.tool_history:
            if "error" not in t and t.get("result"):
                name = t.get("name", "unknown")
                result = t.get("result", "")
                lines.append(f"【{name}】\n{result}\n")
        lines.append("\n---\n由于 LLM 持续返回空响应，以上为基于工具结果自动生成的摘要。")
        return "\n".join(lines)

    def _build_search_exhaustion_hint(self, context: CognitiveContext) -> str | None:
        has_code_tools = any(
            t["name"] in ("write_file", "execute_code", "execute_file")
            for t in context.tool_history
            if "error" not in t
        )
        needs_chart = any(
            kw in context.user_message
            for kw in ("图", "chart", "plot", "折线", "可视化")
        )
        needs_report = any(
            kw in context.user_message
            for kw in ("报告", "report", "文档", "生成", "保存", "导出")
        )
        search_results_sufficient = self._evaluate_search_sufficiency(context)

        if context.task_ir:
            pending = context.task_ir.pending_subtasks()
            pending_search = [s for s in pending if s.tool_hint == "tavily_search"]
            for s in pending_search:
                s.status = "completed"
                if s.id not in context.task_ir.completed_subtasks:
                    context.task_ir.completed_subtasks.append(s.id)
                s.result_summary = "搜索次数耗尽，使用已有结果"

        if has_code_tools:
            return (
                f"\n\n[系统提示] 搜索次数已达上限({context.max_search_count}次)，"
                f"图表和报告已生成，请基于已有结果生成最终总结。"
            )
        elif needs_chart or needs_report:
            return (
                f"\n\n[系统提示] 搜索次数已达上限({context.max_search_count}次)，"
                f"不要再调用 tavily_search。请使用 write_file 和 execute_file 工具"
                f"基于已有搜索结果生成用户要求的图表和报告。"
            )
        elif not search_results_sufficient and context.search_count < context.max_search_count + 2:
            context.max_search_count += 1
            return (
                "\n\n[系统提示] 之前的搜索结果不够充分，没有直接回答用户的问题。"
                "请用更精确的关键词重新搜索，尝试不同的搜索策略。"
            )
        else:
            return (
                f"\n\n[系统提示] 搜索次数已达上限({context.max_search_count}次)，"
                f"请基于已有信息生成最终回答，不要再调用 tavily_search。"
            )

    async def _handle_act(self, ctx: dict[str, Any]) -> dict[str, Any]:
        context: CognitiveContext | None = ctx.get("_cognitive_context")
        tool_calls = ctx.get("_pending_tool_calls", [])

        if not tool_calls:
            return {"has_tool_calls": False}

        search_calls = []
        other_calls = []
        for tc in tool_calls:
            if tc["name"] == "tavily_search":
                search_calls.append(tc)
            else:
                other_calls.append(tc)

        executed = []

        if search_calls and context:
            allowed_searches = []
            for tc in search_calls:
                if context.search_count + len(allowed_searches) >= context.max_search_count:
                    context.tool_history.append({
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                        "result": f"[搜索限制] 已达到最大搜索次数 ({context.max_search_count})",
                        "error": f"[搜索限制] 已达到最大搜索次数 ({context.max_search_count})",
                    })
                    executed.append({
                        "id": tc["id"],
                        "name": tc["name"],
                        "result": f"[搜索限制] 已达到最大搜索次数 ({context.max_search_count})",
                        "intercepted": True,
                    })
                    continue
                allowed_searches.append(tc)

            if allowed_searches:
                search_tasks = [
                    self._execute_single_tool(tc["name"], tc["arguments"], context)
                    for tc in allowed_searches
                ]
                search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

                for tc, sr in zip(allowed_searches, search_results):
                    if isinstance(sr, Exception):
                        executed.append({
                            "id": tc["id"],
                            "name": tc["name"],
                            "result": f"执行失败: {sr}",
                            "error": str(sr),
                        })
                    else:
                        executed.append(sr)
                        if context and sr.get("intercepted") is False:
                            context.tool_history.append({
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                                "result": sr["result"],
                            })
                            context.search_count += 1

                if context:
                    context.last_action_was_search = len(allowed_searches) > 0

        for tc in other_calls:
            # 检测重复工具调用
            tool_key = self._make_tool_call_key(tc["name"], tc["arguments"])
            cached = self._tool_call_cache.get(tool_key)
            is_duplicate = cached is not None

            if is_duplicate:
                # 命中缓存，直接返回缓存结果
                logger.info("拦截重复工具调用: %s (key=%s)", tc["name"], tool_key)
                cached_result, _ts = cached
                context.tool_history.append({
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                    "result": cached_result,
                })
                executed.append({
                    "id": tc["id"],
                    "name": tc["name"],
                    "result": cached_result,
                    "intercepted": True,
                    "cached": True,
                })
                # 更新连续重复计数
                if tool_key == context.last_tool_call_key:
                    context.consecutive_duplicate_calls += 1
                else:
                    context.consecutive_duplicate_calls = 0
                context.last_tool_call_key = tool_key
                continue

            # 首次调用，正常执行
            context.consecutive_duplicate_calls = 0
            context.last_tool_call_key = ""
            sr = await self._execute_single_tool(tc["name"], tc["arguments"], context)
            if isinstance(sr, Exception):
                executed.append({
                    "id": tc["id"],
                    "name": tc["name"],
                    "result": f"执行失败: {sr}",
                    "error": str(sr),
                })
            else:
                executed.append(sr)
                if context and sr.get("intercepted") is False:
                    # 缓存工具结果
                    result_text = sr.get("result", "")
                    self._tool_call_cache[tool_key] = (result_text, time.time())
                    context.tool_history.append({
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                        "result": result_text,
                    })
                    context.last_action_was_search = False

        ctx["_executed_tools"] = executed
        all_intercepted = len(executed) > 0 and all(e.get("intercepted", False) for e in executed)
        if all_intercepted and context:
            context.needs_retry = False
        return {"_all_intercepted": all_intercepted}

    @staticmethod
    def _make_tool_call_key(tool_name: str, arguments: dict[str, Any]) -> str:
        """生成工具调用的缓存 key，包含工具名和参数"""
        args_str = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        return f"{tool_name}:{args_str}"

    async def _execute_single_tool(
        self, tool_name: str, arguments: dict[str, Any], context: CognitiveContext | None
    ) -> dict[str, Any]:
        max_tool_retries = 2
        start_time = time.time()

        # 为工具执行创建 trace span
        trace = current_trace()
        span_ctx = None
        span = None
        if trace is not None:
            span_ctx = trace.span(
                f"tool.{tool_name}",
                attributes={"tool_name": tool_name},
            )
            span = await span_ctx.__aenter__()

        try:
            for attempt in range(max_tool_retries + 1):
                try:
                    result = await asyncio.wait_for(
                        self._tool_execute(tool_name, arguments),
                        timeout=120.0,
                    )
                    break
                except TimeoutError:
                    if attempt < max_tool_retries:
                        logger.debug("工具 %s 超时，重试 %d/%d",
                                       tool_name, attempt + 1, max_tool_retries)
                        continue
                    result = f"工具 {tool_name} 执行超时（已重试 {max_tool_retries} 次）"
                except Exception as e:
                    if attempt < max_tool_retries:
                        logger.debug("工具 %s 执行失败: %s，重试 %d/%d",
                                       tool_name, e, attempt + 1, max_tool_retries)
                        continue
                    result = f"工具 {tool_name} 执行失败: {e}"

            latency = time.time() - start_time
            success = not result.startswith("工具") or "超时" not in result

            if span is not None:
                span.set_attribute("latency_ms", latency * 1000)
                span.set_attribute("success", success)
                if not success:
                    span.finish(SpanStatus.ERROR)

            result = self.compressor.compress(tool_name, result)
            self.tool_router.record_execution(tool_name, arguments)

            if self._tool_capability:
                self._tool_capability.record_call(tool_name, success, latency)

            if context:
                await self._store_tool_result_to_memory(context, tool_name, result, success)
                self._update_task_ir_from_result(context, tool_name, result, success)

            return {
                "id": "",
                "name": tool_name,
                "result": result,
                "intercepted": False,
            }
        except Exception as e:
            if span is not None:
                span.set_attribute("error", str(e))
                span.finish(SpanStatus.ERROR)
            return {
                "id": "",
                "name": tool_name,
                "result": f"执行失败: {e}",
                "error": str(e),
                "intercepted": False,
            }
        finally:
            if span_ctx is not None and span is not None:
                await span_ctx.__aexit__(None, None, None)

    async def _handle_observe(self, ctx: dict[str, Any]) -> dict[str, Any]:
        context: CognitiveContext | None = ctx.get("_cognitive_context")
        executed = ctx.get("_executed_tools", [])
        llm_response = ctx.get("_llm_response")

        if not executed or not context:
            return {}

        all_tool_calls = []
        for tc in llm_response.tool_calls if llm_response else []:
            all_tool_calls.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"]),
                },
            })

        if all_tool_calls:
            context.messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": all_tool_calls,
            })

        # 构建 tool_call_id 映射: 从原始 LLM 响应中获取 tool_call_id
        tool_id_map: dict[str, str] = {}
        if llm_response and llm_response.tool_calls:
            for tc in llm_response.tool_calls:
                tool_id_map[tc["name"]] = tc["id"]

        for exec_result in executed:
            tool_name = exec_result.get("name", "")
            raw_result = exec_result["result"]
            # 使用原始 LLM 响应的 tool_call_id，避免空字符串
            tool_call_id = tool_id_map.get(tool_name, exec_result.get("id", ""))
            # 在工具结果前注入归因标记，让 LLM 明确知道这是自己调用的工具返回的结果
            # 使用指令式标记而非描述式，更有效对抗 LLM 的"您提供"倾向
            if tool_name == "tavily_search":
                attributed_result = f'[注意：这是我（AI）自己调用 tavily_search 搜索得到的结果，不是用户提供的。回复时说"根据搜索结果"而非"根据您提供的"。]\n{raw_result}'
            elif tool_name in ("execute_code", "execute_file"):
                attributed_result = f"[注意：这是我（AI）自己调用 {tool_name} 执行的结果，不是用户提供的。]\n{raw_result}"
            elif tool_name == "write_file":
                attributed_result = f"[注意：这是我（AI）自己调用 write_file 写入的结果。]\n{raw_result}"
            else:
                attributed_result = f"[注意：这是我（AI）自己调用 {tool_name} 的结果。]\n{raw_result}"
            context.messages.append({
                "role": "tool",
                "content": attributed_result,
                "tool_call_id": tool_call_id,
            })

        return {}

    async def _handle_reflect(self, ctx: dict[str, Any]) -> dict[str, Any]:
        context: CognitiveContext | None = ctx.get("_cognitive_context")
        if not context:
            return {"needs_retry": False, "should_continue": False}

        # 检测卡住条件：多轮不调工具 / 重复调用 / 轮次即将耗尽
        has_tool_history = bool(context.tool_history)
        stuck_no_tool = context.round_count >= 3 and not has_tool_history
        stuck_duplicate = context.consecutive_duplicate_calls >= 3
        stuck_near_limit = context.round_count >= context.max_rounds - 1

        if (stuck_no_tool or stuck_duplicate or stuck_near_limit) and not has_tool_history:
            logger.info(
                "REFLECT 检测到卡住: round=%d, tool_history=%d, duplicate=%d",
                context.round_count, len(context.tool_history),
                context.consecutive_duplicate_calls,
            )
            context.needs_retry = False
            return {"needs_retry": False, "should_continue": False, "trigger_planir": True}

        if not has_tool_history:
            return {"needs_retry": False, "should_continue": True}

        reflection = await self.reflector.reflect(context)
        context.reflections.append(reflection.get("reflection", ""))

        if reflection.get("needs_retry") and context.retry_count < context.max_retries:
            context.retry_count += 1
            context.needs_retry = True
            return {"needs_retry": True, "should_continue": True}

        context.needs_retry = False
        return {"needs_retry": False, "should_continue": True}

    async def _handle_plan(self, ctx: dict[str, Any]) -> dict[str, Any]:
        context: CognitiveContext | None = ctx.get("_cognitive_context")
        if not context:
            return {"should_continue": True}

        # 卡住恢复：生成 PlanIR 引导 LLM 走出僵局
        if ctx.get("trigger_planir") and self._plan_executor is not None:
            logger.info("PLAN 卡住恢复: 生成 PlanIR")
            try:
                plan = await self._plan_executor.generate_plan(
                    user_message=context.user_message,
                    history_msgs=context.messages,
                    available_tools=ctx.get("_tools", []),
                )
                if plan and len(plan.steps) > 1:
                    logger.info("PLANIR 生成成功: %s, %d 步", plan.plan_id, len(plan.steps))
                    context.final_output = (
                        "[系统提示] 检测到执行卡住，已生成结构化计划 (%d 步)。"
                        "请按照以下步骤执行：\n%s" % (len(plan.steps), plan.goal)
                    )
                    context.is_complete = True
                    return {"is_complete": True, "should_continue": False}
            except Exception as e:
                logger.warning("PLANIR 生成失败: %s", e)

        plan_result = self.planner.plan(context)
        context.plan_result = plan_result

        if plan_result.is_complete:
            return {"is_complete": True, "should_continue": False}

        if plan_result.next_subtask and context.task_ir:
            context.task_ir.mark_subtask_in_progress(plan_result.next_subtask.id)

        return {"should_continue": True, "is_complete": False}

    async def _handle_output(self, ctx: dict[str, Any]) -> dict[str, Any]:
        context: CognitiveContext | None = ctx.get("_cognitive_context")
        final_text = ctx.get("_final_text", "")

        if context and final_text:
            _CODE_TASK_KEYWORDS = (
                "排序", "算法", "写代码", "实现", "编程", "函数", "程序",
                "生成图表", "画图", "数据分析", "可视化", "折线图", "柱状图",
                "快速排序", "归并排序", "冒泡排序", "桶排序", "树排序",
                "二叉树", "链表", "哈希表", "栈", "队列",
            )
            # 这类任务必须调用工具获取数据，不能纯文本回答
            _TOOL_REQUIRED_KEYWORDS = (
                "天气", "气温", "温度", "下雨", "下雪", "weather",
                "汇率", "股价", "股票", "基金", "比特币",
                "新闻", "热点", "赛事", "比分",
            )
            _FABRICATED_PATTERNS = (
                "测试结果", "运行结果", "执行结果", "排序结果", "输出结果",
                "程序输出", "运行输出", "测试通过", "测试成功",
                "Output:", "Result:", "Test passed",
            )

            needs_code = any(kw in context.user_message for kw in _CODE_TASK_KEYWORDS)
            needs_tool_required = any(kw in context.user_message for kw in _TOOL_REQUIRED_KEYWORDS)
            has_code_tools = any(
                t["name"] in ("write_file", "execute_code", "execute_file")
                for t in context.tool_history
                if "error" not in t
            )
            has_code_exec = any(
                t["name"] in ("execute_code", "execute_file")
                for t in context.tool_history
                if "error" not in t
            )
            has_tool_history = len(context.tool_history) > 0

            logger.debug(
                "OUTPUT 防幻觉检查: needs_code=%s, needs_tool_required=%s, has_code_tools=%s, has_code_exec=%s, round=%d/%d, tool_history=%s",
                needs_code, needs_tool_required, has_code_tools, has_code_exec,
                context.round_count, context.max_rounds,
                [t["name"] for t in context.tool_history],
            )

            can_retry = context.round_count < context.max_rounds

            # 需要工具获取数据的任务，LLM 没有调用任何工具就输出了
            if needs_tool_required and not has_tool_history:
                if can_retry:
                    logger.info("OUTPUT: 检测到需要工具的任务但 LLM 未调用任何工具，重定向到 THINK")
                    context.messages.append({"role": "assistant", "content": final_text})
                    context.messages.append({
                        "role": "user",
                        "content": (
                            "这个任务需要调用工具获取实时数据，你不能直接用文本回答。"
                            "请立即调用以下工具之一获取数据：\n"
                            "- execute_file: 执行已有脚本（如 skills/ 目录下的脚本），参数 path 和 args\n"
                            "- tavily_search: 搜索网络信息，参数 query\n"
                            "- read_skill_md: 读取 skill 文档，了解如何使用某个 skill\n"
                            "不要放弃，不要编造数据。必须调用工具！"
                        ),
                    })
                    return {"_retry_think": True}

            if needs_code and not has_code_exec:
                if can_retry:
                    if not has_code_tools:
                        logger.info("OUTPUT: 需要代码的任务但未调用代码工具，重定向到 THINK")
                        context.messages.append({"role": "assistant", "content": final_text})
                        context.messages.append({
                            "role": "user",
                            "content": (
                                "这个任务需要写代码并执行，"
                                "使用 write_file 写入代码文件，然后用 execute_file 执行该文件（传入 path 参数，如 path='output/xxx.py'）。"
                                "不要只在文本中描述结果，必须实际执行代码。"
                            ),
                        })
                        return {"has_tool_calls": False, "has_final_text": False, "_retry_think": True}
                    else:
                        logger.info("OUTPUT: 代码已写入但未执行，重定向到 THINK")
                        context.messages.append({"role": "assistant", "content": final_text})
                        context.messages.append({
                            "role": "user",
                            "content": (
                                "你已经用 write_file 写入了代码文件，但还没有执行它。"
                                "请立即调用 execute_file 工具执行该文件（传入 path 参数）。"
                                "不要在文本中编造运行结果。"
                            ),
                        })
                        return {"has_tool_calls": False, "has_final_text": False, "_retry_think": True}
                else:
                    logger.warning("OUTPUT: 需要代码的任务但轮次已耗尽，生成降级输出")
                    degraded = await self._generate_degraded_code_output(context, final_text)
                    if degraded:
                        context.final_output = degraded
                        if self._output:
                            await self._output(degraded)
                        return {"is_complete": True, "final_output": degraded, "_retry_think": False}

            has_fabricated = any(p in final_text for p in _FABRICATED_PATTERNS)
            has_code_block = "```" in final_text and (
                "python" in final_text.lower()
                or "def " in final_text
                or "import " in final_text
            )
            if has_fabricated and has_code_block and not has_code_exec:
                if can_retry:
                    logger.info("OUTPUT: 检测到编造的执行结果，重定向到 THINK")
                    context.messages.append({"role": "assistant", "content": final_text})
                    context.messages.append({
                        "role": "user",
                        "content": (
                            "⚠️ 你刚才的回复包含了编造的测试/执行结果。"
                            "你并没有调用 execute_code/execute_file 工具，所以不可能有真实的执行结果。"
                            "请使用 write_file 写入代码文件，然后用 execute_file 执行该文件（传入 path 参数），"
                            "展示真实的工具返回结果。禁止编造执行结果。"
                        ),
                    })
                    return {"has_tool_calls": False, "has_final_text": False, "_retry_think": True}
                else:
                    logger.warning("OUTPUT: 检测到编造的执行结果且轮次已耗尽，剥离幻觉内容")
                    cleaned = self._strip_fabricated_content(final_text)
                    context.final_output = cleaned
                    if self._output:
                        await self._output(cleaned)
                    return {"is_complete": True, "final_output": cleaned, "_retry_think": False}

        if final_text:
            context.final_output = final_text if context else final_text
        elif context and context.tool_history:
            llm_timeout_rate = self._compute_llm_timeout_rate(context)
            consecutive_timeouts = sum(
                1 for e in context.errors
                if "LLM" in e and ("超时" in e or "timeout" in e.lower())
            )

            if llm_timeout_rate > 0.5 or consecutive_timeouts >= 2:
                degraded_output = await self._attempt_degraded_output(context)
                if degraded_output:
                    context.final_output = degraded_output
                else:
                    context.final_output = self._generate_code_summary(context)
            else:
                llm_summary = await self._generate_llm_summary(context)
                if llm_summary:
                    context.final_output = llm_summary
                else:
                    context.final_output = self._generate_code_summary(context)
        else:
            # 空文本且无工具历史，尝试重试而非直接放弃
            if context and context.round_count < context.max_rounds:
                logger.info("OUTPUT: 最终文本为空且无工具历史，重定向到 THINK 重试")
                context.messages.append({
                    "role": "user",
                    "content": "请重新尝试完成任务。如果需要查询信息，请调用工具。",
                })
                return {"has_tool_calls": False, "has_final_text": False, "_retry_think": True}
            context.final_output = "任务未能完成。"

        if context:
            context.is_complete = True

            if self._memory:
                with contextlib.suppress(Exception):
                    await self._memory.auto_promote()

                if context.task_ir:
                    try:
                        from long.memory.base import MemoryType
                        await self._memory.store(
                            content=f"完成任务: {context.task_ir.goal} → {context.final_output[:200]}",
                            memory_type=MemoryType.EPISODIC,
                            importance=0.7,
                            tags=["task_completion"],
                        )
                    except Exception:
                        pass

        if self._output:
            await self._output(context.final_output if context else "")

        return {"is_complete": True, "final_output": context.final_output if context else "", "_retry_think": False}

    async def _attempt_degraded_output(self, context: CognitiveContext) -> str | None:
        successful_tools = [
            t for t in context.tool_history
            if "error" not in t and not t.get("result", "").startswith("[")
        ]
        if not successful_tools:
            return None

        search_results = [t for t in successful_tools if t["name"] == "tavily_search"]
        has_code_tools = any(
            t["name"] in ("write_file", "execute_code", "execute_file")
            for t in successful_tools
        )

        needs_chart = any(
            kw in context.user_message
            for kw in ("图", "chart", "plot", "折线", "可视化", "柱状", "饼图", "柱形")
        )
        needs_report = any(
            kw in context.user_message
            for kw in ("报告", "report", "文档", "生成", "保存", "导出", "详细")
        )

        if not search_results and not has_code_tools:
            return None

        code = await self._generate_degraded_code(context, search_results, needs_chart, needs_report)
        if not code:
            return self._generate_code_summary(context)

        parts = []
        if context.task_ir:
            parts.append(f"# {context.task_ir.goal}\n")

        try:
            write_result = await self._tool_execute("write_file", {
                "path": "degraded_output.py",
                "content": code,
            })

            exec_result = await self._tool_execute("execute_code", {
                "code": code,
                "language": "python",
            })

            context.tool_history.append({
                "name": "write_file",
                "arguments": {"path": "degraded_output.py", "content": code[:200]},
                "result": write_result,
            })
            context.tool_history.append({
                "name": "execute_code",
                "arguments": {"language": "python"},
                "result": exec_result,
            })

            if search_results:
                parts.append("## 搜索结果\n")
                for i, t in enumerate(search_results, 1):
                    preview = t["result"][:600].strip()
                    parts.append(f"### 结果 {i}\n{preview}\n")

            exec_success = "失败" not in exec_result and "Error" not in exec_result and "Traceback" not in exec_result
            if exec_success:
                parts.append("## 生成的图表和报告\n")
                parts.append(exec_result[:2000])
                parts.append("\n\n📄 代码已保存至 `degraded_output.py`")
            else:
                parts.append("## 代码执行结果\n")
                parts.append(exec_result[:800])
                parts.append("\n\n📄 代码已保存至 `degraded_output.py`，可手动修复后执行")

        except Exception as e:
            logger.debug("降级输出执行失败: %s", e)
            parts.append(f"## 降级输出\n代码生成失败: {e}\n")
            parts.append(self._generate_code_summary(context))

        if context.task_ir and context.task_ir.key_facts:
            parts.append("\n## 关键事实\n")
            for fact in context.task_ir.key_facts:
                parts.append(f"- {fact}\n")

        if context.task_ir and context.task_ir.intermediate_conclusions:
            parts.append("\n## 结论\n")
            for c in context.task_ir.intermediate_conclusions:
                parts.append(f"- {c}\n")

        return "\n".join(parts)

    async def _generate_degraded_code(
        self,
        context: CognitiveContext,
        search_results: list[dict[str, Any]],
        needs_chart: bool,
        needs_report: bool,
    ) -> str | None:
        search_text = "\n".join(t.get("result", "") for t in search_results)
        if not search_text.strip():
            return None

        code = await self._try_llm_code_generation(context, search_text, needs_chart, needs_report)
        if code:
            return code

        code = await self._try_llm_code_generation(
            context, search_text, needs_chart, needs_report, fast_mode=True,
        )
        return code

    async def _try_llm_code_generation(
        self,
        context: CognitiveContext,
        search_text: str,
        needs_chart: bool,
        needs_report: bool,
        fast_mode: bool = False,
    ) -> str | None:
        truncated = search_text[:3000]
        task_goal = context.task_ir.goal if context.task_ir else context.user_message

        chart_instr = ""
        if needs_chart:
            chart_instr = (
                "\n- 必须使用 matplotlib 生成图表，设置 Agg 后端，支持中文显示"
                "\n- 图表必须保存为 PNG 文件，使用 plt.savefig('output/chart.png', dpi=150)"
                "\n- 禁止输出 Mermaid 语法（如 xychart-beta、pie 等），禁止输出 ASCII 文本图表"
                "\n- 必须生成真实的可视化图片文件"
            )

        report_instr = ""
        if needs_report:
            report_instr = "\n- 生成 Markdown 格式的详细报告并保存为文件"

        prompt = (
            f"任务目标: {task_goal}\n\n"
            f"搜索结果:\n{truncated}\n\n"
            f"请生成一个完整的 Python 脚本来完成以下要求:"
            f"{chart_instr}{report_instr}"
            f"\n- 只输出 Python 代码，不要解释"
            f"\n- 代码必须完整可执行"
            f"\n- 使用 print() 输出关键信息"
        )

        try:
            chat_kwargs: dict[str, Any] = {"purpose": "chat"}
            if fast_mode:
                chat_kwargs["temperature"] = 0.2
                chat_kwargs["max_tokens"] = 2048
            response = await asyncio.wait_for(
                self._llm_chat(
                    [{"role": "user", "content": prompt}],
                    **chat_kwargs,
                ),
                timeout=30.0,
            )
            content = response.content.strip() if response.content else ""
            if not content:
                return None

            code = self._extract_code_from_response(content)
            if code and len(code) > 20:
                return code
            return None
        except Exception as e:
            logger.debug("LLM 降级代码生成失败: %s", e)
            return None

    def _extract_code_from_response(self, content: str) -> str | None:
        code_block_pattern = r"```(?:python|py)?\s*\n(.*?)\n```"
        matches = re.findall(code_block_pattern, content, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()

        if any(line.strip().startswith(("import ", "from ", "def ", "class ", "print(")) for line in content.split("\n")):
            return content.strip()

        return None

    def _strip_fabricated_content(self, text: str) -> str:
        _FABRICATED_PATTERNS = (
            "测试结果", "运行结果", "执行结果", "排序结果", "输出结果",
            "程序输出", "运行输出", "测试通过", "测试成功",
            "Output:", "Result:", "Test passed",
        )
        lines = text.split("\n")
        cleaned_lines = []
        skip_block = False
        for line in lines:
            if any(p in line for p in _FABRICATED_PATTERNS):
                skip_block = True
                continue
            if skip_block and (line.strip().startswith("-") or line.strip().startswith("|") or line.strip().startswith("```") or not line.strip()):
                continue
            skip_block = False
            cleaned_lines.append(line)

        code_blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)\n```", text, re.DOTALL)
        result = "\n".join(cleaned_lines).strip()
        if not result and code_blocks:
            result = "⚠️ 以下代码未经实际执行，结果为LLM编造，已被剥离：\n\n以下是LLM生成的代码（未执行）：\n```python\n" + code_blocks[0][:2000] + "\n```"
        elif code_blocks:
            result += "\n\n⚠️ 注意：上述代码未经实际执行验证。如需运行，请使用 execute_file 工具。"
        elif not result:
            result = "⚠️ LLM的回复包含编造的执行结果，已被系统自动剥离。请重新要求执行代码。"
        return result

    async def _generate_degraded_code_output(self, context: CognitiveContext, original_text: str) -> str | None:
        code = self._extract_code_from_response(original_text)
        if not code or len(code) < 20:
            return self._strip_fabricated_content(original_text)

        try:
            write_result = await asyncio.wait_for(
                self._tool_execute("write_file", {
                    "path": "output/degraded_code.py",
                    "content": code,
                }),
                timeout=30.0,
            )
            context.tool_history.append({
                "name": "write_file",
                "arguments": {"path": "output/degraded_code.py"},
                "result": write_result,
            })

            exec_result = await asyncio.wait_for(
                self._tool_execute("execute_code", {"code": code, "language": "python"}),
                timeout=60.0,
            )
            context.tool_history.append({
                "name": "execute_code",
                "arguments": {"language": "python"},
                "result": exec_result,
            })

            exec_success = "失败" not in exec_result and "Error" not in exec_result and "Traceback" not in exec_result
            if exec_success:
                return f"✅ 代码已自动执行（降级模式）：\n\n{exec_result[:2000]}\n\n📄 代码已保存至 output/degraded_code.py"
            else:
                return (
                    f"⚠️ 代码执行失败：\n\n{exec_result[:800]}\n\n"
                    f"📄 代码已保存至 output/degraded_code.py，可手动修复后执行"
                )
        except Exception as e:
            logger.debug("降级代码执行失败: %s", e)
            return self._strip_fabricated_content(original_text)

    async def _autonomous_execution(self, context: CognitiveContext) -> str | None:
        if context.autonomous_executed:
            return None

        needs_chart = any(
            kw in context.user_message
            for kw in ("图", "chart", "plot", "折线", "可视化", "柱状", "饼图", "柱形")
        )
        needs_report = any(
            kw in context.user_message
            for kw in ("报告", "report", "文档", "生成", "保存", "导出", "详细")
        )

        search_results = [
            t for t in context.tool_history
            if t["name"] == "tavily_search" and "error" not in t
            and not t.get("result", "").startswith("[")
        ]

        if not search_results and context.search_count < context.max_search_count:
            search_query = context.user_message[:50]
            try:
                search_result = await asyncio.wait_for(
                    self._tool_execute("tavily_search", {"query": search_query}),
                    timeout=30.0,
                )
                context.tool_history.append({
                    "name": "tavily_search",
                    "arguments": {"query": search_query},
                    "result": search_result,
                })
                context.search_count += 1
                search_results = [context.tool_history[-1]]

                if self._memory and search_result:
                    with contextlib.suppress(Exception):
                        await self._store_tool_result_to_memory(
                            context, "tavily_search", search_result, True
                        )
            except Exception as e:
                logger.debug("自主搜索失败: %s", e)

        if not search_results:
            return self._generate_code_summary(context) if context.tool_history else None

        search_text = "\n".join(t.get("result", "") for t in search_results)
        if not search_text.strip():
            return None

        code = await self._try_llm_code_generation(context, search_text, needs_chart, needs_report)
        if not code:
            code = await self._try_llm_code_generation(
                context, search_text, needs_chart, needs_report, fast_mode=True,
            )
        if not code:
            return self._generate_code_summary(context)

        parts = []
        if context.task_ir:
            parts.append(f"# {context.task_ir.goal}\n")

        try:
            write_result = await self._tool_execute("write_file", {
                "path": "autonomous_output.py",
                "content": code,
            })

            exec_result = await asyncio.wait_for(
                self._tool_execute("execute_code", {"code": code, "language": "python"}),
                timeout=120.0,
            )

            context.tool_history.append({
                "name": "write_file",
                "arguments": {"path": "autonomous_output.py"},
                "result": write_result,
            })
            context.tool_history.append({
                "name": "execute_code",
                "arguments": {"language": "python"},
                "result": exec_result,
            })

            if search_results:
                parts.append("## 搜索结果\n")
                for i, t in enumerate(search_results, 1):
                    preview = t["result"][:600].strip()
                    parts.append(f"### 结果 {i}\n{preview}\n")

            exec_success = "失败" not in exec_result and "Error" not in exec_result and "Traceback" not in exec_result
            if exec_success:
                parts.append("## 生成的图表和报告\n")
                parts.append(exec_result[:2000])
                parts.append("\n\n📄 代码已保存至 `autonomous_output.py`")
            else:
                parts.append("## 代码执行结果\n")
                parts.append(exec_result[:800])
                parts.append("\n\n📄 代码已保存至 `autonomous_output.py`，可手动修复后执行")

        except Exception as e:
            logger.debug("自主执行失败: %s", e)
            parts.append(f"## 自主执行\n代码生成/执行失败: {e}\n")
            parts.append(self._generate_code_summary(context))

        if context.task_ir and context.task_ir.key_facts:
            parts.append("\n## 关键事实\n")
            for fact in context.task_ir.key_facts:
                parts.append(f"- {fact}\n")

        return "\n".join(parts)

    def _compute_llm_timeout_rate(self, context: CognitiveContext) -> float:
        if context.llm_call_count == 0:
            total_errors = len(context.errors)
            if total_errors == 0:
                return 0.0
            timeout_errors = sum(
                1 for e in context.errors
                if "超时" in e or "timeout" in e.lower()
            )
            return timeout_errors / max(total_errors, 1)
        return context.llm_timeout_count / max(context.llm_call_count, 1)

    def _generate_code_summary(self, context: CognitiveContext) -> str:
        successful_tools = [
            t for t in context.tool_history
            if "error" not in t and not t.get("result", "").startswith("[")
        ]

        if not successful_tools:
            return "任务未能完成：所有工具执行均失败。"

        parts = []

        if context.task_ir:
            parts.append(f"# {context.task_ir.goal}\n")

        search_results = [t for t in successful_tools if t["name"] == "tavily_search"]
        code_results = [t for t in successful_tools if t["name"] in ("execute_code", "execute_file")]
        file_results = [t for t in successful_tools if t["name"] == "write_file"]

        if search_results:
            parts.append("## 搜索结果\n")
            for i, t in enumerate(search_results, 1):
                preview = t["result"][:600].strip()
                parts.append(f"### 结果 {i}\n{preview}\n")

        if code_results:
            parts.append("## 代码执行结果\n")
            for t in code_results:
                preview = t["result"][:400].strip()
                parts.append(f"{preview}\n")

        if file_results:
            parts.append("## 生成的文件\n")
            for t in file_results:
                args = t.get("arguments", {})
                path = args.get("path", "未知路径")
                parts.append(f"- 📄 {path}\n")

        if context.task_ir and context.task_ir.key_facts:
            parts.append("## 关键事实\n")
            for fact in context.task_ir.key_facts:
                parts.append(f"- {fact}\n")

        if context.task_ir and context.task_ir.intermediate_conclusions:
            parts.append("## 结论\n")
            for c in context.task_ir.intermediate_conclusions:
                parts.append(f"- {c}\n")

        return "\n".join(parts)

    async def _generate_llm_summary(self, context: CognitiveContext) -> str:
        tool_summary_parts = []
        for t in context.tool_history:
            if "error" not in t and not t.get("result", "").startswith("["):
                preview = t["result"][:500].strip()
                tool_summary_parts.append(f"[{t['name']}] {preview}")

        if not tool_summary_parts:
            return ""

        tool_summary = "\n".join(tool_summary_parts)
        prompt = (
            f"用户问题: {context.user_message}\n\n"
            f"以下是工具执行的结果:\n{tool_summary}\n\n"
            f"请基于以上结果，用中文给出完整、清晰的回答。"
        )

        try:
            response = await asyncio.wait_for(
                self._llm_chat(
                    [{"role": "user", "content": prompt}],
                    purpose="chat",
                ),
                timeout=120.0,
            )
            return response.content.strip() if response.content else ""
        except Exception as e:
            logger.debug("LLM 摘要生成失败: %s", e)
            return ""

    async def _handle_error(self, ctx: dict[str, Any]) -> dict[str, Any]:
        error_msg = ctx.get("_error", "未知错误")
        context: CognitiveContext | None = ctx.get("_cognitive_context")

        if context:
            context.errors.append(error_msg)

        consecutive_llm_timeouts = sum(
            1 for e in (context.errors if context else [])
            if "LLM" in e and ("超时" in e or "timeout" in e.lower())
        )

        if consecutive_llm_timeouts >= 2 and context:
            if not context.autonomous_executed:
                logger.info("连续 %d 次 LLM 超时，触发自主执行模式", consecutive_llm_timeouts)
                autonomous_result = await self._autonomous_execution(context)
                if autonomous_result:
                    context.autonomous_executed = True
                    context.final_output = autonomous_result
                    return {"recovered": False, "is_complete": True, "has_final_text": True, "_final_text": autonomous_result}

            if context.tool_history:
                has_results = any(
                    "error" not in t and not t.get("result", "").startswith("[")
                    for t in context.tool_history
                )
                if has_results:
                    return {"recovered": False}

        if context and context.tool_history:
            return {"recovered": True}

        return {"recovered": False}
