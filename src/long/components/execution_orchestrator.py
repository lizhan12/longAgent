from __future__ import annotations

import asyncio
import json
import logging
import re
import time as _time
from typing import Any, Callable

from long.ir.plan_ir import ActionType
from long.llm.base import LLMMessage


def _attribute_tool_result(tool_name: str, raw_result: str) -> str:
    """在工具结果前注入归因标记，让 LLM 明确知道这是自己调用的工具返回的结果"""
    if raw_result.startswith("[") and ("]" in raw_result[:80]):
        return raw_result
    attribution_map = {
        "query_weather": "这是我（AI）自己调用 query_weather 查询和风天气API得到的结果，不是用户提供的。回复时说\"根据天气数据\"而非\"根据您提供的\"",
        "tavily_search": "这是我（AI）自己调用 tavily_search 搜索得到的结果，不是用户提供的。回复时说\"根据搜索结果\"而非\"根据您提供的\"",
        "execute_code": "这是我（AI）自己调用 execute_code 执行的结果，不是用户提供的",
        "execute_file": "这是我（AI）自己调用 execute_file 执行的结果，不是用户提供的",
        "write_file": "这是我（AI）自己调用 write_file 写入的结果",
        "read_file": "这是我（AI）自己调用 read_file 读取的文件内容",
        "list_files": "这是我（AI）自己调用 list_files 获取的文件列表",
    }
    label = attribution_map.get(tool_name, f"这是我（AI）自己调用 {tool_name} 的结果")
    return f"[注意：{label}]\n{raw_result}"
from long.memory.base import MemoryType
from long.observability.tracing import current_trace_id

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 8
MAX_SEARCH_ROUNDS = 2
_MAX_TOOL_RESULT_LEN = 8000
_ROUND_TIMEOUT = 150.0

_CODE_TASK_PATTERNS = (
    "排序", "算法", "写代码", "实现", "编程", "函数", "程序",
    "生成图表", "画图", "数据分析", "可视化", "折线图", "柱状图",
    "快速排序", "归并排序", "冒泡排序", "桶排序", "树排序",
    "二叉树", "链表", "哈希表", "栈", "队列",
)

_FABRICATED_RESULT_PATTERNS = (
    "测试结果", "运行结果", "执行结果", "排序结果", "输出结果",
    "程序输出", "运行输出", "测试通过", "测试成功",
    "Output:", "Result:", "Test passed",
)


def _extract_user_message(history_msgs: list[dict[str, Any]]) -> str:
    for m in reversed(history_msgs):
        if m.get("role") == "user" and m.get("content"):
            return m["content"]
    return ""


def _has_code_tool_history(history_msgs: list[dict[str, Any]]) -> bool:
    return any(
        m.get("role") == "tool"
        and any(
            kw in (m.get("content", "")[:100])
            for kw in ("✅", "成功", "文件已保存", "执行完成")
        )
        for m in history_msgs
    )


def _has_code_execution_history(history_msgs: list[dict[str, Any]]) -> bool:
    tool_msgs = [m for m in history_msgs if m.get("role") == "tool"]
    has_write = False
    has_exec = False
    for m in tool_msgs:
        content = m.get("content", "")[:200]
        if "execute_code" in content or "execute_file" in content or "执行完成" in content or "执行成功" in content:
            has_exec = True
        if "write_file" in content or "文件已保存" in content or "写入成功" in content:
            has_write = True
    return has_write and has_exec


def _detect_fabricated_results(text: str, has_code_exec: bool) -> bool:
    if has_code_exec:
        return False
    if not text:
        return False
    has_fabricated_pattern = any(p in text for p in _FABRICATED_RESULT_PATTERNS)
    has_code_block = "```" in text and ("python" in text.lower() or "def " in text or "import " in text)
    return has_fabricated_pattern and has_code_block


class ExecutionOrchestrator:

    def __init__(
        self,
        *,
        llm: Any,
        tool_manager: Any,
        plan_executor: Any,
        dialog_compressor: Any,
        memory: Any,
        tracer: Any,
        budget_tokens: int,
        constraint_validator: Any,
        state_machine: Any,
        ir_parser: Any,
        type_checker: Any,
        ltl_validator: Any,
        active_session_getter: Callable[[], Any],
        configs: dict[str, Any],
        prompt_builder_getter: Callable[[], Any],
        session_manager_getter: Callable[[], Any],
        memory_bridge_getter: Callable[[], Any],
    ) -> None:
        self.llm = llm
        self.tool_manager = tool_manager
        self.plan_executor = plan_executor
        self.dialog_compressor = dialog_compressor
        self.memory = memory
        self.tracer = tracer
        self.budget_tokens = budget_tokens
        self.constraint_validator = constraint_validator
        self.state_machine = state_machine
        self.ir_parser = ir_parser
        self.type_checker = type_checker
        self.ltl_validator = ltl_validator
        self._active_session_getter = active_session_getter
        self._configs = configs
        self._prompt_builder_getter = prompt_builder_getter
        self._session_manager_getter = session_manager_getter
        self._memory_bridge_getter = memory_bridge_getter

    @property
    def active_session(self) -> Any:
        return self._active_session_getter()

    def _save_session(self) -> None:
        sm = self._session_manager_getter()
        if sm is not None:
            sm.save_session()

    def _schedule_auto_eval(self) -> None:
        mb = self._memory_bridge_getter()
        if mb is not None:
            mb.schedule_auto_eval()

    def _record_llm_stats(self, response: Any) -> None:
        mb = self._memory_bridge_getter()
        if mb is not None:
            mb.record_llm_stats(response)

    def _record_llm_timeout(self) -> None:
        mb = self._memory_bridge_getter()
        if mb is not None:
            mb.record_llm_timeout()

    def _check_output_safety(self, text: str) -> None:
        mb = self._memory_bridge_getter()
        if mb is not None:
            mb.check_output_safety(text)

    @staticmethod
    def to_llm_messages(messages: list[dict[str, Any]]) -> list:
        result = []
        for m in messages:
            if isinstance(m, LLMMessage):
                result.append(m)
            elif isinstance(m, dict):
                result.append(LLMMessage(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                    name=m.get("name"),
                ))
            else:
                result.append(m)
        return result

    async def chat_with_tools_loop(
        self, cli_adapter: Any, history_msgs: list[dict[str, str]], tools: list[dict[str, Any]]
    ) -> None:
        plan_executed = await self.try_plan_execution(
            cli_adapter, history_msgs, tools
        )
        if plan_executed:
            return

        cognitive_ok = await self.cognitive_runtime_loop(
            cli_adapter, history_msgs, tools
        )
        if cognitive_ok:
            return

        await self.fallback_tool_call_loop(cli_adapter, history_msgs, tools)

    async def cognitive_runtime_loop(
        self, cli_adapter: Any, history_msgs: list[dict[str, str]], tools: list[dict[str, Any]]
    ) -> bool:
        try:
            from long.cognitive.runtime import (
                CognitiveRuntime, CognitiveContext,
            )
        except ImportError:
            logger.warning("Cognitive Runtime 不可用，降级到 Fallback 模式")
            return False

        async def llm_chat_fn(messages, purpose="chat"):
            llm_msgs = self.to_llm_messages(messages)
            return await self.llm.chat(llm_msgs, purpose=purpose)

        async def llm_chat_with_tools_fn(messages, tools_list, purpose="chat", **kwargs):
            llm_msgs = self.to_llm_messages(messages)
            return await self.llm.chat_with_tools(llm_msgs, tools_list, purpose=purpose, **kwargs)

        async def tool_execute_fn(tool_name, arguments):
            return await self.tool_manager.execute_tool(tool_name, arguments)

        async def output_fn(text):
            if text:
                cli_adapter.console.print(text)
                if self.active_session is not None:
                    self.active_session.add_message("assistant", text)
                    self._save_session()

        tool_capability_registry = None
        try:
            from long.capabilities.tool_capability import ToolCapabilityRegistry
            tool_capability_registry = ToolCapabilityRegistry()
        except ImportError:
            pass

        runtime = CognitiveRuntime(
            llm_chat_fn=llm_chat_fn,
            llm_chat_with_tools_fn=llm_chat_with_tools_fn,
            tool_execute_fn=tool_execute_fn,
            output_fn=output_fn,
            memory_controller=self.memory,
            tool_capability_registry=tool_capability_registry,
        )

        context = CognitiveContext(
            user_message=history_msgs[-1].get("content", "") if history_msgs else "",
            messages=list(history_msgs),
            max_rounds=8,
        )

        graph_context = {
            "_cognitive_context": context,
            "_tools": self.tool_manager.clean_tools_for_api(tools),
        }

        try:
            result_context = await runtime.run(
                context, extra={"_tools": self.tool_manager.clean_tools_for_api(tools)}
            )
            return result_context.is_complete
        except Exception as e:
            logger.warning("Cognitive Runtime 执行失败: %s，降级到 Fallback", e)
            return False

    async def try_plan_execution(
        self,
        cli_adapter: Any,
        history_msgs: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> bool:
        if self.plan_executor is None:
            return False

        user_msgs = [m for m in history_msgs if m.get("role") == "user"]
        if not user_msgs:
            return False

        user_message = user_msgs[-1].get("content", "")

        complexity = self.plan_executor.classifier.classify(user_message, tools)

        complexity_style = {
            "simple": "[green]简单[/green]",
            "moderate": "[yellow]中等[/yellow]",
            "complex": "[red]复杂[/red]",
        }
        complexity_label = complexity_style.get(complexity.level.value, complexity.level.value)

        logger.info(
            "任务复杂度: %s (score=%.1f, reasons=%s)",
            complexity.level.value,
            complexity.score,
            complexity.reasons,
        )

        if not complexity.needs_planning:
            cli_adapter.console.print(
                f"[dim]任务复杂度: {complexity_label}，使用直接工具调用模式[/dim]"
            )
            return False

        cli_adapter.console.print(
            f"[dim]任务复杂度: {complexity_label}，生成结构化执行计划...[/dim]"
        )

        with cli_adapter.console.status("[bold cyan]📋 正在生成执行计划...[/bold cyan]", spinner="dots"):
            _PLAN_MAX_RETRIES = 2
            plan = None
            for _plan_attempt in range(_PLAN_MAX_RETRIES):
                try:
                    plan = await asyncio.wait_for(
                        self.plan_executor.generate_plan(
                            user_message=user_message,
                            history_msgs=history_msgs,
                            available_tools=tools,
                        ),
                        timeout=180,
                    )
                    if plan is not None:
                        break
                except asyncio.TimeoutError:
                    self._record_llm_timeout()
                    if _plan_attempt < _PLAN_MAX_RETRIES - 1:
                        cli_adapter.console.print(f"[dim]计划生成超时，重试 {_plan_attempt + 2}/{_PLAN_MAX_RETRIES}...[/dim]")
                    else:
                        cli_adapter.console.print("[dim]计划生成超时，降级为直接工具调用模式[/dim]")
                        return False
                except Exception:
                    if _plan_attempt < _PLAN_MAX_RETRIES - 1:
                        cli_adapter.console.print(f"[dim]计划生成失败，重试 {_plan_attempt + 2}/{_PLAN_MAX_RETRIES}...[/dim]")
                    else:
                        cli_adapter.console.print("[dim]计划生成失败，降级为直接工具调用模式[/dim]")
                        return False

        if plan is None:
            cli_adapter.console.print("[dim]计划生成失败，降级为直接工具调用模式[/dim]")
            return False

        if len(plan.steps) <= 1:
            cli_adapter.console.print("[dim]计划仅含单步，使用直接工具调用模式[/dim]")
            return False

        async def tool_executor(tool_name: str, arguments: dict[str, Any]) -> str:
            return await self.tool_manager.execute_tool(tool_name, arguments)

        exec_result = await self.plan_executor.execute_plan(
            plan=plan,
            cli_adapter=cli_adapter,
            tool_executor=tool_executor,
            history_msgs=history_msgs,
        )

        if exec_result.success and exec_result.output_text:
            self.active_session.add_message("assistant", exec_result.output_text)
            cli_adapter.console.print()

            if self.memory is not None:
                try:
                    await self.memory.store(
                        f"assistant: {exec_result.output_text}",
                        memory_type=MemoryType.EPISODIC,
                        importance=0.5,
                    )
                except Exception:
                    pass
            self._save_session()
            self._schedule_auto_eval()
            return True

        if exec_result.success and not exec_result.output_text:
            cli_adapter.console.print()
            gen_status = cli_adapter.console.status(
                "[bold green]✨ 正在生成回复...[/bold green]", spinner="bouncingBar"
            )
            gen_status.start()
            try:
                response_parts: list[str] = []
                first_token = True
                async for token in self.llm.stream_chat(
                    [LLMMessage(role=m["role"], content=m.get("content", ""), tool_calls=m.get("tool_calls"), tool_call_id=m.get("tool_call_id")) for m in history_msgs],
                    purpose="chat",
                ):
                    if first_token:
                        gen_status.stop()
                        first_token = False
                    response_parts.append(token)
                    cli_adapter.console.print(token, end="", highlight=False)
                response_text = "".join(response_parts)
            finally:
                gen_status.stop()

            cli_adapter.console.print()
            cli_adapter.console.print()

            self.active_session.add_message("assistant", response_text)

            if self.memory is not None:
                try:
                    await self.memory.store(
                        f"assistant: {response_text}",
                        memory_type=MemoryType.EPISODIC,
                        importance=0.5,
                    )
                except Exception:
                    pass
            self._save_session()
            self._schedule_auto_eval()
            return True

        if not exec_result.success:
            cli_adapter.console.print("[dim]计划执行未完成，尝试直接工具调用...[/dim]")
            return False

        return True

    async def fallback_tool_call_loop(
        self, cli_adapter: Any, history_msgs: list[dict[str, str]], tools: list[dict[str, Any]]
    ) -> None:
        search_call_count: int = 0
        search_queries_used: list[str] = []
        last_round_had_search: bool = False

        tool_source_map: dict[str, str] = {}
        mcp_server_map: dict[str, str] = {}
        for tool_entry in tools:
            func = tool_entry.get("function", {})
            name = func.get("name", "")
            source = tool_entry.get("_source", "local")
            tool_source_map[name] = source
            mcp_server = tool_entry.get("_mcp_server")
            if mcp_server:
                mcp_server_map[name] = mcp_server

        budget_remaining = MAX_TOOL_ROUNDS

        for _round in range(MAX_TOOL_ROUNDS):
            if (
                self.dialog_compressor is not None
                and _round > 0
                and self.dialog_compressor.should_compress(history_msgs, tool_rounds=_round)
            ):
                history_msgs = await self.dialog_compressor.compress(
                    self.llm, history_msgs, tool_rounds=_round,
                )

            llm_messages = []
            for m in history_msgs:
                msg = LLMMessage(
                    role=m["role"],
                    content=m.get("content", ""),
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                )
                llm_messages.append(msg)

            with cli_adapter.console.status("[bold cyan]⏳ 正在思考...[/bold cyan]", spinner="dots"):
                trace = self.tracer.get_trace(current_trace_id()) if current_trace_id() else None
                span_ctx = None
                if trace is not None:
                    span_ctx = trace.span("llm.chat_with_tools", attributes={"round": _round + 1})
                    span_ctx.__enter__()

                round_deadline = _time.monotonic() + _ROUND_TIMEOUT

                try:
                    response = await asyncio.wait_for(
                        self.llm.chat_with_tools(
                            llm_messages, self.tool_manager.clean_tools_for_api(tools),
                            purpose="chat", deadline=round_deadline,
                        ),
                        timeout=_ROUND_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    self._record_llm_timeout()
                    if span_ctx is not None:
                        from long.observability.tracing import SpanStatus
                        span_ctx._span.finish(SpanStatus.TIMEOUT)
                        span_ctx.__exit__(None, None, None)

                    tool_results_in_history = [
                        m for m in history_msgs if m["role"] == "tool"
                    ]
                    if tool_results_in_history and _round > 0:
                        logger.warning(
                            "LLM 超时 (第%d轮)，尝试用已有 %d 条工具结果生成回复",
                            _round + 1, len(tool_results_in_history),
                        )
                        fallback_msg = (
                            "抱歉，LLM 服务响应超时。以下是我已获取的信息摘要：\n\n"
                        )
                        for tr in tool_results_in_history:
                            content = tr.get("content", "")
                            if content and not content.startswith("❌"):
                                preview = content[:500].strip()
                                fallback_msg += f"```\n{preview}\n```\n\n"
                        fallback_msg += "\n请基于以上信息继续操作，或稍后重试。"
                        cli_adapter.console.print(fallback_msg)
                        if self.active_session is not None:
                            self.active_session.add_message("assistant", fallback_msg)
                            self._save_session()
                        return
                    raise
                except Exception:
                    if span_ctx is not None:
                        span_ctx.__exit__(None, None, None)
                    raise

                if span_ctx is not None:
                    span_ctx.__exit__(None, None, None)

            if response.tool_calls:
                _PARALLEL_SAFE_TOOLS = frozenset({
                    "read_file", "list_files", "get_current_time",
                    "read_skill_md",
                })

                parallel_calls: list[tuple[int, dict[str, Any]]] = []
                serial_calls: list[tuple[int, dict[str, Any]]] = []
                all_tool_calls: list[dict[str, Any]] = []
                intercepted_ids: set[str] = set()
                intercept_reasons: dict[str, str] = {}
                _redirected_ids: set[str] = set()

                for idx, tc in enumerate(response.tool_calls):
                    tool_name = tc["name"]
                    arguments = tc["arguments"]

                    tc_def = {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tool_name, "arguments": json.dumps(arguments)},
                    }
                    all_tool_calls.append(tc_def)

                    if tool_name == "tavily_search":
                        query = arguments.get("query", "")

                        _WEATHER_KEYWORDS = (
                            "天气", "天气预报", "气温", "温度", "湿度", "风力",
                            "下雨", "下雨吗", "降雨", "降水", "紫外线",
                            "weather", "forecast", "temperature",
                        )
                        _query_lower = query.lower()
                        _is_weather_query = any(kw in _query_lower for kw in _WEATHER_KEYWORDS)
                        if _is_weather_query:
                            _city_patterns = [
                                r'([\u4e00-\u9fff]{2,4})(?:的|市|地区)?(?:天气|气温|温度|湿度|风力|降雨|降水|天气预报|实时天气)',
                                r'(?:天气|气温|温度|湿度|风力|降雨|降水|天气预报|实时天气)([\u4e00-\u9fff]{2,4})',
                            ]
                            _city_name = ""
                            for _pat in _city_patterns:
                                _m = re.search(_pat, query)
                                if _m:
                                    _city_name = _m.group(1).strip()
                                    break
                            if not _city_name:
                                _city_name = re.sub(
                                    r'(天气|天气预报|气温|温度|湿度|风力|降雨|降水|实时|今日|今天|明天|本周|一周|未来|的|如何|怎么样|查询|搜索|\d+年?\d*月?\d*日?|[a-zA-Z]+)',
                                    '', query,
                                ).strip()
                            _city_name = re.sub(
                                r'^(今日|明天|后天|大后天|本周|下周|上周|未来|一周|实时|当前|现在|附近|周边)',
                                '', _city_name,
                            ).strip()
                            _city_name = re.sub(
                                r'(市|地区|的|省|县|区|实时|今日|今天|明天|本周|一周|未来|当前|现在|附近|周边)$',
                                '', _city_name,
                            ).strip()
                            _non_city_prefixes = ("今日", "明天", "后天", "本周", "下周", "上周", "未来", "一周", "实时", "当前", "现在")
                            while _city_name and any(_city_name.startswith(p) for p in _non_city_prefixes):
                                for p in _non_city_prefixes:
                                    if _city_name.startswith(p):
                                        _city_name = _city_name[len(p):]
                                        break
                            if _city_name:
                                logger.info(
                                    "天气查询拦截: tavily_search → query_weather, city=%s, query=%s",
                                    _city_name, query,
                                )
                                tc["name"] = "query_weather"
                                tc["arguments"] = {"city": _city_name}
                                tool_name = "query_weather"
                                arguments = {"city": _city_name}
                                tc_def["function"]["name"] = "query_weather"
                                tc_def["function"]["arguments"] = json.dumps(arguments)
                                _redirected_ids.add(tc["id"])
                                cli_adapter.console.print(
                                    f"[dim]🔧 query_weather(city={_city_name!r}) (自动从 tavily_search 重定向)[/dim]"
                                )
                            else:
                                intercept_reasons[tc["id"]] = (
                                    "[天气查询拦截] 天气查询请使用 query_weather 工具，"
                                    "参数 city 传入城市名（如'杭州'）。"
                                    "不要使用 tavily_search 查询天气。"
                                )
                                intercepted_ids.add(tc["id"])
                                continue

                        if tool_name == "tavily_search" and search_call_count >= MAX_SEARCH_ROUNDS:
                            logger.warning(
                                "搜索次数已达上限 (%d/%d)，拦截搜索: %s",
                                search_call_count, MAX_SEARCH_ROUNDS, query,
                            )
                            intercept_reasons[tc["id"]] = (
                                f"[搜索限制] 已达到最大搜索次数 ({MAX_SEARCH_ROUNDS})，"
                                f"请基于已有信息生成回复，不得继续搜索。"
                            )
                            intercepted_ids.add(tc["id"])
                            continue

                        if tool_name == "tavily_search" and last_round_had_search and search_call_count > 0:
                            logger.debug(
                                "ReAct 约束：连续搜索被拦截: %s (上一轮已搜索，需先分析结果)",
                                query,
                            )
                            intercept_reasons[tc["id"]] = (
                                f"[ReAct 约束] 你刚搜索过，请先分析已有搜索结果再决定是否需要再次搜索。"
                                f"如果结果足够，请直接使用；如果确实不足，下一轮再搜索。"
                            )
                            intercepted_ids.add(tc["id"])
                            continue

                        is_duplicate = False
                        if tool_name == "tavily_search":
                            for prev_query in search_queries_used:
                                if query in prev_query or prev_query in query:
                                    logger.debug(
                                        "检测到重复搜索，拦截: %s (已搜索过: %s)",
                                        query, prev_query,
                                    )
                                    intercept_reasons[tc["id"]] = (
                                        f"[搜索限制] 你已搜索过类似内容（{prev_query}），"
                                        f"请基于已有信息生成回复，不得重复搜索。"
                                    )
                                    intercepted_ids.add(tc["id"])
                                    is_duplicate = True
                                    break

                        if tool_name == "tavily_search" and not is_duplicate and tc["id"] not in intercepted_ids:
                            search_call_count += 1
                            search_queries_used.append(query)
                            logger.info(
                                "允许搜索 #%d: %s", search_call_count, query,
                            )

                    # 拦截重复的天气查询
                    if tool_name == "query_weather":
                        _weather_city = arguments.get("city", "")
                        _weather_cities = set(c.strip() for c in _weather_city.replace("，", ",").replace("、", ",").split(",") if c.strip())
                        if hasattr(self, '_weather_queried_cities'):
                            _already_queried = _weather_cities & self._weather_queried_cities
                            if _already_queried:
                                logger.info(
                                    "拦截重复天气查询: %s (已查询: %s)",
                                    _weather_city, ', '.join(_already_queried),
                                )
                                intercept_reasons[tc["id"]] = (
                                    f"[天气查询限制] 以下城市已查询过天气: {', '.join(_already_queried)}，"
                                    f"请直接使用已有数据回答，不要重复查询。"
                                )
                                intercepted_ids.add(tc["id"])
                                continue
                        else:
                            self._weather_queried_cities = set()
                        self._weather_queried_cities.update(_weather_cities)

                    if budget_remaining <= 0:
                        cli_adapter.console.print("[yellow]⚠ 预算已耗尽，停止工具调用[/yellow]")
                        intercepted_ids.add(tc["id"])
                        intercept_reasons[tc["id"]] = "[预算耗尽] 已达到最大工具调用轮次"
                        continue

                    if tool_name in ("delete_file",) and not self.check_dangerous_tool(tool_name, arguments):
                        intercepted_ids.add(tc["id"])
                        intercept_reasons[tc["id"]] = f"[安全拦截] {tool_name} 操作需要用户确认"
                        continue

                    if (tool_name not in ("tavily_search",) or tc["id"] not in intercepted_ids) and tc["id"] not in _redirected_ids:
                        cli_adapter.console.print(f"[dim]🔧 {tool_name}({arguments})[/dim]")

                    if tool_name in _PARALLEL_SAFE_TOOLS and budget_remaining > 0:
                        parallel_calls.append((idx, tc))
                    else:
                        serial_calls.append((idx, tc))

                history_msgs.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": all_tool_calls,
                })

                current_round_has_search = any(
                    tc["name"] == "tavily_search" and tc["id"] not in intercepted_ids
                    for tc in response.tool_calls
                )

                for tc_id in intercepted_ids:
                    history_msgs.append({
                        "role": "tool",
                        "content": intercept_reasons[tc_id],
                        "tool_call_id": tc_id,
                    })

                if len(parallel_calls) > 1:
                    active_parallel = [(idx, tc) for idx, tc in parallel_calls if tc["id"] not in intercepted_ids]
                    if active_parallel:
                        async def _exec_one(tc: dict[str, Any]) -> tuple[str, str]:
                            return tc["id"], await self.tool_manager.execute_tool(tc["name"], tc["arguments"])

                        with cli_adapter.console.status("[bold yellow]🔧 并行执行工具...[/bold yellow]", spinner="line"):
                            results = await asyncio.gather(
                                *[_exec_one(tc) for _, tc in active_parallel],
                                return_exceptions=True,
                            )
                        for result in results:
                            if isinstance(result, Exception):
                                logger.warning("并行工具执行异常: %s", result)
                                continue
                            tc_id, tool_result = result
                            is_error = tool_result.startswith((
                                "未知工具:", "工具执行失败:", "工具异常:", "MCP工具异常:",
                                "Skill 执行错误:", "Skill 工具", "沙箱未初始化",
                                "TAVILY_API_KEY 未配置", "搜索执行失败", "搜索超时",
                                "Tavily 搜索脚本未找到",
                            )) or any(
                                tool_result.startswith(f"{prefix} 错误:")
                                for prefix in ("list_files", "read_file", "write_file", "delete_file")
                            ) or any(
                                tool_result.startswith(f"{prefix} 失败:")
                                for prefix in ("执行文件", "读取 SKILL.md")
                            ) or tool_result.startswith("代码内容为空") or tool_result.startswith("文件路径不能为空") or tool_result.startswith("不支持执行")
                            budget_remaining -= 1
                            history_msgs.append({
                                "role": "tool",
                                "content": _attribute_tool_result(tc["name"], tool_result),
                                "tool_call_id": tc_id,
                            })
                else:
                    for _, tc in parallel_calls:
                        if tc["id"] in intercepted_ids:
                            continue
                        with cli_adapter.console.status(f"[bold yellow]🔧 执行 {tc['name']}...[/bold yellow]", spinner="line"):
                            tool_result = await self.tool_manager.execute_tool(tc["name"], tc["arguments"])

                        if len(tool_result) > _MAX_TOOL_RESULT_LEN:
                            tool_result = tool_result[:_MAX_TOOL_RESULT_LEN] + "\n...(结果已截断)"

                        is_error = tool_result.startswith((
                            "未知工具:", "工具执行失败:", "工具异常:", "MCP工具异常:",
                            "Skill 执行错误:", "Skill 工具", "沙箱未初始化",
                            "TAVILY_API_KEY 未配置", "搜索执行失败", "搜索超时",
                            "Tavily 搜索脚本未找到",
                        )) or any(
                            tool_result.startswith(f"{prefix} 错误:")
                            for prefix in ("list_files", "read_file", "write_file", "delete_file")
                        ) or any(
                            tool_result.startswith(f"{prefix} 失败:")
                            for prefix in ("执行文件", "读取 SKILL.md")
                        ) or tool_result.startswith("代码内容为空") or tool_result.startswith("文件路径不能为空") or tool_result.startswith("不支持执行")
                        budget_remaining -= 1
                        history_msgs.append({
                            "role": "tool",
                            "content": _attribute_tool_result(tc["name"], tool_result),
                            "tool_call_id": tc["id"],
                        })

                for _, tc in serial_calls:
                    if tc["id"] in intercepted_ids:
                        continue
                    tool_name = tc["name"]
                    arguments = tc["arguments"]

                    if budget_remaining <= 0:
                        history_msgs.append({
                            "role": "tool",
                            "content": "[预算耗尽] 已达到最大工具调用轮次",
                            "tool_call_id": tc["id"],
                        })
                        continue

                    if tool_name in ("delete_file",) and not self.check_dangerous_tool(tool_name, arguments):
                        history_msgs.append({
                            "role": "tool",
                            "content": f"[安全拦截] {tool_name} 操作需要用户确认",
                            "tool_call_id": tc["id"],
                        })
                        continue

                    with cli_adapter.console.status(f"[bold yellow]🔧 执行 {tool_name}...[/bold yellow]", spinner="line"):
                        tool_result = await self.tool_manager.execute_tool(tool_name, arguments)

                    if len(tool_result) > _MAX_TOOL_RESULT_LEN:
                        tool_result = tool_result[:_MAX_TOOL_RESULT_LEN] + "\n...(结果已截断)"

                    is_error = tool_result.startswith((
                        "未知工具:", "工具执行失败:", "工具异常:", "MCP工具异常:",
                        "Skill 执行错误:", "Skill 工具", "沙箱未初始化",
                        "TAVILY_API_KEY 未配置", "搜索执行失败", "搜索超时",
                        "Tavily 搜索脚本未找到",
                    )) or any(
                        tool_result.startswith(f"{prefix} 错误:")
                        for prefix in ("list_files", "read_file", "write_file", "delete_file")
                    ) or any(
                        tool_result.startswith(f"{prefix} 失败:")
                        for prefix in ("执行文件", "读取 SKILL.md")
                    ) or tool_result.startswith("代码内容为空") or tool_result.startswith("文件路径不能为空") or tool_result.startswith("不支持执行")
                    if is_error:
                        cli_adapter.console.print(f"[dim]  ❌ {tool_name}: {tool_result[:80]}[/dim]")
                    else:
                        result_preview = tool_result[:60].replace("\n", " ") + ("..." if len(tool_result) > 60 else "")
                        cli_adapter.console.print(f"[dim]  ✅ {tool_name}: {result_preview}[/dim]")
                    budget_remaining -= 1

                    history_msgs.append({
                        "role": "tool",
                        "content": _attribute_tool_result(tool_name, tool_result),
                        "tool_call_id": tc["id"],
                    })

                _CODE_TOOLS = frozenset({"write_file", "execute_code", "execute_file"})
                has_code_action = any(
                    tc["name"] in _CODE_TOOLS and tc["id"] not in intercepted_ids
                    for tc in response.tool_calls
                )
                if has_code_action:
                    last_round_had_search = False
                elif current_round_has_search:
                    last_round_had_search = True
            else:
                cli_adapter.console.print()

                response_text = response.content or ""

                self._record_llm_stats(response)
                self._check_output_safety(response_text)

                user_msg = _extract_user_message(history_msgs)
                needs_code = any(p in user_msg for p in _CODE_TASK_PATTERNS)
                has_code_tools = _has_code_tool_history(history_msgs)
                has_code_exec = _has_code_execution_history(history_msgs)
                has_any_tool = any(m.get("role") == "tool" for m in history_msgs)

                if needs_code and not has_code_tools and _round < MAX_TOOL_ROUNDS - 1:
                    if not has_any_tool:
                        logger.info("检测到需要代码的任务但 LLM 未调用工具，强制重试")
                    else:
                        logger.info("检测到需要代码的任务但 LLM 未调用代码工具，强制重试")
                    history_msgs.append({"role": "assistant", "content": response_text})
                    history_msgs.append({
                        "role": "user",
                        "content": (
                            "这个任务需要写代码并执行，"
                            "使用 write_file 写入代码文件，然后用 execute_file 执行该文件（传入 path 参数，如 path='output/xxx.py'）。"
                            "不要只在文本中描述结果，必须实际执行代码。"
                            "不要调用 list_files 等无关工具。"
                        ),
                    })
                    continue

                if needs_code and has_code_tools and not has_code_exec and _round < MAX_TOOL_ROUNDS - 1:
                    logger.info("检测到需要代码的任务：代码已写入但未执行，强制重试")
                    history_msgs.append({"role": "assistant", "content": response_text})
                    history_msgs.append({
                        "role": "user",
                        "content": (
                            "你已经用 write_file 写入了代码文件，但还没有执行它。"
                            "请立即调用 execute_file 工具执行该文件（传入 path 参数）。"
                            "不要在文本中编造运行结果。"
                        ),
                    })
                    continue

                if _detect_fabricated_results(response_text, has_code_exec):
                    logger.info("检测到 LLM 编造了不存在的执行结果，强制重试")
                    history_msgs.append({"role": "assistant", "content": response_text})
                    history_msgs.append({
                        "role": "user",
                        "content": (
                            "⚠️ 你刚才的回复包含了编造的测试/执行结果。"
                            "你并没有调用 execute_code/execute_file 工具，所以不可能有真实的执行结果。"
                            "请使用 write_file 写入代码文件，然后用 execute_file 执行该文件（传入 path 参数），"
                            "展示真实的工具返回结果。禁止编造执行结果。"
                        ),
                    })
                    continue

                if response_text:
                    cli_adapter.console.print(response_text)
                else:
                    gen_status = cli_adapter.console.status(
                        "[bold green]✨ 正在生成回复...[/bold green]", spinner="bouncingBar"
                    )
                    gen_status.start()
                    try:
                        response_parts: list[str] = []
                        first_token = True
                        async for token in self.llm.stream_chat(
                            [LLMMessage(role=m["role"], content=m.get("content", ""), tool_calls=m.get("tool_calls"), tool_call_id=m.get("tool_call_id")) for m in history_msgs],
                            purpose="chat",
                        ):
                            if first_token:
                                gen_status.stop()
                                first_token = False
                            response_parts.append(token)
                            cli_adapter.console.print(token, end="", highlight=False)
                        response_text = "".join(response_parts)
                    except Exception as e:
                        gen_status.stop()
                        logger.warning("stream_chat 失败: %s", e)
                    finally:
                        gen_status.stop()

                    if not response_text:
                        tool_results = [m for m in history_msgs if m["role"] == "tool"]
                        if tool_results:
                            response_text = "任务已完成。以下是执行结果摘要：\n\n"
                            for tr in tool_results:
                                content = tr.get("content", "")
                                if content and not content.startswith("❌"):
                                    preview = content[:300].strip()
                                    response_text += f"```\n{preview}\n```\n\n"
                            cli_adapter.console.print(response_text)

                cli_adapter.console.print()
                cli_adapter.console.print()

                self.active_session.add_message("assistant", response_text)

                if self.memory is not None:
                    try:
                        await self.memory.store(
                            f"assistant: {response_text}",
                            memory_type=MemoryType.EPISODIC,
                            importance=0.5,
                        )
                    except Exception:
                        pass
                self._save_session()
                self._schedule_auto_eval()
                return

        response_text = response.content or ""
        self._record_llm_stats(response)
        self._check_output_safety(response_text)
        self.active_session.add_message("assistant", response_text)
        cli_adapter.console.print()
        cli_adapter.console.print(response_text)
        cli_adapter.console.print()
        self._save_session()
        self._schedule_auto_eval()

    def check_dangerous_tool(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        high_risk_tools = frozenset({"delete_file"})
        if tool_name in high_risk_tools:
            if self.tool_manager.security_mode == "service":
                return False
        return True

    def build_virtual_step_args(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_source_map: dict[str, str],
        mcp_server_map: dict[str, str],
    ) -> tuple[str, dict[str, Any]]:
        source = tool_source_map.get(tool_name, "local")

        if source == "skill":
            return ActionType.CALL_SKILL.value, {
                "skill_name": tool_name,
                "arguments": arguments,
            }
        elif source == "mcp":
            server_name = mcp_server_map.get(tool_name, "")
            parts = tool_name.split("_", 1)
            actual_tool_name = parts[1] if len(parts) == 2 else tool_name
            return ActionType.CALL_MCP.value, {
                "server_name": server_name,
                "tool_name": actual_tool_name,
                "arguments": arguments,
            }
        else:
            if tool_name in ("list_files", "read_file", "read_skill_md"):
                return ActionType.SEARCH.value, {
                    "tool_name": tool_name,
                    "parameters": arguments,
                }
            return ActionType.CALL_TOOL.value, {
                "tool_name": tool_name,
                "parameters": arguments,
            }

    @staticmethod
    def map_tool_to_action(tool_name: str) -> str:
        tool_action_map = {
            "list_files": ActionType.SEARCH.value,
            "read_file": ActionType.SEARCH.value,
            "read_skill_md": ActionType.SEARCH.value,
            "write_file": ActionType.CALL_TOOL.value,
            "delete_file": ActionType.CALL_TOOL.value,
            "execute_code": ActionType.CALL_TOOL.value,
            "execute_file": ActionType.CALL_TOOL.value,
        }

        if tool_name in tool_action_map:
            return tool_action_map[tool_name]

        if "_" in tool_name:
            parts = tool_name.split("_", 1)
            if len(parts) == 2:
                return ActionType.CALL_MCP.value

        return ActionType.CALL_TOOL.value
