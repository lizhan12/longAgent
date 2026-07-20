from __future__ import annotations

import asyncio
import json
import logging
import time as _time
from typing import Any, Callable

from long.llm.base import LLMMessage
from long.memory.base import MemoryType
from long.observability.tracing import current_trace_id

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 8
MAX_SEARCH_ROUNDS = 2
MAX_TOOL_RESULT_LEN = 8000
ROUND_TIMEOUT = 150.0

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

_PARALLEL_SAFE_TOOLS = frozenset({
    "read_file", "list_files", "get_current_time",
    "read_skill_md",
})

_CODE_TOOLS = frozenset({"write_file", "execute_code", "execute_file"})

_ERROR_PREFIXES = (
    "未知工具:", "工具执行失败:", "工具异常:", "MCP工具异常:",
    "Skill 执行错误:", "Skill 工具", "沙箱未初始化",
    "TAVILY_API_KEY 未配置", "搜索执行失败", "搜索超时",
    "Tavily 搜索脚本未找到",
)

_ERROR_TOOL_PREFIXES = ("list_files", "read_file", "write_file", "delete_file")

_ERROR_FAIL_PREFIXES = ("执行文件", "读取 SKILL.md")

_ERROR_STARTS = ("代码内容为空", "文件路径不能为空", "不支持执行")

_IRREVERSIBLE_TOOLS = frozenset({"delete_file", "execute_code", "execute_file"})

_HITL_CHECKPOINT_THRESHOLD = 6


class FallbackLoop:

    def __init__(
        self,
        *,
        llm: Any,
        tool_manager: Any,
        dialog_compressor: Any,
        memory: Any,
        tracer: Any,
        budget_tokens: int,
        configs: dict[str, Any],
        active_session_getter: Callable[[], Any],
        session_manager_getter: Callable[[], Any],
        memory_bridge_getter: Callable[[], Any],
    ) -> None:
        self.llm = llm
        self.tool_manager = tool_manager
        self.dialog_compressor = dialog_compressor
        self.memory = memory
        self.tracer = tracer
        self.budget_tokens = budget_tokens
        self.configs = configs
        self._active_session_getter = active_session_getter
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
    def should_continue(
        round_count: int,
        max_rounds: int,
        search_count: int,
        max_searches: int,
        budget_remaining: int,
        has_pending_tools: bool,
    ) -> bool:
        if round_count >= max_rounds:
            return False
        if budget_remaining <= 0 and not has_pending_tools:
            return False
        return True

    def prepare_tools(
        self,
        all_tools: list[dict[str, Any]],
        search_count: int,
        max_searches: int,
        last_was_search: bool,
    ) -> list[dict[str, Any]]:
        cleaned = self.tool_manager.clean_tools_for_api(all_tools)
        if search_count >= max_searches:
            cleaned = [
                t for t in cleaned
                if t.get("function", {}).get("name") != "tavily_search"
            ]
        return cleaned

    @staticmethod
    def is_tool_error(result: str) -> bool:
        return (
            result.startswith(_ERROR_PREFIXES)
            or any(
                result.startswith(f"{prefix} 错误:")
                for prefix in _ERROR_TOOL_PREFIXES
            )
            or any(
                result.startswith(f"{prefix} 失败:")
                for prefix in _ERROR_FAIL_PREFIXES
            )
            or any(result.startswith(s) for s in _ERROR_STARTS)
        )

    @staticmethod
    def handle_tool_result(
        tool_name: str, tool_result: str, max_len: int,
    ) -> tuple[str, bool]:
        if len(tool_result) > max_len:
            tool_result = tool_result[:max_len] + "\n...(结果已截断)"
        is_error = FallbackLoop.is_tool_error(tool_result)
        return tool_result, is_error

    @staticmethod
    def check_search_constraint(
        tool_name: str,
        search_count: int,
        max_searches: int,
        last_was_search: bool,
    ) -> tuple[bool, str]:
        if tool_name != "tavily_search":
            return False, ""
        if search_count >= max_searches:
            return True, (
                f"[搜索限制] 已达到最大搜索次数 ({max_searches})，"
                f"请基于已有信息生成回复，不得继续搜索。"
            )
        if last_was_search and search_count > 0:
            return True, (
                "[ReAct 约束] 你刚搜索过，请先分析已有搜索结果再决定是否需要再次搜索。"
                "如果结果足够，请直接使用；如果确实不足，下一轮再搜索。"
            )
        return False, ""

    async def execute_tools_parallel(
        self,
        tool_calls: list[dict[str, Any]],
        execute_fn: Callable,
    ) -> list[tuple[str, str]]:
        if not tool_calls:
            return []

        if len(tool_calls) == 1:
            tc = tool_calls[0]
            result = await execute_fn(tc["name"], tc["arguments"])
            return [(tc["id"], result)]

        async def _exec_one(tc: dict[str, Any]) -> tuple[str, str]:
            return tc["id"], await execute_fn(tc["name"], tc["arguments"])

        raw_results = await asyncio.gather(
            *[_exec_one(tc) for tc in tool_calls],
            return_exceptions=True,
        )

        results: list[tuple[str, str]] = []
        for result in raw_results:
            if isinstance(result, Exception):
                logger.warning("并行工具执行异常: %s", result)
                continue
            results.append(result)
        return results

    async def execute_tools_serial(
        self,
        tool_calls: list[dict[str, Any]],
        execute_fn: Callable,
    ) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []
        for tc in tool_calls:
            result = await execute_fn(tc["name"], tc["arguments"])
            results.append((tc["id"], result))
        return results

    @staticmethod
    def generate_timeout_reply(
        messages: list[dict[str, Any]],
        llm_chat_fn: Callable | None = None,
    ) -> str:
        tool_results_in_history = [m for m in messages if m["role"] == "tool"]
        if not tool_results_in_history:
            return ""
        fallback_msg = (
            "抱歉，LLM 服务响应超时。以下是我已获取的信息摘要：\n\n"
        )
        for tr in tool_results_in_history:
            content = tr.get("content", "")
            if content and not content.startswith("❌"):
                preview = content[:500].strip()
                fallback_msg += f"```\n{preview}\n```\n\n"
        fallback_msg += "\n请基于以上信息继续操作，或稍后重试。"
        return fallback_msg

    def check_hitl_confirmation(
        self, tool_name: str, arguments: dict[str, Any], cli_adapter: Any,
    ) -> tuple[bool, str]:
        if tool_name not in _IRREVERSIBLE_TOOLS:
            return True, ""

        confirm_fn = getattr(cli_adapter, "confirm_action", None)
        if confirm_fn is not None and callable(confirm_fn):
            message = f"确认执行 {tool_name}?"
            allowed = confirm_fn(message)
            if not allowed:
                return False, f"[HITL 拦截] 用户拒绝执行 {tool_name}"
            return True, ""

        logger.warning(
            "HITL: 不可逆操作 %s 在非交互模式下自动放行（cli_adapter 无 confirm_action 方法）",
            tool_name,
        )
        return True, ""

    async def run(
        self,
        history_msgs: list[dict[str, str]],
        tools: list[dict[str, Any]],
        cli_adapter: Any,
        output_fn: Callable | None = None,
    ) -> None:
        search_call_count: int = 0
        search_queries_used: list[str] = []
        last_round_had_search: bool = False
        total_tool_calls: int = 0

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
        response = None

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

            prepared_tools = self.prepare_tools(
                tools, search_call_count, MAX_SEARCH_ROUNDS, last_round_had_search,
            )

            with cli_adapter.console.status("[bold cyan]⏳ 正在思考...[/bold cyan]", spinner="dots"):
                trace = self.tracer.get_trace(current_trace_id()) if current_trace_id() else None
                span_ctx = None
                if trace is not None:
                    span_ctx = trace.span("llm.chat_with_tools", attributes={"round": _round + 1})
                    span_ctx.__enter__()

                round_deadline = _time.monotonic() + ROUND_TIMEOUT

                try:
                    response = await asyncio.wait_for(
                        self.llm.chat_with_tools(
                            llm_messages, prepared_tools,
                            purpose="chat", deadline=round_deadline,
                        ),
                        timeout=ROUND_TIMEOUT,
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
                        fallback_msg = self.generate_timeout_reply(history_msgs)
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
                parallel_calls: list[tuple[int, dict[str, Any]]] = []
                serial_calls: list[tuple[int, dict[str, Any]]] = []
                all_tool_calls: list[dict[str, Any]] = []
                intercepted_ids: set[str] = set()
                intercept_reasons: dict[str, str] = {}

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

                        constrained, constraint_msg = self.check_search_constraint(
                            tool_name, search_call_count, MAX_SEARCH_ROUNDS, last_round_had_search,
                        )
                        if constrained:
                            if search_call_count >= MAX_SEARCH_ROUNDS:
                                logger.warning(
                                    "搜索次数已达上限 (%d/%d)，拦截搜索: %s",
                                    search_call_count, MAX_SEARCH_ROUNDS, query,
                                )
                            else:
                                logger.debug(
                                    "ReAct 约束：连续搜索被拦截: %s (上一轮已搜索，需先分析结果)",
                                    query,
                                )
                            intercept_reasons[tc["id"]] = constraint_msg
                            intercepted_ids.add(tc["id"])
                            continue

                        is_duplicate = False
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

                        if not is_duplicate and tc["id"] not in intercepted_ids:
                            search_call_count += 1
                            search_queries_used.append(query)
                            logger.info(
                                "允许搜索 #%d: %s", search_call_count, query,
                            )

                    if budget_remaining <= 0:
                        cli_adapter.console.print("[yellow]⚠ 预算已耗尽，停止工具调用[/yellow]")
                        intercepted_ids.add(tc["id"])
                        intercept_reasons[tc["id"]] = "[预算耗尽] 已达到最大工具调用轮次"
                        continue

                    if tool_name in _IRREVERSIBLE_TOOLS:
                        allowed, reason = self.check_hitl_confirmation(
                            tool_name, arguments, cli_adapter,
                        )
                        if not allowed:
                            intercepted_ids.add(tc["id"])
                            intercept_reasons[tc["id"]] = reason
                            continue

                    if tool_name not in ("tavily_search",) or tc["id"] not in intercepted_ids:
                        cli_adapter.console.print(f"[dim]🔧 {tool_name}({arguments})[/dim]")

                    if tool_name in _PARALLEL_SAFE_TOOLS and tool_name not in _IRREVERSIBLE_TOOLS and budget_remaining > 0:
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

                active_parallel = [
                    tc for _, tc in parallel_calls if tc["id"] not in intercepted_ids
                ]
                if len(active_parallel) > 1:
                    with cli_adapter.console.status("[bold yellow]🔧 并行执行工具...[/bold yellow]", spinner="line"):
                        results = await self.execute_tools_parallel(
                            active_parallel,
                            self.tool_manager.execute_tool,
                        )
                    for tc_id, tool_result in results:
                        tool_result, is_error = self.handle_tool_result(
                            "", tool_result, MAX_TOOL_RESULT_LEN,
                        )
                        budget_remaining -= 1
                        total_tool_calls += 1
                        history_msgs.append({
                            "role": "tool",
                            "content": tool_result,
                            "tool_call_id": tc_id,
                        })
                else:
                    for tc in active_parallel:
                        with cli_adapter.console.status(f"[bold yellow]🔧 执行 {tc['name']}...[/bold yellow]", spinner="line"):
                            tool_result = await self.tool_manager.execute_tool(tc["name"], tc["arguments"])

                        tool_result, is_error = self.handle_tool_result(
                            tc["name"], tool_result, MAX_TOOL_RESULT_LEN,
                        )
                        budget_remaining -= 1
                        total_tool_calls += 1
                        history_msgs.append({
                            "role": "tool",
                            "content": tool_result,
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

                    if tool_name in _IRREVERSIBLE_TOOLS:
                        allowed, reason = self.check_hitl_confirmation(
                            tool_name, arguments, cli_adapter,
                        )
                        if not allowed:
                            history_msgs.append({
                                "role": "tool",
                                "content": reason,
                                "tool_call_id": tc["id"],
                            })
                            continue

                    with cli_adapter.console.status(f"[bold yellow]🔧 执行 {tool_name}...[/bold yellow]", spinner="line"):
                        serial_results = await self.execute_tools_serial(
                            [tc], self.tool_manager.execute_tool,
                        )
                    for tc_id, tool_result in serial_results:
                        tool_result, is_error = self.handle_tool_result(
                            tool_name, tool_result, MAX_TOOL_RESULT_LEN,
                        )
                        if is_error:
                            cli_adapter.console.print(f"[dim]  ❌ {tool_name}: {tool_result[:80]}[/dim]")
                        else:
                            result_preview = tool_result[:60].replace("\n", " ") + ("..." if len(tool_result) > 60 else "")
                            cli_adapter.console.print(f"[dim]  ✅ {tool_name}: {result_preview}[/dim]")
                        budget_remaining -= 1
                        total_tool_calls += 1
                        history_msgs.append({
                            "role": "tool",
                            "content": tool_result,
                            "tool_call_id": tc_id,
                        })

                has_code_action = any(
                    tc["name"] in _CODE_TOOLS and tc["id"] not in intercepted_ids
                    for tc in response.tool_calls
                )
                if has_code_action:
                    last_round_had_search = False
                elif current_round_has_search:
                    last_round_had_search = True

                if total_tool_calls >= _HITL_CHECKPOINT_THRESHOLD:
                    logger.info(
                        "[HITL 检查点] 已执行 %d 次工具调用，继续执行中...",
                        total_tool_calls,
                    )
            else:
                cli_adapter.console.print()

                response_text = response.content or ""

                self._record_llm_stats(response)
                self._check_output_safety(response_text)

                user_msg = ""
                for m in reversed(history_msgs):
                    if m.get("role") == "user" and m.get("content"):
                        user_msg = m["content"]
                        break

                needs_code = any(p in user_msg for p in _CODE_TASK_PATTERNS)
                has_tool_history = any(m.get("role") == "tool" for m in history_msgs)

                has_code_tools = any(
                    m.get("role") == "tool"
                    and any(
                        kw in (m.get("content", "")[:100])
                        for kw in ("✅", "成功", "文件已保存", "执行完成")
                    )
                    for m in history_msgs
                )
                has_code_exec = any(
                    m.get("role") == "tool"
                    and any(
                        kw in (m.get("content", "")[:200])
                        for kw in ("执行完成", "执行成功", "execute_code", "execute_file")
                    )
                    for m in history_msgs
                )

                if needs_code and not has_code_tools and _round < MAX_TOOL_ROUNDS - 1:
                    if not has_tool_history:
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

                fabricated = (
                    not has_code_exec
                    and response_text
                    and any(p in response_text for p in _FABRICATED_RESULT_PATTERNS)
                    and "```" in response_text
                    and ("python" in response_text.lower() or "def " in response_text or "import " in response_text)
                )
                if fabricated:
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

        if response is not None:
            response_text = response.content or ""
            self._record_llm_stats(response)
            self._check_output_safety(response_text)
            self.active_session.add_message("assistant", response_text)
            cli_adapter.console.print()
            cli_adapter.console.print(response_text)
            cli_adapter.console.print()
            self._save_session()
            self._schedule_auto_eval()
