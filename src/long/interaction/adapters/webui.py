"""WebUI 适配器

实现 InteractionProtocol 接口，提供基于 FastAPI + WebSocket 的 Web 交互体验。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..base import HITLRequest, HITLResponse, InteractionEvent, InteractionEventType, InteractionProtocol

logger = logging.getLogger(__name__)


class WebSocketConnection:
    """管理单个 WebSocket 连接"""

    def __init__(self, websocket: WebSocket, session_id: str) -> None:
        self.websocket = websocket
        self.session_id = session_id
        self._connected = False

    async def connect(self) -> None:
        """接受 WebSocket 连接"""
        await self.websocket.accept()
        self._connected = True

    async def send_json(self, data: dict[str, Any]) -> None:
        """发送 JSON 数据"""
        if self._connected:
            try:
                await self.websocket.send_json(data)
            except Exception as e:
                logger.error("WebSocket send error: %s", e)
                self._connected = False

    async def receive_json(self) -> dict[str, Any] | None:
        """接收 JSON 数据"""
        if not self._connected:
            return None
        try:
            return await self.websocket.receive_json()
        except WebSocketDisconnect:
            self._connected = False
            return None
        except Exception as e:
            logger.error("WebSocket receive error: %s", e)
            self._connected = False
            return None

    def disconnect(self) -> None:
        """标记连接断开"""
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected


class WebUIAdapter(InteractionProtocol):
    """WebUI 适配器

    通过 FastAPI + WebSocket 实现浏览器交互。
    支持:
    - WebSocket 双向通信
    - SSE 流式输出 (通过 WebSocket 发送)
    - HITL 审批回调
    - REST API 备用

    出站消息通过线程安全队列传递，send_event 将消息入队，
    WebSocket 处理循环从队列中读取并推送。
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        static_dir: str | Path | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.static_dir = Path(static_dir) if static_dir else Path(__file__).parent.parent.parent.parent.parent / "static"

        self.app = FastAPI(title="Long - 可控AI智能系统", version="0.1.0")
        self._connections: dict[str, WebSocketConnection] = {}
        self._hitl_requests: dict[str, HITLRequest] = {}
        self._hitl_responses: dict[str, HITLResponse] = {}
        self._hitl_events: dict[str, asyncio.Event] = {}
        # 每个连接独立的出站队列，实现会话隔离
        self._outbound_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._inbound_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()  # (session_id, content)
        self._active_session_id: str | None = None  # 当前活跃会话（最后发消息的）
        self._active = False
        self._server_task: asyncio.Task[Any] | None = None
        self._skill_manager: Any = None
        self._mcp_client: Any = None
        self._tracer: Any = None  # 由 LongSystem 注入

        # 兼容 CLIAdapter 的 console 接口
        self.console = _WebUIConsole(self)

        # 兼容 CLIAdapter 的命令系统
        self._commands: dict[str, Any] = {}

        self._setup_routes()

    @property
    def active_connections(self) -> int:
        """当前活跃连接数"""
        return sum(1 for conn in self._connections.values() if conn.connected)

    def set_skill_manager(self, manager: Any) -> None:
        """设置 Skill 管理器"""
        self._skill_manager = manager

    def set_mcp_client(self, client: Any) -> None:
        """设置 MCP Client"""
        self._mcp_client = client

    def _setup_routes(self) -> None:
        """设置 FastAPI 路由"""

        @self.app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            """SPA 入口"""
            index_file = self.static_dir / "index.html"
            if index_file.exists():
                return HTMLResponse(index_file.read_text(encoding="utf-8"))
            return HTMLResponse("<h1>Long WebUI</h1><p>Static files not found</p>")

        @self.app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
            """WebSocket 连接端点"""
            conn = WebSocketConnection(websocket, session_id)
            await conn.connect()
            self._connections[session_id] = conn
            # 为每个连接创建独立的出站队列
            self._outbound_queues[session_id] = asyncio.Queue()
            logger.info("WebSocket connected: %s", session_id)

            try:
                recv_task = asyncio.create_task(
                    self._ws_recv_loop(conn)
                )
                send_task = asyncio.create_task(
                    self._ws_send_loop(conn)
                )

                done, pending = await asyncio.wait(
                    [recv_task, send_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            except WebSocketDisconnect:
                pass
            finally:
                conn.disconnect()
                self._connections.pop(session_id, None)
                self._outbound_queues.pop(session_id, None)
                logger.info("WebSocket disconnected: %s", session_id)

        @self.app.post("/api/chat/{session_id}")
        async def chat(session_id: str, message: dict[str, Any]) -> dict[str, Any]:
            """REST API 聊天端点"""
            content = message.get("content", "")
            return {"session_id": session_id, "status": "received", "content": content}

        @self.app.get("/api/traces")
        async def get_traces(limit: int = 20) -> dict[str, Any]:
            """获取追踪历史"""
            if self._tracer is None:
                return {"traces": []}
            traces = self._tracer.get_traces(limit=limit)
            return {"traces": [t.to_dict() for t in traces]}

        @self.app.get("/api/traces/{trace_id}")
        async def get_trace(trace_id: str) -> dict[str, Any]:
            """获取单个追踪详情"""
            if self._tracer is None:
                return {"trace": None}
            trace = self._tracer.get_trace(trace_id)
            return {"trace": trace.to_dict() if trace else None}

        @self.app.post("/api/hitl/{request_id}")
        async def hitl_callback(request_id: str, response: dict[str, Any]) -> dict[str, Any]:
            """HITL 审批回调"""
            decision = response.get("decision", "reject")
            feedback = response.get("feedback")

            hitl_response = HITLResponse(
                request_id=request_id,
                decision=decision,
                feedback=feedback,
            )
            self._hitl_responses[request_id] = hitl_response

            if request_id in self._hitl_events:
                self._hitl_events[request_id].set()

            return {"status": "ok", "request_id": request_id}

        @self.app.get("/api/skills")
        async def list_skills() -> dict[str, Any]:
            """Skill 管理 API"""
            skills: list[dict[str, Any]] = []
            if self._skill_manager is not None:
                for skill in self._skill_manager.list_skills():
                    skills.append({
                        "name": skill.manifest.name,
                        "version": skill.manifest.version,
                        "description": skill.manifest.description,
                        "state": skill.state.value,
                        "permissions": skill.manifest.permissions,
                        "tools": skill.manifest.tools,
                    })
            return {"skills": skills}

        @self.app.get("/api/mcp")
        async def list_mcp_servers() -> dict[str, Any]:
            """MCP 管理 API"""
            servers: list[dict[str, Any]] = []
            if self._mcp_client is not None:
                for name, config in self._mcp_client._servers.items():
                    servers.append({
                        "name": name,
                        "transport": config.transport,
                        "command": config.command,
                    })
            return {"servers": servers}

        # 挂载静态文件
        if self.static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(self.static_dir)), name="static")

        # 挂载 output 目录，让生成的报告和图表可通过 HTTP 访问
        output_dir = Path("workspace/output")
        if not output_dir.exists():
            output_dir = Path("output")
        if output_dir.exists():
            self.app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")

    async def _ws_recv_loop(self, conn: WebSocketConnection) -> None:
        """WebSocket 入站消息循环"""
        while conn.connected:
            data = await conn.receive_json()
            if data is None:
                break
            await self._handle_ws_message(conn.session_id, data)

    async def _ws_send_loop(self, conn: WebSocketConnection) -> None:
        """WebSocket 出站消息循环

        从该连接独立的出站队列中读取消息并发送。
        """
        queue = self._outbound_queues.get(conn.session_id)
        if not queue:
            return
        while conn.connected:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            await conn.send_json(data)

    async def _handle_ws_message(self, session_id: str, data: dict[str, Any]) -> None:
        """处理 WebSocket 消息"""
        msg_type = data.get("type", "")

        if msg_type == "hitl_response":
            request_id = data.get("request_id", "")
            decision = data.get("decision", "reject")
            feedback = data.get("feedback")

            hitl_response = HITLResponse(
                request_id=request_id,
                decision=decision,
                feedback=feedback,
            )
            self._hitl_responses[request_id] = hitl_response

            if request_id in self._hitl_events:
                self._hitl_events[request_id].set()

        elif msg_type == "user_input":
            content = data.get("content", "")
            if content:
                # 记录活跃会话，后续回复只发给此会话
                self._active_session_id = session_id
                try:
                    self._inbound_queue.put_nowait((session_id, content))
                except asyncio.QueueFull:
                    logger.warning("Inbound queue full, dropping user input")

    # --- InteractionProtocol 实现 ---

    def start_session(self) -> None:
        """启动 WebUI 服务"""
        self._active = True

    def end_session(self) -> None:
        """结束 WebUI 服务"""
        self._active = False
        for conn in self._connections.values():
            conn.disconnect()
        self._connections.clear()
        self._outbound_queues.clear()

    def send_event(self, event: InteractionEvent) -> None:
        """通过 WebSocket 发送事件到活跃会话

        消息只发送给当前活跃会话（最后发送用户输入的连接），
        不会广播到其他页面。
        """
        if not self._active:
            return

        data = {
            "type": event.type.value,
            "content": event.content,
            "metadata": event.metadata,
            "timestamp": event.timestamp.isoformat(),
        }

        self._enqueue_outbound(data)

    def _enqueue_outbound(self, data: dict[str, Any]) -> None:
        """将消息加入活跃会话的出站队列

        只发给 _active_session_id 对应的连接，不广播。
        """
        sid = self._active_session_id
        if not sid:
            # 没有活跃会话时，广播到所有连接（兼容初始化阶段）
            for q in self._outbound_queues.values():
                try:
                    q.put_nowait(data)
                except asyncio.QueueFull:
                    pass
            return

        queue = self._outbound_queues.get(sid)
        if queue:
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                logger.warning("Outbound queue full for session %s, dropping message", sid)

    def receive_input(self, prompt: str = "") -> str:
        """接收用户输入（从 WebSocket 入站队列获取）"""
        try:
            _sid, content = self._inbound_queue.get_nowait()
            return content
        except asyncio.QueueEmpty:
            return ""

    async def receive_input_async(self, timeout: float = 300.0) -> tuple[str, str]:
        """异步接收用户输入（等待 WebSocket 消息）

        Returns:
            (session_id, content) 元组，超时返回 ("", "")
        """
        try:
            sid, content = await asyncio.wait_for(self._inbound_queue.get(), timeout=timeout)
            return sid, content
        except asyncio.TimeoutError:
            return "", ""

    # --- 命令系统（兼容 CLIAdapter） ---

    def register_command(self, name: str, handler: Any) -> None:
        """注册命令处理器"""
        self._commands[name] = handler

    def parse_command(self, text: str) -> tuple[Any, str]:
        """解析命令，返回 (cmd, args) 或 (None, text)"""
        text = text.strip()
        if text.startswith("/"):
            parts = text.split(None, 1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            if cmd in self._commands:
                return cmd, args
        return None, text

    def handle_command(self, cmd: str, args: str) -> Any:
        """执行命令"""
        handler = self._commands.get(cmd)
        if handler:
            return handler(args)
        self.console.print(f"未知命令: {cmd}")
        return None

    def stream_output(self, tokens: Any) -> None:
        """流式输出（通过 WebSocket 发送）"""
        event = InteractionEvent(
            type=InteractionEventType.STREAM_TOKEN,
            content=tokens if isinstance(tokens, str) else str(tokens),
        )
        self.send_event(event)

    def request_feedback(self, hitl_request: HITLRequest) -> HITLResponse:
        """HITL 人工审核请求（同步方式）"""
        self._hitl_requests[hitl_request.request_id] = hitl_request

        event = InteractionEvent(
            type=InteractionEventType.HITL_REQUEST,
            content=hitl_request.title,
            metadata={
                "request_id": hitl_request.request_id,
                "description": hitl_request.description,
                "risk_level": hitl_request.risk_level,
                "options": hitl_request.options,
            },
        )
        self.send_event(event)

        # 尝试获取已有响应（非阻塞）
        if hitl_request.request_id in self._hitl_responses:
            return self._hitl_responses.pop(hitl_request.request_id)

        return HITLResponse(
            request_id=hitl_request.request_id,
            decision="pending",
        )

    async def request_feedback_async(self, hitl_request: HITLRequest) -> HITLResponse:
        """HITL 人工审核请求（异步方式 - 等待 WebUI 回调）"""
        self._hitl_requests[hitl_request.request_id] = hitl_request

        event = InteractionEvent(
            type=InteractionEventType.HITL_REQUEST,
            content=hitl_request.title,
            metadata={
                "request_id": hitl_request.request_id,
                "description": hitl_request.description,
                "risk_level": hitl_request.risk_level,
                "options": hitl_request.options,
            },
        )
        self.send_event(event)

        wait_event = asyncio.Event()
        self._hitl_events[hitl_request.request_id] = wait_event

        try:
            await asyncio.wait_for(wait_event.wait(), timeout=hitl_request.timeout_seconds)
        except asyncio.TimeoutError:
            return HITLResponse(
                request_id=hitl_request.request_id,
                decision="timeout",
            )
        finally:
            self._hitl_events.pop(hitl_request.request_id, None)

        return self._hitl_responses.pop(hitl_request.request_id, HITLResponse(
            request_id=hitl_request.request_id,
            decision="no_response",
        ))

    # --- 服务管理 ---

    def _is_port_in_use(self, port: int) -> bool:
        """检查端口是否被占用"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((self.host, port))
                return False
            except OSError:
                return True

    async def start_server(self) -> None:
        """启动 WebUI 服务器并等待其运行

        如果指定端口被占用，自动尝试后续端口（最多尝试10个）。
        """
        import uvicorn

        port = self.port
        max_tries = 10
        for attempt in range(max_tries):
            if not self._is_port_in_use(port):
                break
            logger.warning("端口 %d 已被占用，尝试 %d...", port, port + 1)
            port += 1
        else:
            raise RuntimeError(f"端口 {self.port}-{port} 均被占用，无法启动 WebUI")

        self.port = port

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        self._active = True
        await self._server.serve()

    async def stop_server(self) -> None:
        """停止 WebUI 服务器"""
        self._active = False
        if hasattr(self, '_server') and self._server is not None:
            self._server.should_exit = True


class _WebUIConsole:
    """兼容 CLIAdapter.console 接口的适配器

    将 rich Console 的 print 调用转换为 WebSocket 事件发送到前端。
    使用 rich 的 Console(render()) 方法将所有 rich 对象（Panel、Table、
    Markdown 等）渲染为纯文本，避免出现 <rich.panel.Panel object at 0x...>。
    """

    def __init__(self, adapter: WebUIAdapter) -> None:
        self._adapter = adapter
        # 用一个隐藏的 rich Console 来渲染 rich 对象为纯文本
        self._rich_console: Any = None
        # 追踪是否处于流式输出中，用于在流结束时发送 stream_end 事件
        self._streaming_active = False

    def _get_rich_console(self) -> Any:
        """延迟初始化 rich Console（用于渲染 rich 对象为纯文本）"""
        if self._rich_console is None:
            try:
                from rich.console import Console
                import io
                self._rich_console = Console(
                    file=io.StringIO(),
                    force_terminal=False,
                    no_color=True,
                    width=120,
                    legacy_windows=False,
                )
            except ImportError:
                self._rich_console = False
        return self._rich_console if self._rich_console is not False else None

    def _render_to_text(self, obj: Any) -> str:
        """将任意对象（包括 rich 对象和带标签字符串）渲染为纯文本"""
        # 快速路径：纯字符串直接返回（流式 token 大多是纯字符串）
        if isinstance(obj, str):
            # 检查是否包含 rich 标签（方括号标签）
            if '[' not in obj or ']' not in obj:
                return obj
            return self._strip_rich_tags(obj)

        # 尝试用 rich Console 渲染（能正确处理所有 rich 对象和标签）
        console = self._get_rich_console()
        if console is not None:
            try:
                import io
                console.file = io.StringIO()
                console.print(obj, crop=False, overflow="ignore", highlight=False, style=None)
                text = console.file.getvalue().strip()
                return text
            except Exception:
                pass

        # 回退：手动处理
        if hasattr(obj, 'plain'):
            return obj.plain

        return str(obj)

    @staticmethod
    def _strip_rich_tags(text: str) -> str:
        """去掉 rich 的方括号标签如 [green]、[/bold cyan] 等"""
        import re
        # 匹配 [style] 和 [/style] 标签，包括复合标签如 [bold cyan]
        # 不匹配 markdown 链接 [text](url)
        text = re.sub(r'\[/?(?:[a-zA-Z_][\w\s]*)(?:=[^\]]*)?\]', '', text)
        # 清理多余空白
        text = re.sub(r'  +', ' ', text)
        return text.strip()

    # WebUI 模式下需要过滤的内部调试消息模式
    # 只过滤纯状态指示消息，不过滤包含实际内容的消息
    _FILTER_PATTERNS = (
        "━━━",           # Round 分隔线
        "LLM 思考中",     # 思考状态
        "⏳ 正在处理",    # 处理状态
        "🤔 LLM 思考中",  # 思考状态（带 emoji）
        "🔧 Round",       # 工具调用轮次前缀
        "⏳ 正在思考",    # 思考状态
        "🔧 并行执行",    # 工具执行状态
        "🔧 执行",        # 工具执行状态
        "✨ 正在生成回复", # 生成状态
        "📋 正在生成执行计划", # 计划生成状态
    )

    def print(self, *args: Any, **kwargs: Any) -> None:
        """将 print 内容通过 WebSocket 发送，支持流式输出

        关键参数：
        - end: 默认 "\\n"，流式输出时传入 end="" 实现逐 token 输出
        - highlight: rich 的语法高亮参数（WebUI 忽略）
        """
        end_char = kwargs.get("end", "\n")

        # 快速路径：流式 token（单个纯字符串，无 rich 标签）
        if end_char == "" and len(args) == 1 and isinstance(args[0], str) and '[' not in args[0]:
            token = args[0]
            # 过滤 CLI 状态指示
            if token.strip() and any(token.strip().startswith(pat) or token.strip() == pat.strip() for pat in self._FILTER_PATTERNS):
                return
            self._streaming_active = True
            self._adapter.send_event(InteractionEvent(
                type=InteractionEventType.STREAM_TOKEN,
                content=token,
            ))
            return

        text_parts: list[str] = []
        for arg in args:
            text = self._render_to_text(arg)
            text_parts.append(text)

        content = "".join(text_parts) + end_char

        # 过滤 CLI 模式的纯状态指示消息
        first_line = content.split("\n")[0].strip()
        if any(first_line.startswith(pat) or first_line == pat.strip() for pat in self._FILTER_PATTERNS):
            return
        if first_line.startswith("LLM 思考完成") and "→" not in content:
            return

        if end_char == "":
            # 流式 token — 逐 token 输出（不过滤空白 token，空格和换行对格式化重要）
            self._streaming_active = True
            self._adapter.send_event(InteractionEvent(
                type=InteractionEventType.STREAM_TOKEN,
                content=content,
            ))
        else:
            # 完整消息（end="\n"）
            # 如果之前在流式输出中，标记流结束（不再发 STREAM_END 事件，
            # 因为后续可能还有 MESSAGE 事件，前端通过防抖判定对话结束）
            if self._streaming_active:
                self._streaming_active = False
            # 只发送有实际内容的完整消息（过滤纯空行）
            if content.strip():
                self._adapter.send_event(InteractionEvent(
                    type=InteractionEventType.MESSAGE,
                    content=content.rstrip("\n"),
                ))

    def rule(self, title: str = "", **kwargs: Any) -> None:
        """发送分隔线"""
        self._adapter.send_event(InteractionEvent(
            type=InteractionEventType.STREAM_TOKEN,
            content=f"\n{'─' * 40}\n{title}\n{'─' * 40}\n" if title else f"\n{'─' * 40}\n",
        ))

    def status(self, status_text: str = "", **kwargs: Any) -> "_WebUIStatusContext":
        """兼容 rich Console.status() 上下文管理器"""
        return _WebUIStatusContext(self, status_text)

    def log(self, *args: Any, **kwargs: Any) -> None:
        """兼容 rich Console.log()"""
        self.print(*args, **kwargs)


class _WebUIStatusContext:
    """兼容 rich Console.status() 的上下文管理器

    在 CLI 模式下显示 spinner，在 WebUI 模式下不发送任何消息到前端。
    这些是内部处理状态指示，不应作为消息气泡显示给用户。
    前端通过 isProcessing 状态自行管理处理中的 UI 表现。
    """

    def __init__(self, console: "_WebUIConsole", status_text: str) -> None:
        self._console = console
        self._status_text = console._strip_rich_tags(status_text)
        self._active = False

    def __enter__(self) -> "_WebUIStatusContext":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def start(self) -> None:
        """开始显示状态 — WebUI 模式下静默，不发送消息"""
        self._active = True

    def stop(self) -> None:
        """停止显示状态"""
        self._active = False

    def update(self, text: str = "", **kwargs: Any) -> None:
        """更新状态文本 — WebUI 模式下静默"""
        if text:
            clean = self._console._strip_rich_tags(text)
            if clean:
                self._status_text = clean
