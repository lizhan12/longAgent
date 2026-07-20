"""CLI 适配器

实现 InteractionProtocol 接口，提供基于终端的交互体验。
使用 prompt_toolkit 处理输入，Rich 格式化输出。
"""

from __future__ import annotations

import shlex
import sys
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..base import HITLRequest, HITLResponse, InteractionEvent, InteractionEventType, InteractionProtocol
from ..streaming import StreamBuffer, StreamChunk


class CLIAdapter(InteractionProtocol):
    """CLI 适配器

    使用 Rich 和 prompt_toolkit 提供终端交互。

    Attributes:
        console: Rich 控制台
        prompt_session: prompt_toolkit 会话
    """

    COMMANDS = {
        "/help": "显示帮助信息",
        "/status": "显示当前状态",
        "/skill": "Skill 管理",
        "/mcp": "MCP 服务器管理",
        "/history": "查看历史记录",
        "/exit": "退出",
        "/quit": "退出",
        "/clear": "清屏",
    }

    def __init__(
        self,
        *,
        console: Console | None = None,
        prompt_history_size: int = 1000,
    ) -> None:
        self.console = console or Console()

        command_completer = WordCompleter(
            list(self.COMMANDS.keys()),
            ignore_case=True,
        )

        self.prompt_session: PromptSession[str] = PromptSession(
            history=InMemoryHistory(),
            auto_suggest=AutoSuggestFromHistory(),
            completer=command_completer,
        )

        self._active = False
        self._stream_buffer = ""
        self._command_handlers: dict[str, Any] = {}

    def start_session(self) -> None:
        """启动 CLI 会话"""
        self._active = True
        self.console.print(
            Panel(
                "[bold green]Long - 可控AI智能系统[/bold green]\n"
                "输入 /help 查看可用命令",
                title="Welcome",
                border_style="green",
            )
        )

    def end_session(self) -> None:
        """结束 CLI 会话"""
        self._active = False
        self.console.print("[dim]会话已结束[/dim]")

    def send_event(self, event: InteractionEvent) -> None:
        """输出事件到终端"""
        if not self._active:
            return

        if event.type == InteractionEventType.MESSAGE:
            self._render_message(event.content, event.metadata)
        elif event.type == InteractionEventType.ERROR:
            self.console.print(f"[bold red]✗ 错误:[/bold red] {event.content}")
        elif event.type == InteractionEventType.WARNING:
            self.console.print(f"[bold yellow]⚠ 警告:[/bold yellow] {event.content}")
        elif event.type == InteractionEventType.INFO:
            self.console.print(f"[bold blue]ℹ[/bold blue] {event.content}")
        elif event.type == InteractionEventType.PROGRESS:
            self._render_progress(event.content, event.metadata)
        elif event.type == InteractionEventType.STREAM_TOKEN:
            self._render_stream_token(event.content)
        elif event.type == InteractionEventType.STREAM_END:
            self._flush_stream()
        elif event.type == InteractionEventType.SYSTEM:
            self.console.print(f"[dim]{event.content}[/dim]")

    def receive_input(self, prompt: str = "> ") -> str:
        """从终端接收用户输入（同步版本，不能在 async 上下文中使用）"""
        try:
            text = self.prompt_session.prompt(prompt)
            return text.strip()
        except (EOFError, KeyboardInterrupt):
            return "/exit"

    async def receive_input_async(self, prompt: str = "> ") -> str:
        """从终端接收用户输入（异步版本，可在 async 上下文中使用）"""
        try:
            text = await self.prompt_session.prompt_async(prompt)
            return text.strip()
        except (EOFError, KeyboardInterrupt):
            return "/exit"

    def stream_output(self, tokens: Any) -> None:
        """流式输出 token"""
        if isinstance(tokens, str):
            self.console.print(tokens, end="")
        elif hasattr(tokens, "__iter__"):
            for token in tokens:
                self.console.print(token, end="")
        self.console.print()

    def request_feedback(self, hitl_request: HITLRequest) -> HITLResponse:
        """在终端中显示 HITL 请求并等待用户输入"""
        risk_style = {
            "low": "green",
            "medium": "yellow",
            "high": "red",
            "critical": "bold red",
        }.get(hitl_request.risk_level, "yellow")

        self.console.print()
        self.console.print(
            Panel(
                hitl_request.description,
                title=f"[{risk_style}]审核请求: {hitl_request.title}[/{risk_style}]",
                border_style=risk_style,
            )
        )

        options_str = " / ".join(
            f"[bold]{opt}[/bold]" for opt in hitl_request.options
        )
        self.console.print(f"选项: {options_str}")

        while True:
            response = self.prompt_session.prompt(
                "请选择: ",
            ).strip().lower()

            if response in [opt.lower() for opt in hitl_request.options]:
                from datetime import datetime
                return HITLResponse(
                    request_id=hitl_request.request_id,
                    decision=response,
                    timestamp=datetime.now(),
                )

            self.console.print(
                f"[yellow]无效选择，请输入: {', '.join(hitl_request.options)}[/yellow]"
            )

    # --- 命令解析 ---

    def parse_command(self, text: str) -> tuple[str | None, list[str]]:
        """解析命令

        Returns:
            (命令名, 参数列表) 或 (None, []) 如果不是命令
        """
        if not text.startswith("/"):
            return None, []

        parts = shlex.split(text)
        command = parts[0].lower()
        args = parts[1:]

        return command, args

    def handle_command(self, command: str, args: list[str]) -> str | None:
        """处理命令

        Returns:
            命令处理结果，或 None 表示未知命令
        """
        if command in ("/exit", "/quit"):
            return "__exit__"
        elif command == "/help":
            self._show_help()
            return None
        elif command == "/clear":
            self.console.clear()
            return None
        elif command in self._command_handlers:
            result = self._command_handlers[command](*args)
            if result is not None:
                return result
            return None
        elif command == "/status":
            self._show_status()
            return None
        else:
            self.console.print(f"[yellow]未知命令: {command}[/yellow]")
            self._show_help()
            return None

    def register_command(self, command: str, handler: Any) -> None:
        """注册命令处理器"""
        self._command_handlers[command] = handler

    # --- 内部渲染方法 ---

    def _render_message(self, content: str, metadata: dict[str, Any]) -> None:
        """渲染消息"""
        format_type = metadata.get("format", "markdown")

        if format_type == "code":
            language = metadata.get("language", "python")
            syntax = Syntax(content, language, theme="monokai", line_numbers=True)
            self.console.print(syntax)
        elif format_type == "table" and "headers" in metadata and "rows" in metadata:
            table = Table(show_header=True, header_style="bold")
            for header in metadata["headers"]:
                table.add_column(header)
            for row in metadata["rows"]:
                table.add_row(*[str(c) for c in row])
            self.console.print(table)
        elif format_type == "plain":
            self.console.print(content)
        else:
            md = Markdown(content)
            self.console.print(md)

    def _render_progress(self, content: str, metadata: dict[str, Any]) -> None:
        """渲染进度信息"""
        if "percent" in metadata:
            pct = metadata["percent"]
            bar_width = 30
            filled = int(bar_width * pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            self.console.print(f"[blue]{bar}[/blue] {pct}% {content}")
        else:
            self.console.print(f"[blue]→[/blue] {content}")

    def _render_stream_token(self, token: str) -> None:
        """渲染流式 token"""
        self._stream_buffer += token
        self.console.print(token, end="")

    def _flush_stream(self) -> None:
        """刷新流式输出缓冲"""
        if self._stream_buffer:
            self.console.print()
            self._stream_buffer = ""

    def _show_help(self) -> None:
        """显示帮助信息"""
        table = Table(title="可用命令", show_header=True, header_style="bold")
        table.add_column("命令", style="cyan")
        table.add_column("说明")

        shown_commands: set[str] = set()
        for cmd, desc in self.COMMANDS.items():
            shown_commands.add(cmd)
            table.add_row(cmd, desc)

        for cmd, handler in self._command_handlers.items():
            if cmd not in shown_commands:
                table.add_row(cmd, getattr(handler, "__doc__", "自定义命令"))

        self.console.print(table)

    def _show_status(self) -> None:
        """显示状态"""
        self.console.print(
            Panel(
                f"会话状态: {'活跃' if self._active else '未启动'}\n"
                f"流式缓冲: {'有数据' if self._stream_buffer else '空'}",
                title="系统状态",
                border_style="blue",
            )
        )
