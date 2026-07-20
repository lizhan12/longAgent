"""Interaction 模块测试

覆盖交互协议、会话、流式输出、控制器和 CLI 适配器。
"""

from datetime import datetime

import pytest

from long.interaction.base import (
    HITLRequest,
    HITLResponse,
    InteractionEvent,
    InteractionEventType,
    InteractionProtocol,
)
from long.interaction.controller import InteractionController
from long.interaction.session import Session, SessionState
from long.interaction.streaming import StreamBuffer, StreamChunk, StreamingManager


# ========================
# InteractionEvent 测试
# ========================


class TestInteractionEvent:
    """交互事件测试"""

    def test_create_message_event(self):
        event = InteractionEvent(
            type=InteractionEventType.MESSAGE,
            content="Hello",
        )
        assert event.type == InteractionEventType.MESSAGE
        assert event.content == "Hello"
        assert event.metadata == {}

    def test_create_event_with_metadata(self):
        event = InteractionEvent(
            type=InteractionEventType.ERROR,
            content="Error occurred",
            metadata={"code": 500},
        )
        assert event.metadata["code"] == 500

    def test_event_types(self):
        expected_types = {
            "message", "error", "warning", "info", "progress",
            "stream_start", "stream_token", "stream_end",
            "hitl_request", "hitl_response", "system", "command",
        }
        actual_types = {t.value for t in InteractionEventType}
        assert actual_types == expected_types


# ========================
# Session 测试
# ========================


class TestSession:
    """会话测试"""

    def test_create_session(self):
        session = Session()
        assert session.session_id
        assert session.state == SessionState.CREATED
        assert session.is_active is False

    def test_activate_session(self):
        session = Session()
        activated = session.activate()
        assert activated.state == SessionState.ACTIVE
        assert activated.is_active is True

    def test_pause_session(self):
        session = Session()
        activated = session.activate()
        paused = activated.pause()
        assert paused.state == SessionState.PAUSED
        assert paused.is_active is False

    def test_end_session(self):
        session = Session()
        activated = session.activate()
        ended = activated.end()
        assert ended.state == SessionState.ENDED

    def test_session_immutability(self):
        """会话操作返回新对象，原对象不变"""
        session = Session()
        session.activate()
        assert session.state == SessionState.CREATED


# ========================
# StreamBuffer 测试
# ========================


class TestStreamBuffer:
    """流式输出缓冲区测试"""

    def test_start_and_write(self):
        buffer = StreamBuffer()
        buffer.start()
        assert buffer.active is True

        buffer.write(StreamChunk(content="Hello "))
        buffer.write(StreamChunk(content="World"))

        assert buffer.get_content() == "Hello World"
        assert buffer.total_tokens == 2

    def test_end_buffer(self):
        buffer = StreamBuffer()
        buffer.start()
        buffer.end()
        assert buffer.active is False

    def test_write_when_not_active(self):
        buffer = StreamBuffer()
        buffer.write(StreamChunk(content="test"))
        assert buffer.get_content() == ""

    def test_clear(self):
        buffer = StreamBuffer()
        buffer.start()
        buffer.write(StreamChunk(content="data"))
        buffer.clear()
        assert buffer.get_content() == ""
        assert buffer.total_tokens == 0

    def test_subscriber(self):
        buffer = StreamBuffer()
        received: list[StreamChunk] = []

        def callback(chunk: StreamChunk):
            received.append(chunk)

        buffer.subscribe(callback)
        buffer.start()
        buffer.write(StreamChunk(content="test"))

        assert len(received) == 1
        assert received[0].content == "test"

    def test_unsubscribe(self):
        buffer = StreamBuffer()
        received: list[StreamChunk] = []

        def callback(chunk: StreamChunk):
            received.append(chunk)

        buffer.subscribe(callback)
        buffer.unsubscribe(callback)
        buffer.start()
        buffer.write(StreamChunk(content="test"))

        assert len(received) == 0


# ========================
# StreamingManager 测试
# ========================


class TestStreamingManager:
    """流式输出管理器测试"""

    def test_create_stream(self):
        manager = StreamingManager()
        buffer = manager.create_stream("test")
        assert buffer is not None
        assert isinstance(buffer, StreamBuffer)

    def test_get_stream(self):
        manager = StreamingManager()
        manager.create_stream("test")
        buffer = manager.get_stream("test")
        assert buffer is not None

    def test_get_nonexistent_stream(self):
        manager = StreamingManager()
        assert manager.get_stream("nonexistent") is None

    def test_remove_stream(self):
        manager = StreamingManager()
        manager.create_stream("test")
        manager.remove_stream("test")
        assert manager.get_stream("test") is None

    def test_end_all(self):
        manager = StreamingManager()
        b1 = manager.create_stream("s1")
        b2 = manager.create_stream("s2")
        b1.start()
        b2.start()
        manager.end_all()
        assert b1.active is False
        assert b2.active is False


# ========================
# InteractionController 测试
# ========================


class MockProtocol(InteractionProtocol):
    """测试用 Mock 协议"""

    def __init__(self):
        self.events: list[InteractionEvent] = []
        self.inputs: list[str] = []
        self._input_index = 0
        self.session_started = False
        self.session_ended = False

    def send_event(self, event: InteractionEvent) -> None:
        self.events.append(event)

    def receive_input(self, prompt: str = "") -> str:
        if self._input_index < len(self.inputs):
            result = self.inputs[self._input_index]
            self._input_index += 1
            return result
        return ""

    def stream_output(self, tokens) -> None:
        pass

    def request_feedback(self, hitl_request: HITLRequest) -> HITLResponse:
        return HITLResponse(
            request_id=hitl_request.request_id,
            decision="approve",
        )

    def start_session(self) -> None:
        self.session_started = True

    def end_session(self) -> None:
        self.session_ended = True


class TestInteractionController:
    """交互控制器测试"""

    def test_create_session(self):
        ctrl = InteractionController()
        session = ctrl.create_session()
        assert session.state == SessionState.CREATED

    def test_activate_session(self):
        ctrl = InteractionController()
        session = ctrl.create_session()
        activated = ctrl.activate_session(session.session_id)
        assert activated is not None
        assert activated.is_active is True
        assert ctrl.active_session is not None

    def test_end_session(self):
        ctrl = InteractionController()
        session = ctrl.create_session()
        ctrl.activate_session(session.session_id)
        ended = ctrl.end_session(session.session_id)
        assert ended is not None
        assert ended.state == SessionState.ENDED
        assert ctrl.active_session is None

    def test_activate_session_calls_protocol(self):
        protocol = MockProtocol()
        ctrl = InteractionController(protocol=protocol)
        session = ctrl.create_session()
        ctrl.activate_session(session.session_id)
        assert protocol.session_started is True

    def test_end_session_calls_protocol(self):
        protocol = MockProtocol()
        ctrl = InteractionController(protocol=protocol)
        session = ctrl.create_session()
        ctrl.activate_session(session.session_id)
        ctrl.end_session(session.session_id)
        assert protocol.session_ended is True

    def test_publish_event(self):
        protocol = MockProtocol()
        ctrl = InteractionController(protocol=protocol)
        ctrl.publish(InteractionEvent(
            type=InteractionEventType.MESSAGE,
            content="test",
        ))
        assert len(protocol.events) == 1
        assert protocol.events[0].content == "test"

    def test_subscribe_event(self):
        ctrl = InteractionController()
        received: list[InteractionEvent] = []

        ctrl.subscribe(lambda e: received.append(e))

        ctrl.publish(InteractionEvent(
            type=InteractionEventType.MESSAGE,
            content="hello",
        ))

        assert len(received) == 1

    def test_subscribe_with_filter(self):
        ctrl = InteractionController()
        received: list[InteractionEvent] = []

        ctrl.subscribe(
            lambda e: received.append(e),
            event_types={InteractionEventType.ERROR},
        )

        ctrl.publish(InteractionEvent(type=InteractionEventType.MESSAGE, content="msg"))
        ctrl.publish(InteractionEvent(type=InteractionEventType.ERROR, content="err"))

        assert len(received) == 1
        assert received[0].content == "err"

    def test_send_message(self):
        protocol = MockProtocol()
        ctrl = InteractionController(protocol=protocol)
        ctrl.send_message("Hello World")
        assert protocol.events[-1].type == InteractionEventType.MESSAGE

    def test_send_error(self):
        protocol = MockProtocol()
        ctrl = InteractionController(protocol=protocol)
        ctrl.send_error("Something failed")
        assert protocol.events[-1].type == InteractionEventType.ERROR

    def test_get_input(self):
        protocol = MockProtocol()
        protocol.inputs = ["user input"]
        ctrl = InteractionController(protocol=protocol)
        result = ctrl.get_input()
        assert result == "user input"

    def test_request_feedback(self):
        protocol = MockProtocol()
        ctrl = InteractionController(protocol=protocol)
        request = HITLRequest(
            request_id="r1",
            title="Test",
            description="Please approve",
        )
        response = ctrl.request_feedback(request)
        assert response.decision == "approve"

    def test_start_and_end_stream(self):
        ctrl = InteractionController()
        session = ctrl.create_session()
        ctrl.activate_session(session.session_id)

        buffer = ctrl.start_stream()
        assert buffer.active is True

        buffer.write(StreamChunk(content="hello"))
        assert buffer.get_content() == "hello"

        ctrl.end_stream()
        assert buffer.active is False


# ========================
# CLIAdapter 测试
# ========================


class TestCLIAdapter:
    """CLI 适配器测试"""

    def test_cli_adapter_instantiation(self):
        from long.interaction.adapters.cli import CLIAdapter
        from rich.console import Console

        console = Console(width=80, force_terminal=True)
        adapter = CLIAdapter(console=console)
        assert adapter._active is False

    def test_start_session(self):
        from long.interaction.adapters.cli import CLIAdapter
        from io import StringIO
        from rich.console import Console

        console = Console(file=StringIO(), width=80, force_terminal=True)
        adapter = CLIAdapter(console=console)
        adapter.start_session()
        assert adapter._active is True

    def test_end_session(self):
        from long.interaction.adapters.cli import CLIAdapter
        from io import StringIO
        from rich.console import Console

        console = Console(file=StringIO(), width=80, force_terminal=True)
        adapter = CLIAdapter(console=console)
        adapter.start_session()
        adapter.end_session()
        assert adapter._active is False

    def test_send_event_message(self):
        from long.interaction.adapters.cli import CLIAdapter
        from io import StringIO
        from rich.console import Console

        output = StringIO()
        console = Console(file=output, width=80, force_terminal=True, no_color=True)
        adapter = CLIAdapter(console=console)
        adapter.start_session()

        adapter.send_event(InteractionEvent(
            type=InteractionEventType.MESSAGE,
            content="Test message",
            metadata={"format": "plain"},
        ))

        output_str = output.getvalue()
        assert "Test message" in output_str

    def test_send_event_error(self):
        from long.interaction.adapters.cli import CLIAdapter
        from io import StringIO
        from rich.console import Console

        output = StringIO()
        console = Console(file=output, width=80, force_terminal=True)
        adapter = CLIAdapter(console=console)
        adapter.start_session()

        adapter.send_event(InteractionEvent(
            type=InteractionEventType.ERROR,
            content="Error occurred",
        ))

        assert "Error" in output.getvalue()

    def test_send_event_not_active(self):
        from long.interaction.adapters.cli import CLIAdapter
        from io import StringIO
        from rich.console import Console

        output = StringIO()
        console = Console(file=output, width=80, force_terminal=True)
        adapter = CLIAdapter(console=console)
        # 未启动，发送事件应不输出
        adapter.send_event(InteractionEvent(
            type=InteractionEventType.MESSAGE,
            content="Should not appear",
        ))
        assert output.getvalue() == ""

    def test_parse_command(self):
        from long.interaction.adapters.cli import CLIAdapter
        from io import StringIO
        from rich.console import Console

        console = Console(file=StringIO(), width=80)
        adapter = CLIAdapter(console=console)

        cmd, args = adapter.parse_command("/help")
        assert cmd == "/help"
        assert args == []

        cmd, args = adapter.parse_command("/skill load my_skill")
        assert cmd == "/skill"
        assert args == ["load", "my_skill"]

        cmd, args = adapter.parse_command("hello world")
        assert cmd is None
        assert args == []

    def test_handle_command_help(self):
        from long.interaction.adapters.cli import CLIAdapter
        from io import StringIO
        from rich.console import Console

        output = StringIO()
        console = Console(file=output, width=80, force_terminal=True)
        adapter = CLIAdapter(console=console)

        result = adapter.handle_command("/help", [])
        assert result is None

    def test_handle_command_exit(self):
        from long.interaction.adapters.cli import CLIAdapter
        from io import StringIO
        from rich.console import Console

        console = Console(file=StringIO(), width=80)
        adapter = CLIAdapter(console=console)

        result = adapter.handle_command("/exit", [])
        assert result == "__exit__"

    def test_handle_command_clear(self):
        from long.interaction.adapters.cli import CLIAdapter
        from io import StringIO
        from rich.console import Console

        console = Console(file=StringIO(), width=80)
        adapter = CLIAdapter(console=console)

        result = adapter.handle_command("/clear", [])
        assert result is None

    def test_register_custom_command(self):
        from long.interaction.adapters.cli import CLIAdapter
        from io import StringIO
        from rich.console import Console

        console = Console(file=StringIO(), width=80)
        adapter = CLIAdapter(console=console)

        def custom_handler():
            return "custom result"

        adapter.register_command("/custom", custom_handler)
        result = adapter.handle_command("/custom", [])
        assert result == "custom result"

    def test_stream_output_string(self):
        from long.interaction.adapters.cli import CLIAdapter
        from io import StringIO
        from rich.console import Console

        output = StringIO()
        console = Console(file=output, width=80, force_terminal=True)
        adapter = CLIAdapter(console=console)
        adapter.stream_output("test stream")
        assert "test stream" in output.getvalue()

    def test_commands_list(self):
        from long.interaction.adapters.cli import CLIAdapter

        assert "/help" in CLIAdapter.COMMANDS
        assert "/exit" in CLIAdapter.COMMANDS
        assert "/status" in CLIAdapter.COMMANDS
        assert "/skill" in CLIAdapter.COMMANDS
        assert "/mcp" in CLIAdapter.COMMANDS
