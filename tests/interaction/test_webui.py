"""WebUI 适配器测试

覆盖 WebUIAdapter 的核心功能、WebSocket 连接、HITL 审批、REST API 等。
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from long.interaction.adapters.webui import WebUIAdapter, WebSocketConnection
from long.interaction.base import (
    HITLRequest,
    HITLResponse,
    InteractionEvent,
    InteractionEventType,
    InteractionProtocol,
)


# ========================
# WebUIAdapter 基础测试
# ========================


class TestWebUIAdapterInit:
    """WebUI 适配器初始化测试"""

    def test_default_init(self):
        adapter = WebUIAdapter()
        assert adapter.host == "0.0.0.0"
        assert adapter.port == 8000
        assert adapter._active is False
        assert adapter.active_connections == 0

    def test_custom_init(self):
        adapter = WebUIAdapter(host="127.0.0.1", port=9000)
        assert adapter.host == "127.0.0.1"
        assert adapter.port == 9000

    def test_implements_interaction_protocol(self):
        adapter = WebUIAdapter()
        assert isinstance(adapter, InteractionProtocol)


class TestWebUIAdapterSession:
    """WebUI 适配器会话管理测试"""

    def test_start_session(self):
        adapter = WebUIAdapter()
        adapter.start_session()
        assert adapter._active is True

    def test_end_session(self):
        adapter = WebUIAdapter()
        adapter.start_session()
        adapter.end_session()
        assert adapter._active is False

    def test_end_session_clears_connections(self):
        adapter = WebUIAdapter()
        adapter.start_session()
        adapter.end_session()
        assert len(adapter._connections) == 0


class TestWebUIAdapterSendEvent:
    """WebUI 适配器事件发送测试"""

    def test_send_event_when_not_active(self):
        """未启动时发送事件不报错"""
        adapter = WebUIAdapter()
        event = InteractionEvent(
            type=InteractionEventType.MESSAGE,
            content="test",
        )
        adapter.send_event(event)  # 不应报错

    def test_send_event_message(self):
        adapter = WebUIAdapter()
        adapter.start_session()

        event = InteractionEvent(
            type=InteractionEventType.MESSAGE,
            content="Hello",
            metadata={"format": "plain"},
        )
        adapter.send_event(event)  # 无连接也应不报错

    def test_send_event_all_types(self):
        adapter = WebUIAdapter()
        adapter.start_session()

        for event_type in InteractionEventType:
            event = InteractionEvent(type=event_type, content="test")
            adapter.send_event(event)  # 所有事件类型都不应报错


class TestWebUIAdapterStreamOutput:
    """WebUI 适配器流式输出测试"""

    def test_stream_output_string(self):
        adapter = WebUIAdapter()
        adapter.start_session()
        adapter.stream_output("hello")  # 不应报错

    def test_stream_output_non_string(self):
        adapter = WebUIAdapter()
        adapter.start_session()
        adapter.stream_output(42)  # 非字符串转为 str


class TestWebUIAdapterReceiveInput:
    """WebUI 适配器输入接收测试"""

    def test_receive_input_returns_empty(self):
        """WebUI 模式下同步 receive_input 返回空字符串"""
        adapter = WebUIAdapter()
        result = adapter.receive_input()
        assert result == ""

    def test_receive_input_with_prompt(self):
        adapter = WebUIAdapter()
        result = adapter.receive_input("Enter: ")
        assert result == ""


class TestWebUIAdapterHITL:
    """WebUI 适配器 HITL 审批测试"""

    def test_request_feedback_no_response(self):
        """无回调时返回 pending"""
        adapter = WebUIAdapter()
        adapter.start_session()

        request = HITLRequest(
            request_id="test-r1",
            title="Test Approval",
            description="Please approve",
        )
        response = adapter.request_feedback(request)
        assert response.request_id == "test-r1"
        assert response.decision == "pending"

    def test_request_feedback_with_existing_response(self):
        """已有响应时直接返回"""
        adapter = WebUIAdapter()
        adapter.start_session()

        request = HITLRequest(
            request_id="test-r2",
            title="Test",
            description="desc",
        )

        # 预设响应
        adapter._hitl_responses["test-r2"] = HITLResponse(
            request_id="test-r2",
            decision="approve",
        )

        response = adapter.request_feedback(request)
        assert response.decision == "approve"
        assert "test-r2" not in adapter._hitl_responses  # 已消费

    @pytest.mark.asyncio
    async def test_request_feedback_async_timeout(self):
        """异步 HITL 请求超时"""
        adapter = WebUIAdapter()
        adapter.start_session()

        request = HITLRequest(
            request_id="test-r3",
            title="Timeout Test",
            description="desc",
            timeout_seconds=0.1,
        )

        response = await adapter.request_feedback_async(request)
        assert response.decision == "timeout"

    @pytest.mark.asyncio
    async def test_request_feedback_async_with_response(self):
        """异步 HITL 请求收到响应"""
        adapter = WebUIAdapter()
        adapter.start_session()

        request = HITLRequest(
            request_id="test-r4",
            title="Async Test",
            description="desc",
            timeout_seconds=5.0,
        )

        # 模拟延迟响应
        async def respond_later():
            await asyncio.sleep(0.1)
            adapter._hitl_responses["test-r4"] = HITLResponse(
                request_id="test-r4",
                decision="approve",
            )
            if "test-r4" in adapter._hitl_events:
                adapter._hitl_events["test-r4"].set()

        asyncio.create_task(respond_later())

        response = await adapter.request_feedback_async(request)
        assert response.decision == "approve"


# ========================
# WebSocketConnection 测试
# ========================


class TestWebSocketConnection:
    """WebSocket 连接管理测试"""

    def test_connection_properties(self):
        """WebSocketConnection 属性"""
        # 用 None 模拟（实际测试通过 TestClient）
        conn = WebSocketConnection.__new__(WebSocketConnection)
        conn.session_id = "test-session"
        conn._connected = False

        assert conn.session_id == "test-session"
        assert conn.connected is False

    def test_disconnect(self):
        conn = WebSocketConnection.__new__(WebSocketConnection)
        conn._connected = True
        conn.disconnect()
        assert conn.connected is False


# ========================
# REST API 测试
# ========================


class TestWebUIRESTAPI:
    """REST API 端点测试"""

    def setup_method(self):
        self.adapter = WebUIAdapter()
        self.client = TestClient(self.adapter.app)

    def test_index_page(self):
        response = self.client.get("/")
        assert response.status_code == 200
        assert "Long" in response.text

    def test_chat_api(self):
        response = self.client.post(
            "/api/chat/test-session",
            json={"content": "Hello"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session"
        assert data["status"] == "received"

    def test_hitl_callback(self):
        response = self.client.post(
            "/api/hitl/test-req-1",
            json={"decision": "approve", "feedback": "Looks good"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["request_id"] == "test-req-1"

        # 验证响应已存储
        assert "test-req-1" in self.adapter._hitl_responses
        assert self.adapter._hitl_responses["test-req-1"].decision == "approve"

    def test_hitl_callback_reject(self):
        response = self.client.post(
            "/api/hitl/test-req-2",
            json={"decision": "reject"},
        )
        assert response.status_code == 200
        assert self.adapter._hitl_responses["test-req-2"].decision == "reject"

    def test_skills_api(self):
        response = self.client.get("/api/skills")
        assert response.status_code == 200
        assert "skills" in response.json()

    def test_mcp_api(self):
        response = self.client.get("/api/mcp")
        assert response.status_code == 200
        assert "servers" in response.json()


# ========================
# WebSocket 测试
# ========================


class TestWebUIWebSocket:
    """WebSocket 连接测试"""

    def setup_method(self):
        self.adapter = WebUIAdapter()
        self.client = TestClient(self.adapter.app)

    def test_websocket_connect(self):
        with self.client.websocket_connect("/ws/test-session") as ws:
            # 连接成功
            assert "test-session" in self.adapter._connections

    def test_websocket_disconnect(self):
        with self.client.websocket_connect("/ws/test-session") as ws:
            pass
        # 断开后连接应被清除
        assert "test-session" not in self.adapter._connections or \
               not self.adapter._connections.get("test-session", WebSocketConnection.__new__(WebSocketConnection)).connected

    def test_websocket_hitl_response(self):
        with self.client.websocket_connect("/ws/test-session") as ws:
            ws.send_json({
                "type": "hitl_response",
                "request_id": "ws-hitl-1",
                "decision": "approve",
            })

            # 等待处理
            import time
            time.sleep(0.1)

            assert "ws-hitl-1" in self.adapter._hitl_responses
            assert self.adapter._hitl_responses["ws-hitl-1"].decision == "approve"

    def test_websocket_user_input(self):
        with self.client.websocket_connect("/ws/test-session") as ws:
            ws.send_json({
                "type": "user_input",
                "content": "Hello from WebSocket",
            })
            # user_input 消息由 receive_input 异步处理，不报错即可

    def test_active_connections_count(self):
        with self.client.websocket_connect("/ws/session-1"):
            assert self.adapter.active_connections >= 1


# ========================
# 集成测试
# ========================


class TestWebUIIntegration:
    """WebUI 集成测试"""

    def test_hitl_via_rest_then_check(self):
        """通过 REST API 提交 HITL 响应，然后通过适配器获取"""
        adapter = WebUIAdapter()
        adapter.start_session()
        client = TestClient(adapter.app)

        # 先通过 REST 提交审批
        client.post(
            "/api/hitl/integration-1",
            json={"decision": "approve"},
        )

        # 然后通过适配器获取
        request = HITLRequest(
            request_id="integration-1",
            title="Integration Test",
            description="test",
        )
        response = adapter.request_feedback(request)
        assert response.decision == "approve"

    def test_websocket_hitl_then_async_get(self):
        """通过 WebSocket 提交 HITL 响应"""
        adapter = WebUIAdapter()
        adapter.start_session()
        client = TestClient(adapter.app)

        with client.websocket_connect("/ws/integration-session") as ws:
            ws.send_json({
                "type": "hitl_response",
                "request_id": "ws-integration-1",
                "decision": "reject",
                "feedback": "Not safe",
            })

        import time
        time.sleep(0.1)

        assert "ws-integration-1" in adapter._hitl_responses
        assert adapter._hitl_responses["ws-integration-1"].decision == "reject"
        assert adapter._hitl_responses["ws-integration-1"].feedback == "Not safe"

    def test_send_event_to_websocket(self):
        """通过适配器发送事件，WebSocket 客户端接收"""
        adapter = WebUIAdapter()
        adapter.start_session()
        client = TestClient(adapter.app)

        with client.websocket_connect("/ws/event-session") as ws:
            adapter.send_event(InteractionEvent(
                type=InteractionEventType.MESSAGE,
                content="Broadcast message",
            ))

            data = ws.receive_json()
            assert data["type"] == "message"
            assert data["content"] == "Broadcast message"

    def test_stream_output_to_websocket(self):
        """流式输出通过 WebSocket 发送"""
        adapter = WebUIAdapter()
        adapter.start_session()
        client = TestClient(adapter.app)

        with client.websocket_connect("/ws/stream-session") as ws:
            adapter.stream_output("stream token")

            data = ws.receive_json()
            assert data["type"] == "stream_token"
            assert data["content"] == "stream token"
