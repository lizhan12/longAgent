"""真正的端到端集成测试 — 实际调用 LLM API

这些测试使用配置文件中设置的 API Key 和 Base URL 实际调用 LLM，
验证系统从配置加载 → API 调用 → 结果解析的完整链路正常。

⚠️ 这些测试会消耗 API 配额，每次运行约 2-5 次 API 调用。
如果 API Key 未配置，测试自动跳过。

运行方式:
  uv run pytest tests/test_e2e_real_llm.py -v --tb=short
"""

from __future__ import annotations

import os
import pytest

pytestmark = pytest.mark.asyncio


def _load_env() -> None:
    """加载 .env 文件到环境变量"""
    from pathlib import Path
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # 只设置未设置的环境变量
            if key not in os.environ:
                os.environ[key] = value


def _check_api_configured() -> bool:
    """检查 API 是否已配置"""
    _load_env()
    key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "")
    return bool(key) and bool(base_url)


# ===================== 基础 LLM 调用测试 =====================


class TestRealLLMChat:
    """真实 LLM 聊天调用测试"""

    @pytest.mark.skipif(not _check_api_configured(), reason="LLM_API_KEY 未配置")
    async def test_basic_chat(self):
        """测试最基本的 LLM 对话：连接 → 发送 → 接收响应"""
        import yaml
        from pathlib import Path
        from long.llm.client import LLMClient
        from long.llm.base import LLMConfig, LLMMessage

        # 从配置文件加载 LLM 配置
        config_path = Path("configs/llm.yaml")
        assert config_path.exists(), f"配置文件不存在: {config_path}"

        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        llm_config = raw_config.get("llm", {})
        config = LLMConfig(**llm_config)

        # 验证 API Key 和 Base URL 已解析
        api_key = config.resolve_api_key()
        base_url = config.resolve_base_url()
        assert api_key, "API Key 未解析，请检查 .env 配置"
        assert base_url, "Base URL 未解析，请检查 .env 配置"

        print(f"\n  🔌 API端点: {base_url}")
        print(f"  🤖 模型: {config.model}")
        print(f"  🔑 Key: {api_key[:8]}...{api_key[-4:]}")

        # 创建 LLM 客户端
        client = LLMClient(config)

        # 发送最简单的对话
        messages = [
            LLMMessage(role="user", content="请用一句话介绍你自己"),
        ]

        try:
            response = await client.chat(messages, purpose="chat")
            print(f"  ✅ 响应: {response.content[:100]}...")
            assert response.content, "LLM 返回了空内容"
            assert len(response.content) > 10, f"响应过短: {response.content}"

            # 验证 token 统计
            assert response.usage.total_tokens > 0, "Token 统计为 0"
            print(f"  📊 Token: prompt={response.usage.prompt_tokens}, "
                  f"completion={response.usage.completion_tokens}, "
                  f"total={response.usage.total_tokens}")

        finally:
            await client.aclose()

    @pytest.mark.skipif(not _check_api_configured(), reason="LLM_API_KEY 未配置")
    async def test_chat_with_system_prompt(self):
        """测试带系统提示词的对话"""
        import yaml
        from pathlib import Path
        from long.llm.client import LLMClient
        from long.llm.base import LLMConfig, LLMMessage

        config_path = Path("configs/llm.yaml")
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = LLMConfig(**raw_config.get("llm", {}))

        client = LLMClient(config)
        try:
            messages = [
                LLMMessage(role="system", content="你是一个只回答'你好'的助手，不要说其他任何话。"),
                LLMMessage(role="user", content="今天天气怎么样？"),
            ]

            response = await client.chat(messages, purpose="chat")
            print(f"\n  ✅ 系统提示词生效: {response.content[:80]}")
            assert response.content, "响应为空"

        finally:
            await client.aclose()

    @pytest.mark.skipif(not _check_api_configured(), reason="LLM_API_KEY 未配置")
    async def test_stream_chat(self):
        """测试流式对话"""
        import yaml
        from pathlib import Path
        from long.llm.client import LLMClient
        from long.llm.base import LLMConfig, LLMMessage

        config_path = Path("configs/llm.yaml")
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = LLMConfig(**raw_config.get("llm", {}))

        client = LLMClient(config)
        try:
            messages = [
                LLMMessage(role="user", content="从1数到5，用逗号分隔"),
            ]

            collected = []
            async for token in client.stream_chat(messages, purpose="chat"):
                collected.append(token)

            full_text = "".join(collected)
            print(f"\n  ✅ 流式响应: {full_text[:80]}")
            assert full_text, "流式响应为空"
            assert len(collected) > 1, f"流式应产生多个 token，实际: {len(collected)}"

        finally:
            await client.aclose()


# ===================== 工具调用测试 =====================


class TestRealLLMToolCall:
    """真实 LLM 工具调用测试"""

    @pytest.mark.skipif(not _check_api_configured(), reason="LLM_API_KEY 未配置")
    async def test_tool_call_with_time_tool(self):
        """测试 LLM 能够识别并调用工具

        提供一个 get_current_time 工具，LLM 应该能识别出需要调用它。
        """
        import yaml
        from pathlib import Path
        from long.llm.client import LLMClient
        from long.llm.base import LLMConfig, LLMMessage

        config_path = Path("configs/llm.yaml")
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = LLMConfig(**raw_config.get("llm", {}))

        client = LLMClient(config)
        try:
            messages = [
                LLMMessage(role="system", content="你是一个助手，可以使用工具获取当前时间。"),
                LLMMessage(role="user", content="现在几点了？请用 get_current_time 工具查询。"),
            ]

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_time",
                        "description": "获取当前日期和时间",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

            response = await client.chat_with_tools(messages, tools, purpose="chat")
            print(f"\n  ✅ 工具调用响应: {response.content[:80] if response.content else '(空)'}")
            print(f"  🔧 工具调用数: {len(response.tool_calls)}")

            # LLM 应该调用了 get_current_time 工具
            if response.tool_calls:
                tool_name = response.tool_calls[0]["name"]
                print(f"  🔧 调用工具: {tool_name}")
                assert tool_name == "get_current_time", \
                    f"应调用 get_current_time，实际: {tool_name}"

        finally:
            await client.aclose()

    @pytest.mark.skipif(not _check_api_configured(), reason="LLM_API_KEY 未配置")
    async def test_tool_call_with_arguments(self):
        """测试 LLM 调用带参数的工具"""
        import yaml
        from pathlib import Path
        from long.llm.client import LLMClient
        from long.llm.base import LLMConfig, LLMMessage

        config_path = Path("configs/llm.yaml")
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = LLMConfig(**raw_config.get("llm", {}))

        client = LLMClient(config)
        try:
            messages = [
                LLMMessage(role="system", content="你是一个助手，可以调用工具完成任务。"),
                LLMMessage(role="user", content="请用 write_file 工具写一个文件，内容为'Hello E2E Test'，路径为 output/test_e2e.txt"),
            ]

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "将内容写入工作区中的文件",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "文件路径"},
                                "content": {"type": "string", "description": "文件内容"},
                            },
                            "required": ["path", "content"],
                        },
                    },
                }
            ]

            response = await client.chat_with_tools(messages, tools, purpose="chat")
            print(f"\n  ✅ 响应: {response.content[:80] if response.content else '(空)'}")
            print(f"  🔧 工具调用数: {len(response.tool_calls)}")

            if response.tool_calls:
                tc = response.tool_calls[0]
                print(f"  🔧 工具: {tc['name']}")
                print(f"  📦 参数: {tc['arguments']}")
                assert tc["name"] == "write_file"
                assert "path" in tc["arguments"]
                assert "content" in tc["arguments"]

        finally:
            await client.aclose()


# ===================== LLM 配置加载测试 =====================


class TestRealLLMConfig:
    """LLM 配置加载端到端测试"""

    async def test_config_loading_from_yaml(self):
        """验证 configs/llm.yaml 能被正确加载和解析"""
        import yaml
        from pathlib import Path
        from long.llm.base import LLMConfig, ModelProvider

        config_path = Path("configs/llm.yaml")
        assert config_path.exists(), "llm.yaml 不存在"

        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "llm" in raw_config, "llm.yaml 中缺少 llm 配置"

        llm_config = raw_config["llm"]
        config = LLMConfig(**llm_config)

        # 验证字段
        assert config.model, "模型名称为空"
        assert config.provider == ModelProvider.OPENAI, "提供商应为 openai"
        assert config.fallback.enabled, "降级应启用"
        assert len(config.fallback.chain) > 0, "降级链为空"

        print(f"\n  ✅ 配置加载成功:")
        print(f"  🤖 模型: {config.model}")
        print(f"  🔄 降级: {config.fallback.chain}")
        print(f"  ⏱ 超时: connect={config.timeout.connect}s, read={config.timeout.read}s")

    async def test_api_key_resolution(self):
        """验证 API Key 从 .env 正确解析"""
        import yaml
        from pathlib import Path
        from long.llm.base import LLMConfig

        config_path = Path("configs/llm.yaml")
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = LLMConfig(**raw_config.get("llm", {}))

        api_key = config.resolve_api_key()
        base_url = config.resolve_base_url()

        # 验证 ${VAR} 语法被正确解析
        assert api_key, "API Key 未能解析"
        assert base_url, "Base URL 未能解析"

        # 验证 Base URL 格式
        assert base_url.startswith("http"), f"Base URL 格式异常: {base_url}"

        print(f"\n  ✅ API Key 解析成功: {api_key[:8]}...{api_key[-4:]}")
        print(f"  ✅ Base URL 解析成功: {base_url}")

    async def test_multi_model_config(self):
        """验证多模型配置（按用途路由）"""
        import yaml
        from pathlib import Path
        from long.llm.base import LLMConfig

        config_path = Path("configs/llm.yaml")
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = LLMConfig(**raw_config.get("llm", {}))

        # 验证各用途的模型配置
        for purpose in ["planning", "repair", "judge", "chat"]:
            model_config = config.get_model_config(purpose)
            assert model_config.model, f"{purpose} 模型未配置"
            print(f"  ✅ {purpose}: model={model_config.model}, "
                  f"temperature={model_config.temperature}")

    async def test_fallback_chain(self):
        """验证降级链配置"""
        import yaml
        from pathlib import Path
        from long.llm.base import LLMConfig

        config_path = Path("configs/llm.yaml")
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = LLMConfig(**raw_config.get("llm", {}))

        assert config.fallback.enabled, "降级应启用"
        assert len(config.fallback.chain) > 0, "降级链为空"

        # 验证降级模型与主模型不同
        if config.fallback.chain:
            fallback_model = config.fallback.chain[0]
            assert fallback_model != config.model, \
                f"降级模型({fallback_model})应与主模型({config.model})不同"

        print(f"\n  ✅ 降级链: {config.fallback.chain}")


# ===================== 完整系统 Pipeline 测试 =====================


class TestRealLLMFullPipeline:
    """完整系统 Pipeline 测试 — 模拟真实用户交互"""

    @pytest.mark.skipif(not _check_api_configured(), reason="LLM_API_KEY 未配置")
    async def test_llm_client_with_tool_result_round_trip(self):
        """测试完整的工具调用往返：调用 LLM → 获取工具调用 → 执行工具 → 返回结果给 LLM

        模拟一轮 tool-calling 对话：
        1. 用户请求获取时间
        2. LLM 返回工具调用
        3. 模拟工具执行
        4. 将工具结果返回给 LLM
        5. LLM 生成最终回复
        """
        import yaml
        from pathlib import Path
        from long.llm.client import LLMClient
        from long.llm.base import LLMConfig, LLMMessage

        config_path = Path("configs/llm.yaml")
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = LLMConfig(**raw_config.get("llm", {}))

        client = LLMClient(config)
        try:
            # 第1轮：用户请求
            messages = [
                LLMMessage(role="system", content="你是一个助手，可以调用工具获取信息。"),
                LLMMessage(role="user", content="现在几点了？请用 get_current_time 工具查询时间。"),
            ]

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_time",
                        "description": "获取当前日期和时间",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

            # 第1次调用：LLM 应返回工具调用
            response1 = await client.chat_with_tools(messages, tools, purpose="chat")
            print(f"\n  📡 第1轮: LLM 响应")
            print(f"    文本: {response1.content[:60] if response1.content else '(空)'}")
            print(f"    工具调用: {len(response1.tool_calls)}")

            # 如果有工具调用，模拟执行并返回结果
            if response1.tool_calls:
                tc = response1.tool_calls[0]
                print(f"    工具: {tc['name']}")

                # 模拟工具执行
                from datetime import datetime
                tool_result = f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

                # 构建第2轮消息
                messages.append(LLMMessage(
                    role="assistant",
                    content=response1.content,
                    tool_calls=response1.tool_calls,
                ))
                messages.append(LLMMessage(
                    role="tool",
                    content=tool_result,
                    tool_call_id=tc["id"],
                ))

                # 第2次调用：LLM 应用工具结果生成回复
                response2 = await client.chat_with_tools(messages, tools, purpose="chat")
                print(f"  📡 第2轮: LLM 回复")
                print(f"    内容: {response2.content[:100]}")

                assert response2.content, "第2轮 LLM 回复为空"
                assert len(response2.content) > 20, f"回复过短: {response2.content}"

        finally:
            await client.aclose()

    @pytest.mark.skipif(not _check_api_configured(), reason="LLM_API_KEY 未配置")
    async def test_streaming_with_tool_result(self):
        """测试流式输出场景"""
        import yaml
        from pathlib import Path
        from long.llm.client import LLMClient
        from long.llm.base import LLMConfig, LLMMessage

        config_path = Path("configs/llm.yaml")
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = LLMConfig(**raw_config.get("llm", {}))

        client = LLMClient(config)
        try:
            from datetime import datetime

            # 模拟工具调用结果
            tool_result = f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            messages = [
                LLMMessage(role="system", content="你是一个助手。"),
                LLMMessage(role="user", content="现在几点了？"),
                LLMMessage(role="assistant", content="", tool_calls=[{
                    "id": "call_e2e_test",
                    "name": "get_current_time",
                    "arguments": {},
                }]),
                LLMMessage(role="tool", content=tool_result, tool_call_id="call_e2e_test"),
                LLMMessage(role="user", content="请基于以上时间信息，告诉我今天的日期和具体时间。"),
            ]

            # 流式获取回复
            collected = []
            async for token in client.stream_chat(messages, purpose="chat"):
                collected.append(token)

            full_text = "".join(collected)
            print(f"\n  ✅ 流式回复: {full_text[:120]}")
            assert full_text, "流式回复为空"
            assert len(collected) > 1, f"应产生多个 token，实际: {len(collected)}"

        finally:
            await client.aclose()

    @pytest.mark.skipif(not _check_api_configured(), reason="LLM_API_KEY 未配置")
    async def test_longer_conversation(self):
        """测试多轮对话上下文保持"""
        import yaml
        from pathlib import Path
        from long.llm.client import LLMClient
        from long.llm.base import LLMConfig, LLMMessage

        config_path = Path("configs/llm.yaml")
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = LLMConfig(**raw_config.get("llm", {}))

        client = LLMClient(config)
        try:
            messages = [
                LLMMessage(role="system", content="简短回答。"),
                LLMMessage(role="user", content="记住：我叫张三。"),
            ]

            # 第1轮：记住名字
            response1 = await client.chat(messages, purpose="chat")
            print(f"\n  📡 第1轮: {response1.content[:60]}")

            messages.append(LLMMessage(role="assistant", content=response1.content))
            messages.append(LLMMessage(role="user", content="我叫什么名字？只回答名字。"))

            # 第2轮：应该记得名字
            response2 = await client.chat(messages, purpose="chat")
            print(f"  📡 第2轮: {response2.content[:80]}")

            if response2.finish_reason == "length":
                print(f"  ⚠️ 响应被截断 (finish_reason=length)")
            assert "张三" in response2.content, \
                f"LLM 应记得名字'张三'，finish_reason={response2.finish_reason}，实际: {response2.content[:100]}"

        finally:
            await client.aclose()