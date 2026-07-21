"""LLM 模块完整测试 — 配置解析/客户端/预算控制/裁判函数"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from long.llm.base import (
    BudgetConfig,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMUsage,
    ModelConfig,
    ModelProvider,
    RetryConfig,
    TimeoutConfig,
)
from long.llm.client import LLMClient


class TestLLMConfig:
    def test_default_config(self) -> None:
        config = LLMConfig()
        assert config.provider == ModelProvider.OPENAI
        assert config.model == "gpt-4o"
        assert config.default_params.model == "gpt-4o"
        assert config.default_params.temperature == 0.7
        assert config.retry.max_retries == 3
        assert config.budget.max_tokens_per_task == 200000

    def test_from_dict(self) -> None:
        data = {
            "provider": "anthropic",
            "model": "claude-3-opus",
            "api_key": "sk-test",
            "default_params": {
                "model": "claude-3-opus",
                "temperature": 0.5,
                "max_tokens": 2048,
            },
            "models": {
                "judge": {
                    "model": "claude-3-haiku",
                    "temperature": 0.1,
                    "max_tokens": 1024,
                }
            },
        }
        config = LLMConfig(**data)
        assert config.provider == ModelProvider.ANTHROPIC
        assert config.model == "claude-3-opus"
        assert config.default_params.temperature == 0.5
        judge_config = config.get_model_config("judge")
        assert judge_config.model == "claude-3-haiku"
        assert judge_config.temperature == 0.1

    def test_resolve_api_key_direct(self) -> None:
        config = LLMConfig(api_key="sk-direct-key")
        assert config.resolve_api_key() == "sk-direct-key"

    def test_resolve_api_key_env_var(self) -> None:
        config = LLMConfig(api_key="${TEST_LLM_KEY}")
        with patch.dict(os.environ, {"TEST_LLM_KEY": "sk-env-key"}):
            assert config.resolve_api_key() == "sk-env-key"

    def test_resolve_api_key_env_var_with_default(self) -> None:
        config = LLMConfig(api_key="${TEST_MISSING_KEY:default-key}")
        with patch.dict(os.environ, {}, clear=True):
            assert config.resolve_api_key() == "default-key"

    def test_resolve_api_key_fallback_to_provider_env(self) -> None:
        config = LLMConfig(provider=ModelProvider.OPENAI)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai-key"}, clear=True):
            assert config.resolve_api_key() == "sk-openai-key"

    def test_resolve_base_url_env_var(self) -> None:
        config = LLMConfig(base_url="${LLM_BASE_URL:http://localhost:8080}")
        with patch.dict(os.environ, {}, clear=True):
            assert config.resolve_base_url() == "http://localhost:8080"

    def test_get_model_config_fallback(self) -> None:
        config = LLMConfig()
        unknown_config = config.get_model_config("unknown_purpose")
        assert unknown_config.model == "gpt-4o"

    def test_get_model_config_specific(self) -> None:
        config = LLMConfig(
            models={
                "planning": ModelConfig(model="gpt-4o", temperature=0.3, max_tokens=8192),
            }
        )
        planning_config = config.get_model_config("planning")
        assert planning_config.model == "gpt-4o"
        assert planning_config.temperature == 0.3
        assert planning_config.max_tokens == 8192


class TestModelProvider:
    def test_provider_values(self) -> None:
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.ANTHROPIC.value == "anthropic"
        assert ModelProvider.CUSTOM.value == "custom"


class TestLLMMessage:
    def test_message_creation(self) -> None:
        msg = LLMMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None

    def test_message_with_name(self) -> None:
        msg = LLMMessage(role="assistant", content="Hi", name="bot")
        assert msg.name == "bot"


class TestLLMResponse:
    def test_response_creation(self) -> None:
        resp = LLMResponse(content="Hello!", model="gpt-4o")
        assert resp.content == "Hello!"
        assert resp.model == "gpt-4o"
        assert resp.usage.total_tokens == 0

    def test_response_with_usage(self) -> None:
        usage = LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        resp = LLMResponse(content="Hi", model="gpt-4o", usage=usage)
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 20
        assert resp.usage.total_tokens == 30


class TestLLMClient:
    def test_client_init_default(self) -> None:
        client = LLMClient()
        assert client.config.provider == ModelProvider.OPENAI
        assert client.daily_tokens_used == 0
        assert client.task_tokens_used == 0

    def test_client_init_with_dict(self) -> None:
        client = LLMClient({"provider": "anthropic", "model": "claude-3-opus"})
        assert client.config.provider == ModelProvider.ANTHROPIC
        assert client.config.model == "claude-3-opus"

    def test_client_init_with_config(self) -> None:
        config = LLMConfig(provider=ModelProvider.CUSTOM, base_url="http://localhost:8080")
        client = LLMClient(config)
        assert client.config.base_url == "http://localhost:8080"

    def test_client_has_async_openai(self) -> None:
        client = LLMClient()
        from openai import AsyncOpenAI

        assert isinstance(client.client, AsyncOpenAI)

    def test_budget_tracking(self) -> None:
        client = LLMClient()
        usage = LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        client._track_usage(usage)
        assert client.daily_tokens_used == 150
        assert client.task_tokens_used == 150
        assert client.request_count == 1

    def test_reset_task_budget(self) -> None:
        client = LLMClient()
        usage = LLMUsage(total_tokens=100)
        client._track_usage(usage)
        client.reset_task_budget()
        assert client.task_tokens_used == 0
        assert client.daily_tokens_used == 100

    def test_reset_daily_budget(self) -> None:
        client = LLMClient()
        usage = LLMUsage(total_tokens=100)
        client._track_usage(usage)
        client.reset_daily_budget()
        assert client.daily_tokens_used == 0

    def test_check_budget_exceeds_task(self) -> None:
        from long.errors import LLMBudgetExceededError
        config = LLMConfig(budget=BudgetConfig(max_tokens_per_task=100))
        client = LLMClient(config)
        client._track_usage(LLMUsage(total_tokens=80))
        with pytest.raises(LLMBudgetExceededError, match="超出单任务预算"):
            client._check_budget(50)

    def test_check_budget_exceeds_daily(self) -> None:
        from long.errors import LLMBudgetExceededError
        config = LLMConfig(budget=BudgetConfig(daily_token_limit=100))
        client = LLMClient(config)
        client._track_usage(LLMUsage(total_tokens=80))
        with pytest.raises(LLMBudgetExceededError, match="超出日预算"):
            client._check_budget(50)

    @pytest.mark.asyncio
    async def test_chat_no_api_key_raises(self) -> None:
        from long.errors import LLMError
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig(api_key="")
            client = LLMClient(config)
            with pytest.raises(LLMError, match="未配置 API 密钥"):
                await client.chat([LLMMessage(role="user", content="test")])

    @pytest.mark.asyncio
    async def test_chat_calls_sdk(self) -> None:
        client = LLMClient({"api_key": "sk-test", "model": "gpt-4o"})

        # Mock the streaming response that _chat_impl iterates over
        class MockChunk:
            def __init__(self):
                self.choices = []
                self.usage = None

        class MockChunkChoice:
            def __init__(self, delta=None, finish_reason=None):
                self.delta = delta
                self.finish_reason = finish_reason

        class MockDelta:
            def __init__(self, content=""):
                self.content = content

        # First chunk with content
        chunk1 = MockChunk()
        chunk1.choices = [MockChunkChoice(delta=MockDelta(content="Hello"))]

        # Second chunk with content
        chunk2 = MockChunk()
        chunk2.choices = [MockChunkChoice(delta=MockDelta(content=" back!"))]

        # Final chunk with finish_reason and usage
        chunk3 = MockChunk()
        chunk3.choices = [MockChunkChoice(delta=MockDelta(), finish_reason="stop")]
        chunk3.usage = MagicMock()
        chunk3.usage.prompt_tokens = 5
        chunk3.usage.completion_tokens = 10
        chunk3.usage.total_tokens = 15

        class MockAsyncIterable:
            def __init__(self, chunks):
                self._chunks = chunks
                self._index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._index >= len(self._chunks):
                    raise StopAsyncIteration
                chunk = self._chunks[self._index]
                self._index += 1
                return chunk

        mock_stream = MockAsyncIterable([chunk1, chunk2, chunk3])
        client._client.chat.completions.create = AsyncMock(return_value=mock_stream)

        result = await client.chat([LLMMessage(role="user", content="Hello")])
        assert result.content == "Hello back!"
        assert result.usage.total_tokens == 15
        assert client.daily_tokens_used == 15

    def test_get_judge_fn(self) -> None:
        client = LLMClient()
        judge_fn = client.get_judge_fn()
        assert callable(judge_fn)

    def test_get_repair_fn(self) -> None:
        client = LLMClient()
        repair_fn = client.get_repair_fn()
        assert callable(repair_fn)


class TestRetryConfig:
    def test_default_values(self) -> None:
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.backoff_factor == 2.0


class TestTimeoutConfig:
    def test_default_values(self) -> None:
        config = TimeoutConfig()
        assert config.connect == 15
        assert config.read == 180
        assert config.write == 30


class TestBudgetConfig:
    def test_default_values(self) -> None:
        config = BudgetConfig()
        assert config.max_tokens_per_task == 200000
        assert config.daily_token_limit == 1000000
        assert config.max_tokens_per_request == 16384
