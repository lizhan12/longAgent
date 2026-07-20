"""LLM 基础类型定义"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    """模型提供商"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


class RetryConfig(BaseModel):
    """重试配置"""

    max_retries: int = 3
    base_delay: float = 2.0
    max_delay: float = 120.0
    backoff_factor: float = 2.0


class FallbackConfig(BaseModel):
    """模型降级配置"""

    enabled: bool = True
    chain: list[str] = Field(default_factory=lambda: ["gpt-4o-mini"])


class TimeoutConfig(BaseModel):
    """超时配置"""

    connect: int = 15
    read: int = 180
    write: int = 30


class ProxyConfig(BaseModel):
    """代理配置"""

    http_proxy: str = ""
    https_proxy: str = ""


class BudgetConfig(BaseModel):
    """预算控制"""

    max_tokens_per_task: int = 200000
    daily_token_limit: int = 1000000
    max_tokens_per_request: int = 16384


class CacheConfig(BaseModel):
    """缓存配置"""

    enabled: bool = True
    ttl: int = 120          # 响应缓存 TTL（秒）
    max_size: int = 200     # 最大缓存条目数
    tool_ttl: int = 300     # 工具结果缓存 TTL（秒）


class ModelConfig(BaseModel):
    """单个模型配置"""

    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    response_format: dict[str, Any] | None = None


class LLMConfig(BaseModel):
    """LLM 完整配置"""

    provider: ModelProvider = ModelProvider.OPENAI
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = ""
    default_params: ModelConfig = Field(default_factory=ModelConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    models: dict[str, ModelConfig] = Field(default_factory=dict)

    def resolve_api_key(self) -> str:
        """解析 API 密钥，支持环境变量

        优先级: config.api_key 直接值 > ${VAR} 解析 > LLM_API_KEY 环境变量 > 提供商专属环境变量
        """
        key = self.api_key
        if key.startswith("${") and key.endswith("}"):
            env_name = key[2:-1]
            if ":" in env_name:
                env_name, default = env_name.split(":", 1)
                return os.environ.get(env_name, default)
            return os.environ.get(env_name, "")
        if not key:
            val = os.environ.get("LLM_API_KEY", "")
            if val:
                return val
            provider_env = {
                ModelProvider.OPENAI: "OPENAI_API_KEY",
                ModelProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            }
            env_name = provider_env.get(self.provider, "LLM_API_KEY")
            return os.environ.get(env_name, "")
        return key

    def resolve_base_url(self) -> str:
        """解析 Base URL，支持环境变量

        优先级: config.base_url 直接值 > ${VAR} 解析 > LLM_BASE_URL 环境变量
        """
        url = self.base_url
        if url.startswith("${") and url.endswith("}"):
            env_name = url[2:-1]
            if ":" in env_name:
                env_name, default = env_name.split(":", 1)
                return os.environ.get(env_name, default)
            return os.environ.get(env_name, "")
        if not url:
            return os.environ.get("LLM_BASE_URL", "")
        return url

    def get_model_config(self, purpose: str) -> ModelConfig:
        """获取指定用途的模型配置

        Args:
            purpose: 用途名称 (planning/repair/judge/chat)

        Returns:
            该用途的模型配置，如无则返回以顶层 model 为默认的配置
        """
        if purpose in self.models:
            return self.models[purpose]
        return ModelConfig(
            model=self.model,
            temperature=self.default_params.temperature,
            max_tokens=self.default_params.max_tokens,
            top_p=self.default_params.top_p,
        )


class LLMMessage(BaseModel):
    """LLM 消息"""

    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class LLMUsage(BaseModel):
    """Token 使用统计"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """LLM 响应"""

    content: str
    model: str = ""
    usage: LLMUsage = Field(default_factory=LLMUsage)
    finish_reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
