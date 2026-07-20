"""SemanticCompressor - 语义压缩模块

两阶段压缩：结构化提取（无需 LLM）+ LLM 摘要（仅超长部分）。
在不丢失关键信息的前提下压榨 Context Window。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .base import MemoryItem

logger = logging.getLogger(__name__)

_ROLES_TO_COMPRESS = {"assistant", "tool"}
_ROLES_TO_PRESERVE = {"system", "user"}
_CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_URL_PATTERN = re.compile(r"https?://\S+")
_NUMBER_PATTERN = re.compile(r"\d+\.?\d*%?")


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文约 1.5 字/token，英文约 4 字符/token）"""
    cn_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    en_chars = len(text) - cn_chars
    return int(cn_chars / 1.5 + en_chars / 4)


def _extract_structure(messages: list[dict]) -> list[dict]:
    """阶段1：结构化提取（无需 LLM）

    - 保留 system/user 消息完整
    - 压缩 assistant 消息：保留代码块 + 关键数字 + 首尾句
    - 压缩 tool 消息：保留结果摘要
    """
    compressed = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if not content:
            continue

        if role in _ROLES_TO_PRESERVE:
            compressed.append(msg)
        elif role in _ROLES_TO_COMPRESS:
            compressed_content = _compress_content(content)
            compressed.append({"role": role, "content": compressed_content})
        else:
            compressed.append(msg)

    return compressed


def _compress_content(content: str) -> str:
    """压缩单条内容：保留代码块 + 关键信息 + 首尾句"""
    code_blocks = _CODE_BLOCK_PATTERN.findall(content)
    urls = _URL_PATTERN.findall(content)
    numbers = _NUMBER_PATTERN.findall(content)

    text_without_code = _CODE_BLOCK_PATTERN.sub("[CODE]", content)
    sentences = re.split(r"[。！？\n.!?]", text_without_code)
    sentences = [s.strip() for s in sentences if s.strip()]

    parts = []
    if sentences:
        parts.append(sentences[0])
        if len(sentences) > 1:
            parts.append(sentences[-1])

    if numbers:
        unique_nums = list(dict.fromkeys(numbers))[:5]
        parts.append("关键数据: " + ", ".join(unique_nums))

    if urls:
        parts.append("相关链接: " + ", ".join(urls[:3]))

    result = "。".join(p for p in parts if p)

    if code_blocks:
        for block in code_blocks[:2]:
            result += "\n" + block

    return result if result else content[:200]


class SemanticCompressor:
    """语义压缩器

    两阶段压缩：
    1. 结构化提取：保留关键信息，压缩冗余内容
    2. LLM 摘要：仅对超长部分调用 LLM 压缩

    Attributes:
        max_tokens: 压缩后的最大 token 数
        llm_client: LLM 客户端（可选，用于阶段2）
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        llm_client: Any | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.llm_client = llm_client

    async def compress(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
    ) -> list[dict]:
        """压缩消息列表

        Args:
            messages: 消息列表
            max_tokens: 压缩后的最大 token 数

        Returns:
            压缩后的消息列表
        """
        target_tokens = max_tokens or self.max_tokens

        if not messages:
            return messages

        total_tokens = sum(_estimate_tokens(m.get("content", "")) for m in messages)

        if total_tokens <= target_tokens:
            return messages

        stage1 = _extract_structure(messages)
        stage1_tokens = sum(_estimate_tokens(m.get("content", "")) for m in stage1)

        if stage1_tokens <= target_tokens:
            logger.info(
                "语义压缩: 阶段1完成, %d→%d tokens",
                total_tokens,
                stage1_tokens,
            )
            return stage1

        if self.llm_client is not None:
            stage2 = await self._llm_summarize(stage1, target_tokens)
            stage2_tokens = sum(_estimate_tokens(m.get("content", "")) for m in stage2)
            logger.info(
                "语义压缩: 阶段2完成, %d→%d→%d tokens",
                total_tokens,
                stage1_tokens,
                stage2_tokens,
            )
            return stage2

        ratio = target_tokens / max(stage1_tokens, 1)
        truncated = self._truncate_messages(stage1, target_tokens)
        logger.info(
            "语义压缩: 截断完成, %d→%d tokens (ratio=%.2f)",
            total_tokens,
            sum(_estimate_tokens(m.get("content", "")) for m in truncated),
            ratio,
        )
        return truncated

    async def _llm_summarize(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> list[dict]:
        """阶段2：LLM 摘要"""
        try:
            from long.llm.base import LLMMessage

            conversation = "\n".join(
                f"[{m.get('role', 'unknown')}]: {m.get('content', '')}"
                for m in messages
            )

            prompt_messages = [
                LLMMessage(
                    role="system",
                    content=(
                        "你是一个对话压缩助手。请将以下对话压缩为简洁版本，"
                        "保留所有关键信息（用户需求、决策、代码、数据），"
                        "去除冗余和重复。输出格式与输入相同：每行以[role]:开头。"
                    ),
                ),
                LLMMessage(
                    role="user",
                    content=f"请压缩以下对话（目标 {max_tokens} tokens）：\n\n{conversation}",
                ),
            ]

            response = await self.llm_client.chat(prompt_messages, purpose="compress")
            summary = response.content.strip()

            if not summary:
                return messages

            compressed_messages = []
            for line in summary.split("\n"):
                line = line.strip()
                if not line:
                    continue
                match = re.match(r"\[(\w+)\]:\s*(.*)", line)
                if match:
                    compressed_messages.append({
                        "role": match.group(1).lower(),
                        "content": match.group(2),
                    })
                else:
                    if compressed_messages:
                        compressed_messages[-1]["content"] += "\n" + line
                    else:
                        compressed_messages.append({"role": "assistant", "content": line})

            return compressed_messages if compressed_messages else messages

        except Exception as e:
            logger.warning("LLM 摘要失败: %s，回退到截断", e)
            return self._truncate_messages(messages, max_tokens)

    def _truncate_messages(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> list[dict]:
        """截断消息列表到目标 token 数"""
        result = []
        remaining = max_tokens

        for msg in messages:
            content = msg.get("content", "")
            msg_tokens = _estimate_tokens(content)

            if msg_tokens <= remaining:
                result.append(msg)
                remaining -= msg_tokens
            else:
                ratio = remaining / max(msg_tokens, 1)
                truncated_len = int(len(content) * ratio)
                if truncated_len > 50:
                    result.append({
                        **msg,
                        "content": content[:truncated_len] + "...[truncated]",
                    })
                break

        return result

    def compress_items(self, items: list[MemoryItem], max_tokens: int = 2000) -> list[MemoryItem]:
        """压缩 MemoryItem 列表（同步，仅阶段1）"""
        if not items:
            return items

        total = sum(_estimate_tokens(item.content) for item in items)
        if total <= max_tokens:
            return items

        result = []
        remaining = max_tokens

        for item in items:
            tokens = _estimate_tokens(item.content)
            if tokens <= remaining:
                result.append(item)
                remaining -= tokens
            else:
                ratio = remaining / max(tokens, 1)
                truncated_len = int(len(item.content) * ratio)
                if truncated_len > 50:
                    truncated = item.model_copy(update={
                        "content": item.content[:truncated_len] + "...[compressed]"
                    })
                    result.append(truncated)
                break

        return result
