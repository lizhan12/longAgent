"""对话压缩器 — 当上下文过长时自动压缩历史消息

Harness Engineering 核心理念：Agent 应该管理自己的上下文，而不是被上下文压垮。
当工具调用多轮后 prompt 膨胀到阈值，自动用 LLM 对历史做摘要压缩。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from long.llm.base import LLMMessage

logger = logging.getLogger(__name__)

COMPRESSION_PROMPT = """你是一个上下文压缩助手。请将以下对话历史压缩为简洁的摘要。

压缩规则：
1. 保留用户的原始需求/问题
2. 保留已获取的关键数据（搜索结果中的具体数字、日期、名称）
3. 保留代码执行的成功/失败状态和关键输出
4. 丢弃冗余的工具调用细节（如文件路径、参数等）
5. 丢弃失败的尝试和错误信息（除非所有尝试都失败）
6. 保留当前待解决的问题
7. 总长度控制在 500 字以内

对话历史：
{history}

请输出压缩后的摘要（直接输出内容，不要加前缀说明）："""


@dataclass
class CompressConfig:
    """压缩配置"""

    max_prompt_chars: int = 12000
    """触发压缩的 prompt 字符数阈值"""
    min_rounds_before_compress: int = 3
    """至少经过这么多轮工具调用后才考虑压缩"""
    keep_recent_rounds: int = 2
    """压缩后保留最近 N 轮的完整消息"""


class DialogCompressor:
    """对话压缩器"""

    def __init__(self, config: CompressConfig | None = None) -> None:
        self._config = config or CompressConfig()
        self._compressed_rounds: int = 0

    def should_compress(self, history_msgs: list[dict[str, Any]], tool_rounds: int) -> bool:
        if tool_rounds < self._config.min_rounds_before_compress:
            return False

        total_chars = sum(len(m.get("content", "")) for m in history_msgs)
        return total_chars > self._config.max_prompt_chars

    def extract_compressible_messages(
        self, history_msgs: list[dict[str, Any]], tool_rounds: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """将历史消息分为：可压缩部分 和 保留部分

        保留：system prompt + 用户消息 + 最近 N 轮工具调用
        压缩：中间的旧工具调用和结果
        """
        keep_count = self._config.keep_recent_rounds * 2
        system_msgs = [m for m in history_msgs if m["role"] == "system"]
        user_msgs = [m for m in history_msgs if m["role"] == "user"]
        rest_msgs = [m for m in history_msgs if m["role"] not in ("system", "user")]

        recent = rest_msgs[-keep_count:] if len(rest_msgs) > keep_count else []
        old = rest_msgs[:-keep_count] if len(rest_msgs) > keep_count else []

        preserved = system_msgs + user_msgs + recent
        
        # 确保保留部分至少有 system + user + assistant 结构
        # 如果压缩后只剩下 system 和 1 条 user message，需要额外保留一些上下文
        has_assistant = any(m["role"] == "assistant" for m in preserved)
        has_tool = any(m["role"] == "tool" for m in preserved)
        
        if not has_assistant and not has_tool and user_msgs and old:
            # 缺少 assistant 响应，从 old 中取最后一条 assistant 消息
            for msg in reversed(old):
                if msg["role"] == "assistant":
                    preserved.insert(len(system_msgs), msg)
                    old.remove(msg)
                    break

        return old, preserved

    def build_compressible_text(self, messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for m in messages:
            role = m["role"]
            content = m.get("content", "")
            if not content:
                tool_calls = m.get("tool_calls", [])
                if tool_calls:
                    calls_desc = ", ".join(
                        tc.get("function", {}).get("name", "?") for tc in tool_calls
                    )
                    lines.append(f"[{role}] 调用工具: {calls_desc}")
                continue
            truncated = content[:800]
            if len(content) > 800:
                truncated += f"...[{len(content)}字符已截断]"
            lines.append(f"[{role}] {truncated}")
        return "\n".join(lines)

    async def compress(
        self, llm: Any, history_msgs: list[dict[str, Any]], tool_rounds: int
    ) -> list[dict[str, Any]]:
        """压缩对话历史

        Returns:
            压缩后的 history_msgs
        """
        old_msgs, preserved = self.extract_compressible_messages(history_msgs, tool_rounds)

        if not old_msgs:
            return history_msgs

        compress_text = self.build_compressible_text(old_msgs)
        prompt = COMPRESSION_PROMPT.format(history=compress_text)

        try:
            response = await llm.chat(
                [
                    LLMMessage(role="system", content="你是一个对话摘要助手，负责压缩对话历史。"),
                    LLMMessage(role="user", content=prompt),
                ],
                purpose="summarize",
            )
            summary = response.content.strip()
        except Exception:
            logger.warning("对话压缩失败，保留原始上下文", exc_info=True)
            return history_msgs

        self._compressed_rounds += 1

        compressed_msg = {
            "role": "system",
            "content": (
                f"[上下文压缩 #{self._compressed_rounds}]\n"
                f"以下为之前对话轮次的摘要：\n{summary}\n"
                f"--- 以下是最近的消息 ---"
            ),
        }

        result = list(preserved)
        insert_pos = 1
        result.insert(insert_pos, compressed_msg)

        logger.info(
            "对话压缩完成: %d 条消息压缩为摘要, 总消息数 %d → %d",
            len(old_msgs), len(history_msgs), len(result),
        )
        return result