"""Semantic Compression — 语义感知压缩

替代无语义截断，保护关键信息（数字、日期、名称、异常）。

两阶段压缩：
  Stage 1: 结构化提取（零延迟，纯规则）
  Stage 2: 语义摘要（可选，LLM 辅助）
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class KeyInfoProtector:
    """关键信息保护器

    在压缩前提取包含关键信息的句子，确保不被截断丢失。
    """

    NUMBER_PATTERN = re.compile(r'\d+\.?\d*%?')
    DATE_PATTERN = re.compile(r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?|\d{1,2}月\d{1,2}日')
    NAME_PATTERN = re.compile(r'[A-Z][a-z]+ [A-Z][a-z]+|[\u4e00-\u9fff]{2,4}(?:省|市|县|区|公司|集团)')
    ERROR_PATTERN = re.compile(r'(?:Error|Exception|失败|错误|Traceback|WARNING|Warning|警告).*', re.IGNORECASE)
    SENTENCE_SPLIT = re.compile(r'[。！？\n.!?]+')

    def extract_key_sentences(self, text: str, max_sentences: int = 5) -> list[str]:
        sentences = self.SENTENCE_SPLIT.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        scored = [(s, self._key_info_score(s)) for s in sentences]
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = [s for s, _ in scored[:max_sentences]]
        original_order = []
        for s in sentences:
            if s in selected:
                original_order.append(s)
                selected.remove(s)
                if not selected:
                    break

        return original_order

    def _key_info_score(self, sentence: str) -> float:
        score = 0.0
        if self.NUMBER_PATTERN.search(sentence):
            score += 2.0
        if self.DATE_PATTERN.search(sentence):
            score += 2.0
        if self.NAME_PATTERN.search(sentence):
            score += 1.0
        if self.ERROR_PATTERN.search(sentence):
            score += 3.0
        if len(sentence) > 20:
            score += 0.5
        return score


@dataclass
class CompressionResult:
    """压缩结果"""
    compressed_text: str
    original_length: int
    compressed_length: int
    method: str = "structural"
    key_sentences_preserved: int = 0


class SemanticCompressor:
    """语义感知压缩器

    两阶段压缩策略：
    1. 结构化提取：提取标题、关键句子、首尾段落（零延迟）
    2. 语义摘要：当结构化提取后仍超阈值时，用 LLM 生成摘要
    """

    MAX_SEARCH_LEN = 800
    MAX_CODE_LEN = 2000
    MAX_GENERAL_LEN = 1000

    def __init__(
        self,
        llm_chat_fn: Callable[..., Awaitable[Any]] | None = None,
        key_info_protector: KeyInfoProtector | None = None,
    ) -> None:
        self._llm_chat = llm_chat_fn
        self._protector = key_info_protector or KeyInfoProtector()

    def compress(self, tool_name: str, result: str) -> str:
        """压缩工具结果（同步，纯规则阶段）"""
        if tool_name == "tavily_search":
            return self._compress_search(result)
        if tool_name in ("execute_code", "execute_file"):
            return self._compress_code(result)
        if len(result) <= self.MAX_GENERAL_LEN:
            return result
        return self._compress_general(result)

    async def compress_async(self, tool_name: str, result: str) -> CompressionResult:
        """异步压缩（含 LLM 摘要阶段）"""
        original_length = len(result)

        stage1 = self.compress(tool_name, result)
        if len(stage1) <= self._get_max_len(tool_name):
            return CompressionResult(
                compressed_text=stage1,
                original_length=original_length,
                compressed_length=len(stage1),
                method="structural",
            )

        if self._llm_chat is not None:
            try:
                stage2 = await self._semantic_summarize(stage1, self._get_max_len(tool_name))
                if stage2:
                    return CompressionResult(
                        compressed_text=stage2,
                        original_length=original_length,
                        compressed_length=len(stage2),
                        method="semantic",
                    )
            except Exception as e:
                logger.debug("语义摘要压缩失败: %s", e)

        return CompressionResult(
            compressed_text=stage1,
            original_length=original_length,
            compressed_length=len(stage1),
            method="structural",
        )

    def _get_max_len(self, tool_name: str) -> int:
        if tool_name == "tavily_search":
            return self.MAX_SEARCH_LEN
        if tool_name in ("execute_code", "execute_file"):
            return self.MAX_CODE_LEN
        return self.MAX_GENERAL_LEN

    def _compress_search(self, result: str) -> str:
        if len(result) <= self.MAX_SEARCH_LEN:
            return result

        key_sentences = self._protector.extract_key_sentences(result, max_sentences=3)
        key_budget = self.MAX_SEARCH_LEN // 3
        key_text = "\n".join(key_sentences)
        if len(key_text) > key_budget:
            key_text = key_text[:key_budget] + "..."

        body_budget = self.MAX_SEARCH_LEN - len(key_text) - 80

        lines = result.split("\n")
        compressed = []
        total = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if total + len(stripped) + 1 > body_budget:
                break
            compressed.append(stripped)
            total += len(stripped) + 1

        body = "\n".join(compressed)
        if key_text and key_text not in body:
            return f"{body}\n\n[关键信息] {key_text}\n...(搜索结果已压缩，原始 {len(result)} 字符)"
        return body + f"\n...(搜索结果已压缩，原始 {len(result)} 字符)"

    def _compress_code(self, result: str) -> str:
        if len(result) <= self.MAX_CODE_LEN:
            return result

        # 天气查询结果：保留每个城市的关键行，不粗暴截断
        if any(marker in result for marker in ("【", "实况:", "预报:")) and any(
            city in result for city in ("北京", "上海", "杭州", "广州", "深圳", "阜阳", "成都", "武汉", "南京", "重庆", "天津")
        ):
            return self._compress_weather(result)

        key_sentences = self._protector.extract_key_sentences(result, max_sentences=3)
        key_budget = self.MAX_CODE_LEN // 4
        key_text = "\n".join(key_sentences)
        if len(key_text) > key_budget:
            key_text = key_text[:key_budget] + "..."

        head_budget = self.MAX_CODE_LEN * 2 // 3
        tail_budget = self.MAX_CODE_LEN // 3

        head = result[:head_budget]
        tail = result[-tail_budget:]

        if key_text and key_text not in head and key_text not in tail:
            return head + "\n...(中间输出已省略)...\n" + tail + f"\n[关键信息] {key_text}"

        return head + "\n...(中间输出已省略)...\n" + tail

    def _compress_weather(self, result: str) -> str:
        """压缩天气查询结果 — 保留每个城市的关键数据行"""
        lines = result.split("\n")
        kept_lines: list[str] = []
        budget = self.MAX_CODE_LEN

        for line in lines:
            stripped = line.strip()
            # 保留的关键行：城市标题、实况、预报日期行、警告
            is_key = bool(stripped) and (
                stripped.startswith("【")           # 城市标题
                or stripped.startswith("实况:")      # 实况数据
                or stripped.startswith("预报:")      # 预报标题
                or re.match(r'\d{4}-\d{2}-\d{2}', stripped)  # 日期预报行
                or stripped.startswith("⚠")         # 警告
                or stripped.startswith("输出:")       # 执行状态
                or stripped.startswith("执行成功")    # 执行状态
            )
            if is_key:
                kept_lines.append(line)

        compressed = "\n".join(kept_lines)
        if len(compressed) <= budget:
            return compressed

        # 如果还是太长，截断预报只保留前2天
        final_lines: list[str] = []
        date_count = 0
        for line in kept_lines:
            if re.match(r'\s*\d{4}-\d{2}-\d{2}', line.strip()):
                date_count += 1
                if date_count > 2:
                    continue
            final_lines.append(line)

        compressed = "\n".join(final_lines)
        if len(compressed) <= budget:
            return compressed

        return compressed[:budget] + "..."

    def _compress_general(self, result: str) -> str:
        if len(result) <= self.MAX_GENERAL_LEN:
            return result

        key_sentences = self._protector.extract_key_sentences(result, max_sentences=4)
        key_budget = self.MAX_GENERAL_LEN // 3
        key_text = "\n".join(key_sentences)
        if len(key_text) > key_budget:
            key_text = key_text[:key_budget] + "..."

        budget = self.MAX_GENERAL_LEN - len(key_text) - 80
        if budget > 200:
            head = result[:budget]
            return head + f"\n\n[关键信息] {key_text}\n...(结果已压缩，原始 {len(result)} 字符)"

        return f"[关键信息] {key_text}\n...(结果已压缩，原始 {len(result)} 字符)"

    async def _semantic_summarize(self, text: str, max_chars: int) -> str | None:
        if not self._llm_chat:
            return None

        prompt = (
            f"请将以下工具执行结果压缩为简洁摘要，保留所有关键信息（数字、日期、名称、错误信息）。\n"
            f"摘要长度不超过{max_chars}字符。\n\n"
            f"原始结果：\n{text[:3000]}"
        )

        try:
            import asyncio
            response = await asyncio.wait_for(
                self._llm_chat([{"role": "user", "content": prompt}], purpose="summarize"),
                timeout=30.0,
            )
            summary: str | None = response.content.strip() if response and response.content else None
            if summary and len(summary) <= max_chars:
                return str(summary)
            return None
        except Exception as e:
            logger.debug("LLM 语义摘要失败: %s", e)
            return None
