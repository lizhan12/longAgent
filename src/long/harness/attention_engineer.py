"""上下文注意力工程 — 给关键判断留干净带宽

Harness Engineering 原则：注意力是最稀缺算力（Context Engineering）
从"对话压缩"升级到"主动上下文管理"：
- 关键信息（约束、红线、当前目标）始终保留在上下文头部
- 冗余信息自动降级（完整内容 → 摘要 → 关键词 → 丢弃）
- 根据任务阶段动态调整上下文权重
- 与 compressor.py 集成，实现分层压缩策略

设计约束：
- 零外部依赖
- 与现有 DialogCompressor 兼容
- 降级策略可配置
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ContextPriority(str, Enum):
    """上下文优先级"""
    CRITICAL = "critical"   # 始终保留（约束、红线、当前目标）
    HIGH = "high"           # 尽量保留（最近工具结果、用户指令）
    MEDIUM = "medium"       # 可压缩（历史工具结果、中间步骤）
    LOW = "low"             # 可丢弃（早期对话、重复信息）


class CompressionLevel(str, Enum):
    """压缩级别"""
    FULL = "full"           # 完整内容
    SUMMARY = "summary"     # 摘要
    KEYWORDS = "keywords"   # 关键词
    DISCARD = "discard"     # 丢弃


@dataclass
class ContextSlot:
    """上下文槽位"""
    slot_id: str = ""
    priority: ContextPriority = ContextPriority.MEDIUM
    compression: CompressionLevel = CompressionLevel.FULL
    content: str = ""
    compressed_content: str = ""
    token_estimate: int = 0
    category: str = ""  # system / constraint / tool_result / user_msg / history
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0


@dataclass
class AttentionConfig:
    """注意力工程配置"""
    max_context_tokens: int = 12000
    critical_reserve_ratio: float = 0.3   # CRITICAL 保留 30% 带宽
    high_reserve_ratio: float = 0.4       # HIGH 保留 40% 带宽
    compression_threshold: float = 0.8    # 上下文使用率超过 80% 时触发压缩
    discard_threshold: float = 0.95       # 超过 95% 时开始丢弃 LOW 内容


class ContextAttentionEngineer:
    """上下文注意力工程师

    主动管理上下文窗口的内容优先级和压缩策略。

    用法：
        engine = ContextAttentionEngineer(AttentionConfig())
        engine.add_slot("constraint_1", "不可删除 /etc/ 目录下的文件", priority="critical", category="constraint")
        engine.add_slot("tool_result_1", "文件内容: ...", priority="medium", category="tool_result")

        # 获取优化后的上下文
        context = engine.build_context()
        print(f"总 token 估计: {engine.total_token_estimate}")
    """

    def __init__(self, config: AttentionConfig | None = None) -> None:
        self._config = config or AttentionConfig()
        self._slots: dict[str, ContextSlot] = {}

    @property
    def config(self) -> AttentionConfig:
        return self._config

    def add_slot(
        self,
        slot_id: str,
        content: str,
        priority: ContextPriority | str = ContextPriority.MEDIUM,
        category: str = "",
    ) -> ContextSlot:
        """添加上下文槽位"""
        if isinstance(priority, str):
            priority = ContextPriority(priority)

        token_estimate = self._estimate_tokens(content)
        slot = ContextSlot(
            slot_id=slot_id,
            priority=priority,
            content=content,
            token_estimate=token_estimate,
            category=category,
        )
        self._slots[slot_id] = slot
        return slot

    def update_slot(self, slot_id: str, content: str) -> None:
        """更新槽位内容"""
        slot = self._slots.get(slot_id)
        if slot is not None:
            slot.content = content
            slot.token_estimate = self._estimate_tokens(content)
            slot.last_accessed = time.time()
            slot.access_count += 1

    def remove_slot(self, slot_id: str) -> None:
        """移除槽位"""
        self._slots.pop(slot_id, None)

    def touch(self, slot_id: str) -> None:
        """标记槽位被访问（提升优先级）"""
        slot = self._slots.get(slot_id)
        if slot is not None:
            slot.last_accessed = time.time()
            slot.access_count += 1

    @property
    def total_token_estimate(self) -> int:
        return sum(s.token_estimate for s in self._slots.values() if s.compression != CompressionLevel.DISCARD)

    @property
    def usage_ratio(self) -> float:
        if self._config.max_context_tokens == 0:
            return 0.0
        return self.total_token_estimate / self._config.max_context_tokens

    def build_context(self) -> list[dict[str, str]]:
        """构建优化后的上下文消息列表

        策略：
        1. 检查使用率，超过阈值时触发压缩
        2. 按优先级排序：CRITICAL → HIGH → MEDIUM → LOW
        3. 同优先级内按访问时间排序（最近优先）
        4. 依次填充直到达到 token 上限
        """
        # 检查是否需要压缩
        if self.usage_ratio > self._config.compression_threshold:
            self._compress()

        # 按优先级排序
        priority_order = {
            ContextPriority.CRITICAL: 0,
            ContextPriority.HIGH: 1,
            ContextPriority.MEDIUM: 2,
            ContextPriority.LOW: 3,
        }
        sorted_slots = sorted(
            [s for s in self._slots.values() if s.compression != CompressionLevel.DISCARD],
            key=lambda s: (priority_order.get(s.priority, 2), -s.last_accessed),
        )

        # 分配 token 预算
        budget = self._config.max_context_tokens
        critical_budget = int(budget * self._config.critical_reserve_ratio)
        high_budget = int(budget * self._config.high_reserve_ratio)
        remaining_budget = budget - critical_budget - high_budget

        messages: list[dict[str, str]] = []
        used_tokens = 0

        # 先放 CRITICAL
        for slot in sorted_slots:
            if slot.priority != ContextPriority.CRITICAL:
                continue
            content = self._get_content(slot)
            tokens = self._estimate_tokens(content)
            if used_tokens + tokens <= critical_budget:
                messages.append(self._slot_to_message(slot, content))
                used_tokens += tokens

        # 再放 HIGH
        high_used = 0
        for slot in sorted_slots:
            if slot.priority != ContextPriority.HIGH:
                continue
            content = self._get_content(slot)
            tokens = self._estimate_tokens(content)
            if high_used + tokens <= high_budget:
                messages.append(self._slot_to_message(slot, content))
                high_used += tokens

        # 最后放 MEDIUM 和 LOW
        medium_used = 0
        for slot in sorted_slots:
            if slot.priority not in (ContextPriority.MEDIUM, ContextPriority.LOW):
                continue
            content = self._get_content(slot)
            tokens = self._estimate_tokens(content)
            if medium_used + tokens <= remaining_budget:
                messages.append(self._slot_to_message(slot, content))
                medium_used += tokens

        return messages

    def _compress(self) -> None:
        """执行压缩策略"""
        # 阶段1: LOW 内容降级为关键词
        for slot in self._slots.values():
            if slot.priority == ContextPriority.LOW and slot.compression == CompressionLevel.FULL:
                slot.compression = CompressionLevel.KEYWORDS
                slot.compressed_content = self._extract_keywords(slot.content)
                slot.token_estimate = self._estimate_tokens(slot.compressed_content)
                logger.debug("压缩 [LOW→KEYWORDS]: %s", slot.slot_id)

        # 阶段2: MEDIUM 内容压缩为摘要
        if self.usage_ratio > self._config.compression_threshold:
            for slot in self._slots.values():
                if slot.priority == ContextPriority.MEDIUM and slot.compression == CompressionLevel.FULL:
                    slot.compression = CompressionLevel.SUMMARY
                    slot.compressed_content = self._summarize(slot.content)
                    slot.token_estimate = self._estimate_tokens(slot.compressed_content)
                    logger.debug("压缩 [MEDIUM→SUMMARY]: %s", slot.slot_id)

        # 阶段3: 超过丢弃阈值时丢弃 LOW 内容
        if self.usage_ratio > self._config.discard_threshold:
            for slot in self._slots.values():
                if slot.priority == ContextPriority.LOW:
                    slot.compression = CompressionLevel.DISCARD
                    slot.token_estimate = 0
                    logger.debug("丢弃 [LOW]: %s", slot.slot_id)

    def _get_content(self, slot: ContextSlot) -> str:
        """获取槽位的当前内容（优先压缩后的）"""
        if slot.compression == CompressionLevel.DISCARD:
            return ""
        if slot.compressed_content:
            return slot.compressed_content
        return slot.content

    def _slot_to_message(self, slot: ContextSlot, content: str) -> dict[str, str]:
        """将槽位转换为消息格式"""
        role_map = {
            "system": "system",
            "constraint": "system",
            "user_msg": "user",
            "tool_result": "user",
            "history": "user",
        }
        role = role_map.get(slot.category, "user")
        prefix = ""
        if slot.priority == ContextPriority.CRITICAL:
            prefix = "[重要] "
        elif slot.compression == CompressionLevel.SUMMARY:
            prefix = "[摘要] "
        elif slot.compression == CompressionLevel.KEYWORDS:
            prefix = "[关键词] "
        return {"role": role, "content": prefix + content}

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略估算 token 数（中文约 1.5 字/token，英文约 4 字符/token）"""
        if not text:
            return 0
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4) + 1

    @staticmethod
    def _extract_keywords(text: str, max_keywords: int = 10) -> str:
        """提取关键词（简单实现：取每行前几个词）"""
        if not text:
            return ""
        lines = text.strip().split("\n")
        keywords = []
        for line in lines[:max_keywords]:
            line = line.strip()
            if line:
                # 取每行前 30 个字符
                keywords.append(line[:30] + ("..." if len(line) > 30 else ""))
        return " | ".join(keywords)

    @staticmethod
    def _summarize(text: str, max_ratio: float = 0.3) -> str:
        """简单摘要（取前 30% 内容 + 最后几行）"""
        if not text:
            return ""
        lines = text.strip().split("\n")
        if len(lines) <= 3:
            return text

        head_count = max(1, int(len(lines) * max_ratio))
        head = "\n".join(lines[:head_count])
        tail = "\n".join(lines[-2:]) if len(lines) > head_count + 2 else ""

        if tail:
            return f"{head}\n... (省略 {len(lines) - head_count - 2} 行) ...\n{tail}"
        return head

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        by_priority: dict[str, int] = {}
        by_compression: dict[str, int] = {}
        for slot in self._slots.values():
            by_priority[slot.priority.value] = by_priority.get(slot.priority.value, 0) + 1
            by_compression[slot.compression.value] = by_compression.get(slot.compression.value, 0) + 1

        return {
            "total_slots": len(self._slots),
            "total_tokens": self.total_token_estimate,
            "max_tokens": self._config.max_context_tokens,
            "usage_ratio": round(self.usage_ratio, 3),
            "by_priority": by_priority,
            "by_compression": by_compression,
        }

    def reset(self) -> None:
        """重置所有槽位"""
        self._slots.clear()
