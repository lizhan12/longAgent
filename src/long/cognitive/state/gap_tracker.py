from __future__ import annotations
import re
from typing import Any
from .base import KnowledgeGap


class KnowledgeGapTracker:
    """Tracks knowledge gaps from user queries and tool results"""

    _QUESTION_PATTERNS = [
        re.compile(r'(什么|如何|怎么|为什么|哪些|哪个|多少|是否|能否|会不会)'),
        re.compile(r'(趋势|对比|分析|评估|影响|原因|方案|建议|规划)'),
    ]

    _INTENT_KEYWORDS = {
        "chart": ["图表", "折线图", "柱状图", "饼图", "可视化", "趋势图"],
        "report": ["报告", "分析", "总结", "调研", "综述"],
        "code": ["代码", "脚本", "程序", "接口", "函数"],
        "comparison": ["对比", "比较", "区别", "差异", "优劣"],
        "data": ["数据", "统计", "指标", "排名", "增长率"],
    }

    def __init__(self):
        self._gaps: list[KnowledgeGap] = []
        self._detected_intents: list[str] = []

    def analyze_query(self, query: str) -> list[KnowledgeGap]:
        gaps = []

        for pattern in self._QUESTION_PATTERNS:
            if pattern.search(query):
                gaps.append(KnowledgeGap(
                    description=f"用户问题包含疑问词: {pattern.pattern}",
                    priority=0.6,
                    search_queries=[query],
                ))
                break

        for intent, keywords in self._INTENT_KEYWORDS.items():
            if any(kw in query for kw in keywords):
                self._detected_intents.append(intent)
                if intent in ("chart", "report", "comparison"):
                    gaps.append(KnowledgeGap(
                        description=f"用户需要{intent}，需要收集数据",
                        priority=0.8,
                        search_queries=[query],
                    ))

        self._gaps.extend(gaps)
        return gaps

    @property
    def detected_intents(self) -> list[str]:
        return list(self._detected_intents)

    @property
    def gaps(self) -> list[KnowledgeGap]:
        return list(self._gaps)

    def mark_resolved(self, gap_description: str) -> None:
        self._gaps = [g for g in self._gaps if gap_description.lower() not in g.description.lower()]

    def has_intent(self, intent: str) -> bool:
        return intent in self._detected_intents

    def reset(self) -> None:
        self._gaps = []
        self._detected_intents = []
