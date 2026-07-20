from __future__ import annotations
import re
from typing import Any
from .base import CognitiveState, EvidenceItem, KnowledgeGap, SufficiencyReport

_SPECIFIC_DATA_PATTERNS = [
    re.compile(r'\d+\.?\d*\s*[%％]'),
    re.compile(r'\d{4}[-/年]\d{1,2}[-/月]\d{0,2}[日号]?'),
    re.compile(r'\d+\.?\d*\s*(万|亿|元|美元|美元|kg|km|米|吨|GW|MW|TB|GB)'),
    re.compile(r'(增长|下降|上升|减少|同比|环比)\s*\d+\.?\d*'),
]

_ENTITY_PATTERNS = [
    re.compile(r'[\u4e00-\u9fff]{2,8}(?:省|市|区|县|镇)'),
    re.compile(r'[\u4e00-\u9fff]{2,6}(?:公司|集团|机构|部门|委员会)'),
    re.compile(r'[A-Z][a-zA-Z]+(?:\.com|\.cn|\.org)'),
]


class ConfidenceBasedState:
    """Confidence-based cognitive state

    Replaces fixed flow (search 3 times -> analyze -> output)
    with confidence-driven decisions:
    - confidence >= 0.8 -> can output
    - 0.3 <= confidence < 0.8 -> need more info
    - confidence < 0.3 -> need strategy change
    """

    def __init__(self, output_threshold: float = 0.8, search_threshold: float = 0.3):
        self._confidence: float = 0.0
        self._evidence: list[EvidenceItem] = []
        self._gaps: list[KnowledgeGap] = []
        self._output_threshold = output_threshold
        self._search_threshold = search_threshold
        self._query_entities: list[str] = []
        self._covered_entities: set[str] = set()

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def knowledge_gaps(self) -> list[KnowledgeGap]:
        return list(self._gaps)

    @property
    def evidence(self) -> list[EvidenceItem]:
        return list(self._evidence)

    def set_query_entities(self, entities: list[str]) -> None:
        self._query_entities = entities

    def update_from_tool_result(self, tool_name: str, result: str, query: str = "") -> None:
        if tool_name in ("tavily_search", "search"):
            self._update_from_search(result, query)
        elif tool_name in ("execute_code", "execute_file"):
            self._update_from_execution(result)
        elif tool_name in ("read_file", "list_files"):
            self._update_from_file_read(result)
        else:
            self._update_from_general(result)

    def _update_from_search(self, result: str, query: str = "") -> None:
        has_data = self._has_specific_data(result)
        relevance = self._estimate_relevance(result, query)

        self._evidence.append(EvidenceItem(
            source="search",
            content=result[:200],
            relevance=relevance,
            has_specific_data=has_data,
        ))

        if len(result) < 100:
            self._confidence = max(0.05, self._confidence - 0.15)
            self._gaps.append(KnowledgeGap(
                description=f"搜索结果过短（{len(result)}字符）",
                priority=0.7,
                search_queries=[query] if query else [],
            ))
        elif has_data and relevance > 0.5:
            self._confidence = min(1.0, self._confidence + 0.25)
            self._remove_resolved_gaps(query)
        elif has_data:
            self._confidence = min(1.0, self._confidence + 0.15)
        elif relevance > 0.5:
            self._confidence = min(1.0, self._confidence + 0.1)
        else:
            self._confidence = min(1.0, self._confidence + 0.05)

        self._update_entity_coverage(result)
        self._recalculate_confidence()

    def _update_from_execution(self, result: str) -> None:
        is_error = any(p in result.lower() for p in ["error", "traceback", "失败", "exception"])
        self._evidence.append(EvidenceItem(
            source="execution",
            content=result[:200],
            relevance=0.8 if not is_error else 0.2,
            has_specific_data=not is_error,
        ))
        if is_error:
            self._confidence = max(0.1, self._confidence - 0.1)
        else:
            self._confidence = min(1.0, self._confidence + 0.2)

    def _update_from_file_read(self, result: str) -> None:
        self._evidence.append(EvidenceItem(
            source="file",
            content=result[:200],
            relevance=0.7,
            has_specific_data=True,
        ))
        self._confidence = min(1.0, self._confidence + 0.15)

    def _update_from_general(self, result: str) -> None:
        self._evidence.append(EvidenceItem(
            source="tool",
            content=result[:200],
            relevance=0.5,
            has_specific_data=False,
        ))
        self._confidence = min(1.0, self._confidence + 0.05)

    def _has_specific_data(self, text: str) -> bool:
        return any(p.search(text) for p in _SPECIFIC_DATA_PATTERNS)

    def _estimate_relevance(self, text: str, query: str) -> float:
        if not query:
            return 0.5
        query_words = set(query.lower().split())
        text_lower = text.lower()
        matched = sum(1 for w in query_words if w in text_lower)
        return min(1.0, matched / max(len(query_words), 1))

    def _update_entity_coverage(self, text: str) -> None:
        for pattern in _ENTITY_PATTERNS:
            for match in pattern.finditer(text):
                entity = match.group()
                if entity in self._query_entities:
                    self._covered_entities.add(entity)

    def _remove_resolved_gaps(self, query: str) -> None:
        if not query:
            return
        query_lower = query.lower()
        self._gaps = [g for g in self._gaps if query_lower not in g.description.lower()]

    def _recalculate_confidence(self) -> None:
        if self._query_entities:
            coverage = len(self._covered_entities) / max(len(self._query_entities), 1)
            self._confidence = self._confidence * 0.7 + coverage * 0.3

        high_relevance_evidence = [e for e in self._evidence if e.relevance > 0.5 and e.has_specific_data]
        if len(high_relevance_evidence) >= 3:
            self._confidence = min(1.0, self._confidence + 0.1)

    def should_search_more(self) -> bool:
        return self._confidence < self._output_threshold and len(self._gaps) > 0

    def should_proceed_to_output(self) -> bool:
        return self._confidence >= self._output_threshold

    def get_sufficiency_report(self) -> SufficiencyReport:
        return SufficiencyReport(
            is_sufficient=self._confidence >= self._output_threshold,
            confidence=self._confidence,
            gaps=list(self._gaps),
            evidence_count=len(self._evidence),
            reason=self._generate_reason(),
        )

    def _generate_reason(self) -> str:
        if self._confidence >= self._output_threshold:
            return f"信心度 {self._confidence:.0%} >= {self._output_threshold:.0%}，可以输出"
        if self._confidence < self._search_threshold:
            return f"信心度 {self._confidence:.0%} < {self._search_threshold:.0%}，需要换策略"
        parts = [f"信心度 {self._confidence:.0%}"]
        if self._gaps:
            parts.append(f"有 {len(self._gaps)} 个信息缺口")
        if self._query_entities and self._covered_entities:
            coverage = len(self._covered_entities) / max(len(self._query_entities), 1)
            parts.append(f"实体覆盖率 {coverage:.0%}")
        return "，".join(parts) + "，需要更多信息"

    def reset(self) -> None:
        self._confidence = 0.0
        self._evidence = []
        self._gaps = []
        self._query_entities = []
        self._covered_entities = set()
