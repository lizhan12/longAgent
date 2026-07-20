from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class EvidenceItem:
    source: str
    content: str
    relevance: float = 0.5
    has_specific_data: bool = False


@dataclass
class KnowledgeGap:
    description: str
    priority: float = 0.5
    search_queries: list[str] = field(default_factory=list)


@dataclass
class SufficiencyReport:
    is_sufficient: bool
    confidence: float
    gaps: list[KnowledgeGap]
    evidence_count: int
    reason: str = ""


@runtime_checkable
class CognitiveState(Protocol):
    @property
    def confidence(self) -> float: ...

    @property
    def knowledge_gaps(self) -> list[KnowledgeGap]: ...

    @property
    def evidence(self) -> list[EvidenceItem]: ...

    def update_from_tool_result(self, tool_name: str, result: str, query: str = "") -> None: ...

    def should_search_more(self) -> bool: ...

    def should_proceed_to_output(self) -> bool: ...

    def get_sufficiency_report(self) -> SufficiencyReport: ...

    def reset(self) -> None: ...
