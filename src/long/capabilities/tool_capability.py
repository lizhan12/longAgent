"""Tool Capability — 工具能力建模

为每个工具建立能力模型，让 Planner 做出基于能力的工具选择。

核心思想：
  旧：LLM 盲目选工具
  新：Planner 基于 capability_tags / latency / reliability 选工具
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolCapability(BaseModel):
    """工具能力模型"""

    latency_p50: float = 1.0
    latency_p95: float = 5.0
    cost_per_call: float = 0.0
    reliability: float = 0.95
    capability_tags: list[str] = Field(default_factory=list)
    coverage_domains: list[str] = Field(default_factory=list)
    side_effects: list[str] = Field(default_factory=list)
    max_output_tokens: int = 4000

    model_config = {"extra": "forbid"}


class ToolStats(BaseModel):
    """工具运行时统计"""

    call_count: int = 0
    success_count: int = 0
    total_latency: float = 0.0
    last_error: str | None = None
    last_call_time: float = 0.0

    model_config = {"extra": "forbid"}

    def record_call(self, success: bool, latency: float, error: str | None = None) -> None:
        self.call_count += 1
        if success:
            self.success_count += 1
        self.total_latency += latency
        self.last_call_time = time.time()
        if error:
            self.last_error = error

    @property
    def success_rate(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count

    @property
    def avg_latency(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_latency / self.call_count


DEFAULT_CAPABILITIES: dict[str, dict[str, Any]] = {
    "tavily_search": {
        "latency_p50": 2.0,
        "latency_p95": 8.0,
        "capability_tags": ["search", "information_retrieval"],
        "coverage_domains": ["general", "news", "weather", "finance"],
        "side_effects": ["network_call"],
        "reliability": 0.85,
    },
    "execute_code": {
        "latency_p50": 3.0,
        "latency_p95": 15.0,
        "capability_tags": ["code_exec", "data_viz", "computation"],
        "coverage_domains": ["code", "math", "data", "visualization"],
        "side_effects": ["compute"],
        "reliability": 0.80,
    },
    "execute_file": {
        "latency_p50": 2.0,
        "latency_p95": 10.0,
        "capability_tags": ["code_exec", "file_ops"],
        "coverage_domains": ["code", "scripts"],
        "side_effects": ["compute", "filesystem_write"],
        "reliability": 0.85,
    },
    "write_file": {
        "latency_p50": 0.5,
        "latency_p95": 2.0,
        "capability_tags": ["file_ops", "output"],
        "coverage_domains": ["documents", "reports", "code"],
        "side_effects": ["filesystem_write"],
        "reliability": 0.98,
    },
    "read_file": {
        "latency_p50": 0.3,
        "latency_p95": 1.0,
        "capability_tags": ["file_ops", "information_retrieval"],
        "coverage_domains": ["documents", "code", "data"],
        "side_effects": [],
        "reliability": 0.99,
    },
    "delete_file": {
        "latency_p50": 0.2,
        "latency_p95": 0.5,
        "capability_tags": ["file_ops"],
        "side_effects": ["filesystem_delete"],
        "reliability": 0.99,
    },
}


class ToolCapabilityRegistry:
    """工具能力注册表

    管理工具的能力模型和运行时统计，
    提供基于能力的工具推荐。
    """

    def __init__(self) -> None:
        self._capabilities: dict[str, ToolCapability] = {}
        self._stats: dict[str, ToolStats] = {}

        for name, cap_dict in DEFAULT_CAPABILITIES.items():
            self._capabilities[name] = ToolCapability(**cap_dict)
            self._stats[name] = ToolStats()

    def register_capability(self, tool_name: str, capability: ToolCapability) -> None:
        self._capabilities[tool_name] = capability
        if tool_name not in self._stats:
            self._stats[tool_name] = ToolStats()

    def get_capability(self, tool_name: str) -> ToolCapability | None:
        return self._capabilities.get(tool_name)

    def get_stats(self, tool_name: str) -> ToolStats | None:
        return self._stats.get(tool_name)

    def record_call(self, tool_name: str, success: bool, latency: float, error: str | None = None) -> None:
        if tool_name not in self._stats:
            self._stats[tool_name] = ToolStats()
        self._stats[tool_name].record_call(success, latency, error)

        cap = self._capabilities.get(tool_name)
        if cap and self._stats[tool_name].call_count >= 5:
            stats = self._stats[tool_name]
            cap.reliability = stats.success_rate
            cap.latency_p50 = stats.avg_latency

    def recommend_tools(
        self,
        task_tags: list[str],
        max_latency: float | None = None,
        min_reliability: float = 0.5,
    ) -> list[str]:
        """基于能力标签推荐工具"""
        candidates = []
        for name, cap in self._capabilities.items():
            tag_match = any(t in cap.capability_tags for t in task_tags)
            if not tag_match:
                continue
            latency_ok = max_latency is None or cap.latency_p95 <= max_latency
            if not latency_ok:
                continue
            reliability_ok = cap.reliability >= min_reliability
            if not reliability_ok:
                continue
            candidates.append(name)

        candidates.sort(
            key=lambda n: self._capabilities[n].reliability,
            reverse=True,
        )
        return candidates

    def infer_tool_hint(self, description: str) -> str | None:
        """从子任务描述推断建议工具"""
        search_kw = ("搜索", "查询", "查找", "了解", "调研", "search", "query")
        code_kw = ("代码", "执行", "运行", "计算", "图表", "可视化", "折线", "code", "execute", "chart", "plot")
        file_kw = ("文件", "保存", "写入", "报告", "导出", "file", "save", "write", "report")

        if any(kw in description for kw in search_kw):
            tools = self.recommend_tools(["search"])
            return tools[0] if tools else "tavily_search"
        if any(kw in description for kw in code_kw):
            tools = self.recommend_tools(["code_exec", "data_viz"])
            return tools[0] if tools else "execute_code"
        if any(kw in description for kw in file_kw):
            tools = self.recommend_tools(["file_ops", "output"])
            return tools[0] if tools else "write_file"

        return None
