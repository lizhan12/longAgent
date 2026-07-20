"""系统健康度仪表盘

提供 CLI 命令 `long health` 显示系统运行状态。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from long.circuit_breaker import circuit_registry, CircuitState
from long.optimization.collector import MetricsCollector
from long.observability.tracing import Tracer
from long.retry import retry_registry

logger = logging.getLogger(__name__)


class HealthDashboard:
    """系统健康度仪表盘"""

    def __init__(
        self,
        collector: MetricsCollector | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._collector = collector
        self._tracer = tracer

    def get_overview(self) -> dict[str, Any]:
        """获取系统概览"""
        now = time.time()
        since = now - 3600

        overview: dict[str, Any] = {
            "timestamp": now,
            "llm": self._get_llm_status(since),
            "tools": self._get_tool_status(since),
            "retry": self._get_retry_status(),
            "circuit_breakers": self._get_circuit_breaker_status(),
            "memory": self._get_memory_status(),
            "optimization": self._get_optimization_status(),
        }

        return overview

    def _get_llm_status(self, since: float) -> dict[str, Any]:
        if self._collector is None:
            return {"status": "unknown"}

        success_agg = self._collector.get_aggregation("llm.success", since=since)
        latency_agg = self._collector.get_aggregation("llm.latency_ms", since=since)
        prompt_agg = self._collector.get_aggregation("llm.prompt_tokens", since=since)
        completion_agg = self._collector.get_aggregation("llm.completion_tokens", since=since)

        success_rate = success_agg.get("mean", 0)
        status = "healthy"
        if success_rate < 0.5:
            status = "critical"
        elif success_rate < 0.8:
            status = "degraded"

        return {
            "status": status,
            "success_rate": f"{success_rate * 100:.1f}%",
            "total_calls": int(success_agg.get("count", 0)),
            "avg_latency_ms": f"{latency_agg.get('mean', 0):.0f}",
            "max_latency_ms": f"{latency_agg.get('max', 0):.0f}",
            "total_prompt_tokens": int(prompt_agg.get("count", 0)),
            "total_completion_tokens": int(completion_agg.get("count", 0)),
        }

    def _get_tool_status(self, since: float) -> dict[str, Any]:
        if self._collector is None:
            return {"status": "unknown"}

        success_agg = self._collector.get_aggregation("tool.success", since=since)
        latency_agg = self._collector.get_aggregation("tool.latency_ms", since=since)

        tool_metrics: dict[str, dict[str, Any]] = {}
        for name in self._collector.get_all_metric_names():
            if name.startswith("tool."):
                agg = self._collector.get_aggregation(name, since=since)
                if agg["count"] > 0:
                    tool_metrics[name] = agg

        return {
            "success_rate": f"{success_agg.get('mean', 0) * 100:.1f}%",
            "total_calls": int(success_agg.get("count", 0)),
            "avg_latency_ms": f"{latency_agg.get('mean', 0):.0f}",
            "metrics": tool_metrics,
        }

    def _get_retry_status(self) -> dict[str, Any]:
        summary = retry_registry.get_summary()
        return {
            "total_calls": summary.get("total_calls", 0),
            "total_retries": summary.get("total_retries", 0),
            "total_failures": summary.get("total_failures", 0),
            "by_function": summary.get("by_function", {}),
        }

    def _get_circuit_breaker_status(self) -> dict[str, Any]:
        all_stats = circuit_registry.get_all_stats()
        open_breakers = circuit_registry.get_open_breakers()

        return {
            "total": len(all_stats),
            "open": open_breakers,
            "details": {
                name: {
                    "state": stats.state.value,
                    "failure_count": stats.failure_count,
                    "failure_rate": f"{stats.failure_rate * 100:.1f}%",
                    "last_failure": stats.last_failure_reason[:100] if stats.last_failure_reason else "",
                }
                for name, stats in all_stats.items()
            },
        }

    def _get_memory_status(self) -> dict[str, Any]:
        return {
            "status": "active",
        }

    def _get_optimization_status(self) -> dict[str, Any]:
        return {
            "status": "active",
        }

    def format_report(self) -> str:
        """格式化健康报告为可读文本"""
        overview = self.get_overview()

        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  Long AI 系统健康报告")
        lines.append("=" * 60)

        llm = overview["llm"]
        lines.append(f"\n📡 LLM 状态: {llm.get('status', 'unknown')}")
        lines.append(f"   成功率: {llm.get('success_rate', 'N/A')}")
        lines.append(f"   调用次数: {llm.get('total_calls', 0)}")
        lines.append(f"   平均延迟: {llm.get('avg_latency_ms', 'N/A')} ms")
        lines.append(f"   最大延迟: {llm.get('max_latency_ms', 'N/A')} ms")

        tools = overview["tools"]
        lines.append(f"\n🔧 工具状态:")
        lines.append(f"   成功率: {tools.get('success_rate', 'N/A')}")
        lines.append(f"   调用次数: {tools.get('total_calls', 0)}")
        lines.append(f"   平均延迟: {tools.get('avg_latency_ms', 'N/A')} ms")

        retry = overview["retry"]
        lines.append(f"\n🔄 重试统计:")
        lines.append(f"   总调用: {retry.get('total_calls', 0)}")
        lines.append(f"   总重试: {retry.get('total_retries', 0)}")
        lines.append(f"   总失败: {retry.get('total_failures', 0)}")
        for fn_name, fn_stats in retry.get("by_function", {}).items():
            lines.append(f"   - {fn_name}: 成功率 {fn_stats.get('success_rate', 0) * 100:.1f}%, 平均重试 {fn_stats.get('avg_retries', 0):.1f}")

        cb = overview["circuit_breakers"]
        lines.append(f"\n⚡ 熔断器:")
        lines.append(f"   总数: {cb.get('total', 0)}, 断开: {len(cb.get('open', []))}")
        for name, details in cb.get("details", {}).items():
            state_icon = "🔴" if details["state"] == "open" else "🟢" if details["state"] == "closed" else "🟡"
            lines.append(f"   {state_icon} {name}: {details['state']} (失败率 {details['failure_rate']})")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
