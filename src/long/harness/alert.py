"""告警闭环 — 内建告警规则 + 控制台/日志输出

Harness Engineering 原则：可观测性（Observability）
从"采集+日志"升级到"采集+日志+告警+行动"：
- 超时率 > 20% 触发告警
- Token 消耗接近上限触发告警
- 连续失败触发告警

设计约束：
- 零外部依赖，控制台 + 结构化日志输出
- 防抖：同类告警 60s 内不重复
- 不要告警洪水：相同根因的告警聚合
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(str, Enum):
    TIMEOUT_RATE = "timeout_rate"
    TOKEN_BUDGET = "token_budget"
    CONSECUTIVE_FAILURES = "consecutive_failures"
    LLM_ERROR = "llm_error"
    SANDBOX_FAILURE = "sandbox_failure"
    PII_DETECTED = "pii_detected"
    SENSITIVE_WORD = "sensitive_word"


@dataclass
class AlertRule:
    alert_type: AlertType
    level: AlertLevel = AlertLevel.WARNING
    threshold: float = 0.0
    cooldown_seconds: float = 60.0
    message_template: str = ""


@dataclass
class AlertEvent:
    alert_type: AlertType
    level: AlertLevel
    message: str
    timestamp: float = field(default_factory=time.monotonic)
    metadata: dict = field(default_factory=dict)


class AlertManager:
    """告警管理器

    用法：
        alert_mgr = AlertManager(default_rules=True)
        alert_mgr.check("timeout_rate", current_value=0.25)
        alert_mgr.trigger(AlertType.LLM_ERROR, "LLM API 返回 500")
    """

    def __init__(self, default_rules: bool = True) -> None:
        self._rules: dict[AlertType, AlertRule] = {}
        self._last_triggered: dict[AlertType, float] = {}
        self._history: list[AlertEvent] = []
        self._window_stats: dict[str, list[float]] = {}
        self._max_history = 100
        self._max_window_size = 50

        if default_rules:
            self._init_default_rules()

    def _init_default_rules(self) -> None:
        self.add_rule(AlertRule(
            alert_type=AlertType.TIMEOUT_RATE,
            level=AlertLevel.WARNING,
            threshold=0.5,
            cooldown_seconds=120.0,
            message_template="LLM 超时率 {value:.0%}，超过阈值 {threshold:.0%}",
        ))
        self.add_rule(AlertRule(
            alert_type=AlertType.TOKEN_BUDGET,
            level=AlertLevel.WARNING,
            threshold=0.8,
            cooldown_seconds=300.0,
            message_template="Token 消耗已达预算的 {value:.0%}",
        ))
        self.add_rule(AlertRule(
            alert_type=AlertType.CONSECUTIVE_FAILURES,
            level=AlertLevel.CRITICAL,
            threshold=3,
            cooldown_seconds=60.0,
            message_template="连续 {value:.0f} 次 LLM 调用失败",
        ))
        self.add_rule(AlertRule(
            alert_type=AlertType.SANDBOX_FAILURE,
            level=AlertLevel.WARNING,
            threshold=3,
            cooldown_seconds=120.0,
            message_template="沙箱执行连续 {value:.0f} 次失败",
        ))

    def add_rule(self, rule: AlertRule) -> None:
        self._rules[rule.alert_type] = rule

    def record(self, stat_name: str, value: float) -> None:
        """记录统计数据（用于滑动窗口计算）"""
        if stat_name not in self._window_stats:
            self._window_stats[stat_name] = []
        self._window_stats[stat_name].append(time.monotonic())
        if len(self._window_stats[stat_name]) > self._max_window_size:
            self._window_stats[stat_name].pop(0)

    def get_rate(self, stat_name: str, reference_name: str) -> float:
        """计算比率"""
        numerator = len(self._window_stats.get(stat_name, []))
        denominator = len(self._window_stats.get(reference_name, []))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _should_trigger(self, alert_type: AlertType) -> bool:
        now = time.monotonic()
        last = self._last_triggered.get(alert_type, 0)
        rule = self._rules.get(alert_type)
        if rule is None:
            return True
        return (now - last) >= rule.cooldown_seconds

    def check(self, alert_type: str, value: float) -> bool:
        """检查阈值并触发告警"""
        try:
            at = AlertType(alert_type)
        except ValueError:
            return False

        rule = self._rules.get(at)
        if rule is None:
            return False

        if value < rule.threshold:
            return False

        if not self._should_trigger(at):
            return False

        message = rule.message_template.format(value=value, threshold=rule.threshold)
        self._trigger_event(AlertEvent(
            alert_type=at,
            level=rule.level,
            message=message,
            metadata={"value": value, "threshold": rule.threshold},
        ))
        return True

    def trigger(self, alert_type: AlertType, message: str, level: AlertLevel | None = None, metadata: dict | None = None) -> None:
        """手动触发告警"""
        if not self._should_trigger(alert_type):
            return

        actual_level = level
        if actual_level is None:
            rule = self._rules.get(alert_type)
            actual_level = rule.level if rule else AlertLevel.WARNING

        self._trigger_event(AlertEvent(
            alert_type=alert_type,
            level=actual_level,
            message=message,
            metadata=metadata or {},
        ))

    def _trigger_event(self, event: AlertEvent) -> None:
        self._last_triggered[event.alert_type] = time.monotonic()
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        prefix_map = {
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.ERROR: "❌",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.INFO: "ℹ️",
        }
        prefix = prefix_map.get(event.level, "")

        log_method = {
            AlertLevel.CRITICAL: logger.critical,
            AlertLevel.ERROR: logger.error,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.INFO: logger.info,
        }.get(event.level, logger.warning)

        log_method(
            "%s [%s] %s (metadata=%s)",
            prefix, event.alert_type.value, event.message, event.metadata,
        )

    def get_history(self, limit: int = 20) -> list[AlertEvent]:
        return list(reversed(self._history[-limit:]))

    def clear_history(self) -> None:
        self._history.clear()
        self._last_triggered.clear()

    def collect_metrics_alert(self, llm_call_total: int, llm_call_timeout: int, total_tokens: int, budget_tokens: int) -> None:
        """批量检查多个指标"""
        if llm_call_total > 0:
            timeout_rate = llm_call_timeout / llm_call_total
            self.check("timeout_rate", timeout_rate)

        if budget_tokens > 0:
            budget_usage = total_tokens / budget_tokens
            self.check("token_budget", budget_usage)