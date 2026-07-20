"""输出安全治理 — PII 检测 + 敏感词过滤

Harness Engineering 原则：输出治理（Output Governance）
在 LLM 输出返回给用户之前做后置校验：
- PII 检测：中国身份证号、手机号、邮箱（本地正则，零延迟）
- 敏感词过滤：可配置的敏感词列表
- 可选：接入外部内容审核 API

设计约束：
- 本地正则必须快（<1ms），不能影响响应延迟
- 只做检测+标记，不做自动删除（保留用户知情权）
- PII 检测覆盖：写入文件的内容也做检查
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Pattern

logger = logging.getLogger(__name__)

_CHINA_ID_PATTERN = re.compile(r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b")
_CHINA_PHONE_PATTERN = re.compile(r"\b1[3-9]\d{9}\b")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_BANK_CARD_PATTERN = re.compile(r"\b\d{16,19}\b")

_PII_PATTERNS: dict[str, Pattern[str]] = {
    "china_id": _CHINA_ID_PATTERN,
    "china_phone": _CHINA_PHONE_PATTERN,
    "email": _EMAIL_PATTERN,
    "ip": _IP_PATTERN,
    "bank_card": _BANK_CARD_PATTERN,
}

_DEFAULT_SENSITIVE_WORDS = [
    "password", "secret", "token", "api_key",
    "access_key", "private_key",
]


@dataclass
class PIIMatch:
    type: str
    value: str
    start: int
    end: int


@dataclass
class OutputGuardResult:
    passed: bool
    pii_matches: list[PIIMatch] = field(default_factory=list)
    sensitive_matches: list[str] = field(default_factory=list)
    warning: str = ""


@dataclass
class OutputGuardConfig:
    enabled: bool = True
    pii_patterns: list[str] = field(default_factory=lambda: list(_PII_PATTERNS.keys()))
    sensitive_words: list[str] = field(default_factory=lambda: list(_DEFAULT_SENSITIVE_WORDS))
    block_on_pii: bool = False
    block_on_sensitive: bool = False


class OutputGuard:
    """输出安全守卫

    用法：
        guard = OutputGuard(OutputGuardConfig())
        result = guard.scan(llm_output_text)
        if not result.passed:
            logger.warning("输出包含敏感信息: %s", result.warning)
    """

    def __init__(self, config: OutputGuardConfig | None = None) -> None:
        self._config = config or OutputGuardConfig()
        self._patterns: dict[str, Pattern[str]] = {
            k: v for k, v in _PII_PATTERNS.items()
            if k in self._config.pii_patterns
        }
        self._sensitive_lowered = [w.lower() for w in self._config.sensitive_words]

    def scan(self, text: str) -> OutputGuardResult:
        """扫描文本中的敏感信息"""
        if not self._config.enabled or not text:
            return OutputGuardResult(passed=True)

        pii_matches: list[PIIMatch] = []
        for pii_type, pattern in self._patterns.items():
            for match in pattern.finditer(text):
                pii_matches.append(PIIMatch(
                    type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                ))

        sensitive_matches: list[str] = []
        text_lower = text.lower()
        for word in self._sensitive_lowered:
            if word in text_lower:
                sensitive_matches.append(word)

        passed = True
        warning_parts: list[str] = []

        if pii_matches and self._config.block_on_pii:
            passed = False
            types = set(m.type for m in pii_matches)
            warning_parts.append(f"检测到 PII ({', '.join(sorted(types))})，共 {len(pii_matches)} 处")
        elif pii_matches:
            types = set(m.type for m in pii_matches)
            warning_parts.append(f"PII 标记: ({', '.join(sorted(types))})，共 {len(pii_matches)} 处")

        if sensitive_matches and self._config.block_on_sensitive:
            passed = False
            warning_parts.append(f"检测到敏感词: {', '.join(sensitive_matches)}")
        elif sensitive_matches:
            warning_parts.append(f"敏感词标记: {', '.join(sensitive_matches)}")

        return OutputGuardResult(
            passed=passed,
            pii_matches=pii_matches,
            sensitive_matches=sensitive_matches,
            warning="; ".join(warning_parts),
        )

    def mask_text(self, text: str) -> str:
        """遮蔽文本中的 PII"""
        if not self._config.enabled or not text:
            return text

        result = text
        offset = 0
        for pii_type, pattern in self._patterns.items():
            for match in pattern.finditer(result):
                replacement = self._mask_value(pii_type, match.group())
                adjusted_start = match.start() - offset
                adjusted_end = match.end() - offset
                result = result[:adjusted_start] + replacement + result[adjusted_end:]
                offset += (match.end() - match.start()) - len(replacement)

        return result

    @staticmethod
    def _mask_value(pii_type: str, value: str) -> str:
        if pii_type == "china_phone":
            return value[:3] + "****" + value[-4:]
        if pii_type == "email":
            parts = value.split("@")
            if len(parts) == 2 and len(parts[0]) > 2:
                parts[0] = parts[0][:2] + "***"
            return "@".join(parts)
        if len(value) > 4:
            return value[:2] + "*" * (len(value) - 4) + value[-2:]
        return "***"