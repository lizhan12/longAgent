from __future__ import annotations

import re
from typing import Any

INVARIANT_SAFETY_RULES = [
    "delete_file 始终需要确认",
    "execute_code 始终需要 AST 扫描",
    "path 始终不允许 .. 或 /etc/",
    "token 预算始终有上限",
    "总轮次始终有上限",
]

_DANGEROUS_PATH_PATTERNS = [
    re.compile(r'\.\.'),
    re.compile(r'^/etc/'),
    re.compile(r'^/root/'),
    re.compile(r'^/var/'),
    re.compile(r'^/sys/'),
    re.compile(r'^/proc/'),
]

_DANGEROUS_CODE_PATTERNS = [
    re.compile(r'os\.system\s*\('),
    re.compile(r'subprocess\.(call|run|Popen)\s*\('),
    re.compile(r'eval\s*\('),
    re.compile(r'exec\s*\('),
    re.compile(r'__import__\s*\('),
    re.compile(r'shutil\.rmtree\s*\('),
    re.compile(r'os\.remove\s*\('),
]

_PROMPT_INJECTION_PATTERNS = [
    re.compile(r'ignore\s+previous\s+instructions?', re.IGNORECASE),
    re.compile(r'forget\s+(?:all\s+)?(?:previous\s+)?instructions?', re.IGNORECASE),
    re.compile(r'you\s+are\s+now\s+(?:a|an)\s+', re.IGNORECASE),
    re.compile(r'system\s*:\s*', re.IGNORECASE),
    re.compile(r'<\|im_start\|>system', re.IGNORECASE),
]


class SafetyBoundary:
    """Invariant safety rules that cannot be overridden by any execution mode"""

    @staticmethod
    def check_path_safety(path: str) -> tuple[bool, str]:
        for pattern in _DANGEROUS_PATH_PATTERNS:
            if pattern.search(path):
                return False, f"路径包含危险模式: {pattern.pattern}"
        return True, ""

    @staticmethod
    def check_code_safety(code: str) -> tuple[bool, str]:
        for pattern in _DANGEROUS_CODE_PATTERNS:
            if pattern.search(code):
                return False, f"代码包含危险调用: {pattern.pattern}"
        return True, ""

    @staticmethod
    def check_prompt_injection(text: str) -> tuple[bool, str]:
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(text):
                return True, f"检测到潜在 prompt injection: {pattern.pattern}"
        return False, ""

    @staticmethod
    def sanitize_tool_result(result: str) -> str:
        is_injection, pattern = SafetyBoundary.check_prompt_injection(result)
        if is_injection:
            return f"[FILTERED: potential prompt injection detected - {pattern}]"
        return result

    @staticmethod
    def get_invariant_rules() -> list[str]:
        return list(INVARIANT_SAFETY_RULES)
