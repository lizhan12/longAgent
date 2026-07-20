"""代码预扫描

在沙箱执行前扫描代码中的恶意模式。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ThreatLevel(str, Enum):
    """威胁级别"""

    SAFE = "safe"
    WARNING = "warning"
    DANGEROUS = "dangerous"


@dataclass
class ScanResult:
    """扫描结果"""

    safe: bool = True
    threats: list[dict[str, Any]] = field(default_factory=list)
    threat_level: ThreatLevel = ThreatLevel.SAFE


# 恶意模式定义
MALICIOUS_PATTERNS: list[dict[str, Any]] = [
    # Fork 炸弹
    {
        "name": "fork_bomb",
        "pattern": r"(?:os\.fork|subprocess\.Popen.*shell\s*=\s*True|while\s+True\s*:.*os\.fork)",
        "description": "Fork 炸弹检测: 可能创建大量子进程",
        "level": ThreatLevel.DANGEROUS,
    },
    # Reverse shell
    {
        "name": "reverse_shell",
        "pattern": r"(?:socket\.socket.*connect|subprocess\.(?:call|Popen|run).*\(.*\/bin\/(?:sh|bash|zsh)|nc\s+-[elp])",
        "description": "Reverse shell 检测: 可能建立反向 shell 连接",
        "level": ThreatLevel.DANGEROUS,
    },
    # 系统命令执行
    {
        "name": "system_exec",
        "pattern": r"(?:os\.system|os\.exec\w+|subprocess\.(?:call|run|Popen)\s*\()",
        "description": "系统命令执行: 可能执行任意系统命令",
        "level": ThreatLevel.WARNING,
    },
    # 文件系统破坏
    {
        "name": "filesystem_destruction",
        "pattern": r"(?:shutil\.rmtree\s*\(\s*[\"']\/|os\.remove\s*\(\s*[\"']\/|os\.unlink\s*\(\s*[\"']\/)",
        "description": "文件系统破坏: 可能删除关键系统文件",
        "level": ThreatLevel.DANGEROUS,
    },
    # 危险导入
    {
        "name": "dangerous_import",
        "pattern": r"(?:import\s+ctypes|from\s+ctypes\s+import|__import__\s*\(\s*[\"']ctypes)",
        "description": "危险导入: ctypes 可用于绕过 Python 安全限制",
        "level": ThreatLevel.WARNING,
    },
    # 环境篡改
    {
        "name": "env_tampering",
        "pattern": r"(?:os\.environ\[|os\.putenv|os\.unsetenv)",
        "description": "环境变量篡改: 可能修改 PATH 或安全相关环境变量",
        "level": ThreatLevel.WARNING,
    },
    # 动态代码执行
    {
        "name": "dynamic_exec",
        "pattern": r"(?:exec\s*\(|eval\s*\(|compile\s*\()",
        "description": "动态代码执行: 可能执行任意代码字符串",
        "level": ThreatLevel.WARNING,
    },
    # 权限提升
    {
        "name": "privilege_escalation",
        "pattern": r"(?:os\.setuid|os\.setgid|os\.seteuid|os\.setegid)",
        "description": "权限提升: 尝试修改进程权限",
        "level": ThreatLevel.DANGEROUS,
    },
    # 信号操纵
    {
        "name": "signal_manipulation",
        "pattern": r"(?:signal\.signal\s*\(|os\.kill\s*\()",
        "description": "信号操纵: 可能发送信号给其他进程",
        "level": ThreatLevel.WARNING,
    },
    # 内存映射
    {
        "name": "mmap_usage",
        "pattern": r"(?:mmap\.mmap\s*\()",
        "description": "内存映射: 可能绕过内存限制",
        "level": ThreatLevel.WARNING,
    },
    # 网络监听
    {
        "name": "network_listen",
        "pattern": r"(?:socket\.socket.*bind|http\.server|socketserver)",
        "description": "网络监听: 可能启动网络服务",
        "level": ThreatLevel.WARNING,
    },
    # 资源耗尽
    {
        "name": "resource_exhaustion",
        "pattern": r"(?:\[\s*\w+\s*\]\s*\*\s*\d{5,}|range\s*\(\s*\d{6,}|while\s+True\s*:(?:(?!sleep|break).)*$)",
        "description": "资源耗尽: 可能消耗大量内存或 CPU",
        "level": ThreatLevel.WARNING,
    },
]


class CodeScanner:
    """代码预扫描器

    在代码送入沙箱执行前，使用正则匹配快速检测常见恶意模式。
    """

    def __init__(self, custom_patterns: list[dict[str, Any]] | None = None) -> None:
        self._patterns = list(MALICIOUS_PATTERNS)
        if custom_patterns:
            self._patterns.extend(custom_patterns)

        # 预编译正则
        self._compiled: list[tuple[re.Pattern, dict[str, Any]]] = []
        for pattern_def in self._patterns:
            try:
                compiled = re.compile(pattern_def["pattern"], re.MULTILINE | re.DOTALL)
                self._compiled.append((compiled, pattern_def))
            except re.error:
                pass

    def scan(self, code: str) -> ScanResult:
        """扫描代码

        Args:
            code: 源代码

        Returns:
            扫描结果
        """
        threats: list[dict[str, Any]] = []
        max_level = ThreatLevel.SAFE

        for compiled, pattern_def in self._compiled:
            matches = compiled.findall(code)
            if matches:
                threat = {
                    "name": pattern_def["name"],
                    "description": pattern_def["description"],
                    "level": pattern_def["level"].value,
                    "match_count": len(matches),
                }
                threats.append(threat)

                if pattern_def["level"] == ThreatLevel.DANGEROUS:
                    max_level = ThreatLevel.DANGEROUS
                elif pattern_def["level"] == ThreatLevel.WARNING and max_level == ThreatLevel.SAFE:
                    max_level = ThreatLevel.WARNING

        safe = max_level != ThreatLevel.DANGEROUS

        return ScanResult(
            safe=safe,
            threats=threats,
            threat_level=max_level,
        )
