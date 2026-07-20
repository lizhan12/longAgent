"""内容质量校验器 — 检测生成文件是否完整、无占位、无截断"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ContentIssue:
    severity: str  # "error" | "warning"
    category: str  # "placeholder" | "empty_section" | "truncated" | "too_short"
    detail: str
    line: int | None = None

    def format(self) -> str:
        prefix = "❌" if self.severity == "error" else "⚠️"
        loc = f" (第{self.line}行)" if self.line else ""
        return f"{prefix} [{self.category}]{loc}: {self.detail}"


@dataclass
class ValidationResult:
    filepath: str
    passed: bool
    content_length: int
    line_count: int
    issues: list[ContentIssue] = field(default_factory=list)

    def format_feedback(self) -> str:
        if self.passed:
            return ""
        lines = [f"\n⚠️ 内容质量警告（{self.filepath}）:"]
        for issue in self.issues:
            lines.append(f"  {issue.format()}")
        return "\n".join(lines)

    def format_summary(self) -> str:
        if self.passed:
            return f"✅ {self.filepath}: 内容完整 ({self.content_length} 字符)"
        return f"❌ {self.filepath}: {len(self.issues)} 个问题 ({self.content_length} 字符)"


class ContentValidator:
    """生成内容质量校验器

    检测模式：
    - 占位符文本（placeholder/待补充/TODO）
    - Markdown 空章节（标题下无实质内容）
    - 内容截断（末尾句子不完整）
    - 内容过短
    """

    PLACEHOLDER_PATTERNS: list[tuple[str, str]] = [
        (r"\bplaceholder\b", "包含英文占位符 'placeholder'"),
        (r"占位", "包含中文占位符 '占位'"),
        (r"待补充", "包含 '待补充'"),
        (r"待完善", "包含 '待完善'"),
        (r"\bTODO\b", "包含 'TODO'"),
        (r"\bTBD\b", "包含 'TBD'"),
        (r"\(暂无数据\)", "包含 '(暂无数据)'"),
        (r"（基于搜索数据整理）", "占位文本 '（基于搜索数据整理）'"),
        (r"在此处填入", "模板占位 '在此处填入'"),
        (r"请根据实际", "模板占位 '请根据实际'"),
    ]

    # 不含实质内容的正则：只含空白、标点、单字
    _EMPTY_CONTENT_RE = re.compile(r"^[\s\d\W_]{0,5}$")

    # 截断标志：末尾句子不完整（无句号、无问号、无感叹号、无冒号结尾）
    _TRUNCATED_END_RE = re.compile(r"[。！？\.\!\?\)」]$")

    # Markdown 标题匹配
    _MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    MIN_CONTENT_CHARS: dict[str, int] = {
        ".md": 200,
        ".html": 100,
        ".py": 10,
        ".txt": 50,
        ".json": 20,
    }

    MIN_NON_EMPTY_LINES: dict[str, int] = {
        ".md": 8,
        ".html": 5,
        ".py": 3,
        ".txt": 5,
        ".json": 3,
    }

    def validate(self, filepath: str | Path, content: str) -> ValidationResult:
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()
        content = content.strip()
        content_length = len(content)
        lines = content.split("\n")
        line_count = len(lines)
        issues: list[ContentIssue] = []

        self._check_placeholders(content, issues)
        self._check_min_length(content, suffix, content_length, issues)

        if suffix == ".md":
            self._check_markdown_sections(lines, issues)
            self._check_truncation(content, issues)

        self._check_non_empty_lines(lines, suffix, issues)

        total_issues = len(issues)
        errors = sum(1 for i in issues if i.severity == "error")
        passed = errors == 0 and total_issues <= 1

        return ValidationResult(
            filepath=str(filepath),
            passed=passed,
            content_length=content_length,
            line_count=line_count,
            issues=issues,
        )

    def _check_placeholders(self, content: str, issues: list[ContentIssue]) -> None:
        for pattern, description in self.PLACEHOLDER_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                line_num = self._find_line(content, pattern)
                issues.append(ContentIssue(
                    severity="error",
                    category="placeholder",
                    detail=description,
                    line=line_num,
                ))
                return

    def _check_min_length(
        self, content: str, suffix: str, length: int, issues: list[ContentIssue]
    ) -> None:
        threshold = self.MIN_CONTENT_CHARS.get(suffix, 100)
        if length < threshold:
            issues.append(ContentIssue(
                severity="error",
                category="too_short",
                detail=f"内容仅 {length} 字符，最低要求 {threshold} 字符",
            ))

    def _check_non_empty_lines(
        self, lines: list[str], suffix: str, issues: list[ContentIssue]
    ) -> None:
        non_empty = [l for l in lines if l.strip()]
        threshold = self.MIN_NON_EMPTY_LINES.get(suffix, 5)
        if len(non_empty) < threshold:
            issues.append(ContentIssue(
                severity="warning",
                category="too_short",
                detail=f"仅 {len(non_empty)} 行非空内容，最低要求 {threshold} 行",
            ))

    def _check_markdown_sections(
        self, lines: list[str], issues: list[ContentIssue]
    ) -> None:
        """检测 Markdown 中标题下无实质内容的章节（跳过 H1 文档大标题）"""
        headings = [(i, len(m.group(1)), m.group(2))
                    for i, line in enumerate(lines)
                    if (m := self._MD_HEADING_RE.match(line))]

        for idx, (line_idx, level, title) in enumerate(headings):
            if level == 1:
                continue

            # 找到下一个同级或更高级标题的位置
            next_idx = len(lines)
            for j in range(idx + 1, len(headings)):
                if headings[j][1] <= level:
                    next_idx = headings[j][0]
                    break

            section_lines = lines[line_idx + 1:next_idx]
            section_content = "\n".join(section_lines).strip()

            # 如果段落中有更深层级的子标题且有内容，不算空
            has_substance = bool(section_content) and not self._EMPTY_CONTENT_RE.match(
                section_content[:50]
            )

            if not has_substance:
                issues.append(ContentIssue(
                    severity="warning",
                    category="empty_section",
                    detail=f"标题 '{title}' 下无实质内容",
                    line=line_idx + 1,
                ))

    def _check_truncation(self, content: str, issues: list[ContentIssue]) -> None:
        """检测内容是否截断（末尾句子不完整）"""
        stripped = content.rstrip()
        if not stripped:
            return
        last_char = stripped[-1]
        if last_char not in "。！？.!?)\"\'」》\n":
            last_line = stripped.split("\n")[-1].strip()
            # 如果最后一行以正常标点结尾，不算截断
            if self._TRUNCATED_END_RE.search(last_line[-3:] if len(last_line) >= 3 else last_line):
                return
            if last_line.endswith(("```", "---", "***")):
                return
            # 最后一行长度小于 10 且以逗号/顿号结尾 = 截断
            if len(last_line) < 10 or last_line[-1] in "，,、;；:":
                issues.append(ContentIssue(
                    severity="warning",
                    category="truncated",
                    detail="内容可能被截断，末尾句子不完整",
                ))

    def _find_line(self, content: str, pattern: str) -> int | None:
        for i, line in enumerate(content.split("\n"), 1):
            if re.search(pattern, line, re.IGNORECASE):
                return i
        return None


content_validator = ContentValidator()


def validate_content(filepath: str | Path, content: str) -> tuple[bool, str]:
    """快捷校验函数，返回 (是否通过, 反馈信息)"""
    result = content_validator.validate(filepath, content)
    return result.passed, result.format_feedback()