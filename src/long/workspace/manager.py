"""工作区管理器"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .filesystem import resolve_within


class WorkspaceManager:
    """工作区管理器 — 管理项目工作区目录结构"""

    def __init__(self, root: str) -> None:
        self._root = Path(root).resolve()
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """确保工作区目录结构存在"""
        dirs = [
            self._root,
            self._root / "data",
            self._root / "logs",
            self._root / "traces",
            self._root / "skills",
            self._root / "sandbox",
            self._root / "cache",
            self._root / "output",
            self._root / "memory",
            self._root / "knowledge",
            self._root / "subagents",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def data_dir(self) -> Path:
        return self._root / "data"

    @property
    def traces_dir(self) -> Path:
        return self._root / "traces"

    @property
    def skills_dir(self) -> Path:
        return self._root / "skills"

    @property
    def sandbox_dir(self) -> Path:
        return self._root / "sandbox"

    def list_files(self, path: str = "") -> list[Path]:
        """列出工作区目录下的文件和子目录"""
        target = self._root
        if path:
            target = self._root / path
        if not target.exists():
            return []
        if target.is_file():
            return [target]
        return sorted(target.iterdir())

    def read_file(self, path: str) -> str | bytes:
        """读取工作区中文件的内容"""
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if target.suffix in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".pdf", ".zip", ".gz"):
            return target.read_bytes()
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str | bytes) -> None:
        """将内容写入工作区中的文件"""
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            target.write_text(content, encoding="utf-8")
        else:
            target.write_bytes(content)

    def delete_file(self, path: str) -> bool:
        """删除工作区中的文件或目录"""
        target = self._resolve(path)
        if not target.exists():
            return False
        if target.is_dir():
            import shutil
            shutil.rmtree(target)
        else:
            target.unlink()
        return True

    def resolve(self, path: str) -> Path:
        """解析路径，确保在工作区范围内

        越界抛 PathTraversalError，绝对路径抛 AbsolutePathError，
        两者都是 PermissionError 的子类，既有捕获逻辑不受影响。
        """
        return resolve_within(self._root, path)

    # 兼容旧调用方
    _resolve = resolve