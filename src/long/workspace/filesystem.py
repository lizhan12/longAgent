"""工作区文件系统抽象层"""

from __future__ import annotations

import os
from pathlib import Path

from .exceptions import AbsolutePathError, PathTraversalError


def resolve_within(root: Path, path: str) -> Path:
    """把相对路径解析到 root 之内，越界即抛异常

    旧实现用 ``str(resolved).startswith(str(root))`` 判断包含关系，这是可绕过的：
    root=/tmp/workspace 时 ../workspace2/secret 解析成 /tmp/workspace2/secret，
    前缀匹配同样成立，于是能直接读到工作区外的兄弟目录。
    改用 Path.is_relative_to 做真正的祖先判断。
    """
    candidate = Path(path)
    # Windows 上 Path("/etc/passwd").is_absolute() 为 False（没有盘符），
    # 所以还要单独判断以分隔符开头的写法。
    if candidate.is_absolute() or candidate.drive or str(path).startswith(("/", "\\")):
        raise AbsolutePathError(f"不接受绝对路径: {path}")

    root_resolved = root.resolve()
    resolved = (root_resolved / candidate).resolve()
    if not resolved.is_relative_to(root_resolved):
        raise PathTraversalError(f"路径超出工作区边界: {path}")
    return resolved


class WorkspacePath:
    """工作区路径解析器

    把所有相对路径限制在 root 之内，供文件系统层和管理器共用。
    """

    def __init__(self, root: str) -> None:
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, path: str) -> Path:
        """解析路径，越界抛 PathTraversalError / AbsolutePathError"""
        return resolve_within(self._root, path)


class LocalFilesystem:
    """本地文件系统操作封装"""

    def __init__(self, root: str) -> None:
        self._root = Path(root).resolve()
        self._path = WorkspacePath(root)

    def _resolve(self, path: str) -> Path:
        """解析路径，确保在工作区范围内"""
        return self._path.resolve(path)

    async def list_dir(self, path: str = "") -> list[str]:
        """列出目录下的文件和子目录名"""
        # 这里以前直接拼 self._root / path，绕过了边界检查，
        # list_dir("../workspace2") 能列出工作区外的目录。
        try:
            target = self._resolve(path) if path else self._root
        except (PathTraversalError, AbsolutePathError):
            return []
        if not target.exists() or not target.is_dir():
            return []
        return sorted(
            str(p.name) for p in target.iterdir()
        )

    async def is_dir(self, path: str) -> bool:
        """检查路径是否为目录"""
        target = self._resolve(path)
        return target.is_dir()

    async def read(self, path: str) -> str:
        """读取文件内容"""
        target = self._resolve(path)
        return target.read_text(encoding="utf-8")

    async def write(self, path: str, content: str) -> None:
        """写入文件内容"""
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    async def exists(self, path: str) -> bool:
        """检查文件或目录是否存在"""
        target = self._resolve(path)
        return target.exists()

    async def delete(self, path: str) -> None:
        """删除文件或目录"""
        target = self._resolve(path)
        if target.is_dir():
            import shutil
            shutil.rmtree(target)
        else:
            target.unlink()