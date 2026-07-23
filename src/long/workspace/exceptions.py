"""工作区路径相关异常

这些异常同时继承 PermissionError 和 ValueError：
  - PermissionError：保持与既有 `except PermissionError` 调用方的兼容
  - ValueError：路径非法本质上是入参错误，便于按值错误捕获
"""

from __future__ import annotations


class WorkspacePathError(PermissionError, ValueError):
    """工作区路径非法的基类"""


class PathTraversalError(WorkspacePathError):
    """路径穿越：解析后的路径落在工作区之外"""


class AbsolutePathError(WorkspacePathError):
    """绝对路径：工作区只接受相对路径"""
