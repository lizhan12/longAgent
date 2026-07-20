"""级联路由异常定义"""

from __future__ import annotations


class CascadeExhaustedError(Exception):
    """所有模型 Tier 均不可用"""

    def __init__(self, message: str = "所有模型 Tier 均不可用，需要人工介入") -> None:
        super().__init__(message)