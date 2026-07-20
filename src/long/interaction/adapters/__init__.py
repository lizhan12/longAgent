"""交互适配器"""

from .cli import CLIAdapter
from .webui import WebUIAdapter, WebSocketConnection

__all__ = ["CLIAdapter", "WebUIAdapter", "WebSocketConnection"]
