"""Pytest 配置文件"""

import sys
from pathlib import Path

import pytest

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """创建临时工作区目录"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace
