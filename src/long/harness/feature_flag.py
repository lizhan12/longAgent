"""渐进式发布 — YAML 配置 + Session 粘性灰度切换

Harness Engineering 原则：渐进式发布
通过 Feature Flag 机制支持：
- Prompt 版本灰度（不同比例用户使用不同 AGENTS.md 版本）
- Model 灰度切换（按 session_id 哈希分流）
- 即时回滚（修改配置即可，无需重启）

设计约束：
- 最小可用：零外部依赖，纯 YAML + 哈希分流
- Session 粘性：同一会话始终使用同一版本，体验一致
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_FLAG_CONFIG = """
feature_flags:
  prompt_version:
    strategy: session_hash
    variants:
      - name: stable
        weight: 80
        file: AGENTS.md
      - name: canary
        weight: 20
        file: AGENTS.canary.md
    sticky: true

  model_version:
    strategy: session_hash
    variants:
      - name: primary
        weight: 70
        model: deepseek-chat
      - name: fallback
        weight: 30
        model: deepseek-v3-0324
    sticky: true

  output_pii_filter:
    enabled: true
    strategy: global

  memory_consolidator:
    enabled: true
    strategy: global

  auto_eval_feedback:
    enabled: true
    strategy: global
    approval_required: true
"""


def _hash_session(session_id: str) -> int:
    """确定性哈希，同一 session 总是返回同一 bucket"""
    if not session_id:
        return 0
    return int(hashlib.md5(session_id.encode()).hexdigest()[:8], 16)


class FeatureFlag:
    """特征开关管理器

    用法：
        flags = FeatureFlag.from_yaml(config_path)
        prompt_file = flags.get_variant("prompt_version", session_id="abc123")
        model = flags.get_variant("model_version", session_id="abc123")
        if flags.is_enabled("output_pii_filter"):
            ...
    """

    def __init__(self, flags: dict[str, Any]) -> None:
        self._flags = flags

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FeatureFlag":
        import yaml
        import logging

        logger = logging.getLogger(__name__)

        path = Path(path)
        if not path.exists():
            return cls(yaml.safe_load(DEFAULT_FLAG_CONFIG))

        try:
            content = path.read_text(encoding="utf-8")
            config = yaml.safe_load(content) if content.strip() else {}
            return cls(config.get("feature_flags", {}))
        except Exception as e:
            logger.warning("FeatureFlag YAML 解析失败 (%s): %s，使用默认配置", path, e)
            return cls.defaults()

    @classmethod
    def defaults(cls) -> "FeatureFlag":
        import yaml

        config = yaml.safe_load(DEFAULT_FLAG_CONFIG)
        return cls(config.get("feature_flags", {}))

    def is_enabled(self, flag_name: str) -> bool:
        """检查全局开关是否启用"""
        flag = self._flags.get(flag_name, {})
        if isinstance(flag, dict):
            return flag.get("enabled", True)
        return bool(flag)

    def get_variant(self, flag_name: str, *, session_id: str = "") -> str:
        """获取灰度分流结果

        基于 session_id 哈希做粘性分流，确保同一会话始终命中同一变体。
        """
        flag = self._flags.get(flag_name)
        if not flag or not isinstance(flag, dict):
            return ""

        variants = flag.get("variants", [])
        if not variants:
            return ""

        total_weight = sum(v.get("weight", 0) for v in variants)
        if total_weight == 0:
            return variants[0].get("name", "")

        bucket = _hash_session(session_id) % total_weight

        cumulative = 0
        for v in variants:
            cumulative += v.get("weight", 0)
            if bucket < cumulative:
                return v.get("name", "")

        return variants[-1].get("name", "")

    def get_variant_value(self, flag_name: str, *, session_id: str = "", key: str = "name") -> str:
        """获取灰度分流变体的指定字段值"""
        variant_name = self.get_variant(flag_name, session_id=session_id)
        flag = self._flags.get(flag_name, {})
        variants = flag.get("variants", []) if isinstance(flag, dict) else []
        for v in variants:
            if v.get("name") == variant_name:
                return v.get(key, "")
        return ""

    def list_flags(self) -> dict[str, Any]:
        return dict(self._flags)


class PromptVersion:
    """Prompt 版本管理

    根据 FeatureFlag 的分流结果读取对应的 AGENTS.md 版本。
    默认文件：AGENTS.md
    Canary 文件：AGENTS.canary.md（灰度用户使用此版本）
    """

    def __init__(self, workspace_root: str | Path, flags: FeatureFlag) -> None:
        self._root = Path(workspace_root)
        self._flags = flags

    def get_prompt(self, session_id: str = "") -> str:
        """获取当前会话应使用的 Agent 人格 Prompt"""
        if not self._flags.is_enabled("prompt_version"):
            agents_path = self._root / "AGENTS.md"
            if agents_path.exists():
                return agents_path.read_text(encoding="utf-8").strip()
            return ""

        variant = self._flags.get_variant("prompt_version", session_id=session_id)
        file_name = self._flags.get_variant_value(
            "prompt_version", session_id=session_id, key="file",
        ) or "AGENTS.md"

        file_path = self._root / file_name
        if file_path.exists():
            logger.info("Prompt 版本 [%s] → %s (session=%s)", variant, file_name, session_id[:8])
            return file_path.read_text(encoding="utf-8").strip()

        fallback = self._root / "AGENTS.md"
        if fallback.exists():
            return fallback.read_text(encoding="utf-8").strip()
        return ""

    def get_model(self, session_id: str = "") -> str:
        """获取当前会话应使用的模型名"""
        if not self._flags.is_enabled("model_version"):
            return ""

        return self._flags.get_variant_value(
            "model_version", session_id=session_id, key="model",
        )

    def canary_info(self, session_id: str = "") -> dict[str, str]:
        """获取当前会话的灰度信息（用于调试）"""
        return {
            "prompt_version": self._flags.get_variant("prompt_version", session_id=session_id),
            "model_version": self._flags.get_variant("model_version", session_id=session_id),
            "prompt_file": self._flags.get_variant_value(
                "prompt_version", session_id=session_id, key="file",
            ),
            "model": self._flags.get_variant_value(
                "model_version", session_id=session_id, key="model",
            ),
        }