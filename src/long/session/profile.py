from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class UserProfile:
    """用户画像（三阶记忆）

    从多次对话摘要中提炼的稳定特征。
    比偏好更高层级——偏好是具体规则，画像是整体特征。

    存储在 workspace/data/user_profile.json
    """

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "user_profile.json"
        self._profile: dict[str, Any] = {
            "tech_stack": [],
            "work_style": "",
            "communication_preference": "",
            "code_preferences": [],
            "domain_expertise": [],
            "frequent_topics": [],
            "updated_at": "",
        }
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in self._profile:
                        self._profile[k] = v
        except Exception as e:
            logger.warning("加载用户画像失败: %s", e)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._profile, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("保存用户画像失败: %s", e)

    def get(self, key: str) -> Any:
        return self._profile.get(key)

    def update(self, key: str, value: Any) -> None:
        if key in self._profile:
            self._profile[key] = value
            self._profile["updated_at"] = datetime.now().isoformat()
            self._save()

    def update_from_dict(self, data: dict[str, Any]) -> int:
        count = 0
        for k, v in data.items():
            if k in self._profile and v:
                old = self._profile[k]
                if isinstance(old, list) and isinstance(v, list):
                    merged = list(set(old + v))
                    if len(merged) > len(old):
                        self._profile[k] = merged
                        count += 1
                elif old != v:
                    self._profile[k] = v
                    count += 1
        if count > 0:
            self._profile["updated_at"] = datetime.now().isoformat()
            self._save()
        return count

    def format_for_prompt(self) -> str:
        non_empty = {
            k: v for k, v in self._profile.items()
            if v and k != "updated_at"
        }
        if not non_empty:
            return ""

        lines = ["## 用户画像", ""]
        field_labels = {
            "tech_stack": "技术栈",
            "work_style": "工作风格",
            "communication_preference": "沟通偏好",
            "code_preferences": "代码偏好",
            "domain_expertise": "领域专长",
            "frequent_topics": "常聊话题",
        }
        for key, label in field_labels.items():
            val = non_empty.get(key)
            if val:
                if isinstance(val, list):
                    lines.append(f"- **{label}**: {', '.join(str(v) for v in val)}")
                else:
                    lines.append(f"- **{label}**: {val}")
        lines.append("")
        return "\n".join(lines)

    async def extract_from_summaries(
        self,
        summaries: list[str],
        llm_client: Any | None = None,
    ) -> int:
        """从摘要中提取/更新用户画像

        Returns:
            更新的字段数
        """
        if not summaries:
            return 0

        combined = "\n".join(summaries)
        if len(combined) > 6000:
            combined = combined[:6000] + "...(截断)"

        if llm_client is None:
            return self._rule_based_extract(combined)

        try:
            from long.llm.base import LLMMessage

            messages = [
                LLMMessage(
                    role="system",
                    content=(
                        "从以下对话摘要中提取用户的稳定特征，返回 JSON 格式：\n"
                        '{"tech_stack": ["技术栈"], "work_style": "工作风格", '
                        '"communication_preference": "沟通偏好", '
                        '"code_preferences": ["代码偏好"], '
                        '"domain_expertise": ["领域专长"], '
                        '"frequent_topics": ["常聊话题"]}\n'
                        "只提取明确的信息，不确定的留空数组或空字符串。"
                    ),
                ),
                LLMMessage(role="user", content=f"对话摘要：\n\n{combined}"),
            ]

            response = await llm_client.chat(messages, purpose="profile")
            content = response.content.strip()

            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                raw_json = content[json_start:json_end]
                try:
                    data = json.loads(raw_json)
                except json.JSONDecodeError:
                    import re
                    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', raw_json)
                    cleaned = re.sub(r',\s*}', '}', cleaned)
                    cleaned = re.sub(r',\s*]', ']', cleaned)
                    cleaned = re.sub(r'"\s*:\s*"', '": "', cleaned)
                    data = json.loads(cleaned)
                return self.update_from_dict(data)

        except Exception as e:
            logger.debug("LLM 画像提取失败: %s，降级为规则提取", e)
            return self._rule_based_extract(combined)

        return 0

    def _rule_based_extract(self, text: str) -> int:
        """基于规则的画像提取（降级方案）"""
        count = 0
        tech_keywords = [
            "Python", "JavaScript", "TypeScript", "Go", "Rust", "Java",
            "FastAPI", "Django", "Flask", "React", "Vue", "Next.js",
            "Docker", "Kubernetes", "vLLM", "PostgreSQL", "Redis",
        ]
        found_tech = [kw for kw in tech_keywords if kw.lower() in text.lower()]
        if found_tech:
            existing = set(self._profile.get("tech_stack", []))
            new_tech = [t for t in found_tech if t not in existing]
            if new_tech:
                self._profile["tech_stack"] = list(existing) + new_tech
                count += 1

        style_keywords = {
            "第一性原理": "第一性原理思维",
            "系统级": "系统级思维",
            "简洁": "偏好简洁",
            "详细": "偏好详细",
            "架构": "架构思维",
        }
        for kw, style in style_keywords.items():
            if kw in text and not self._profile.get("work_style"):
                self._profile["work_style"] = style
                count += 1
                break

        if count > 0:
            self._profile["updated_at"] = datetime.now().isoformat()
            self._save()

        return count
