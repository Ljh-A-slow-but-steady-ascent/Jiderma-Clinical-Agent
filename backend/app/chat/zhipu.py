from __future__ import annotations

import json
import os
from typing import Any

import requests


class ZhipuChatProvider:
    """Structured JSON planner used by clinical extraction and report services."""

    name = "zhipu_glm"

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("已选择智谱 GLM，但未设置 ZHIPUAI_API_KEY")
        self.model = model or os.getenv("ZHIPUAI_MODEL", "glm-4.7-flash")
        self.base_url = base_url or os.getenv("ZHIPUAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")

    def plan(self, payload: dict[str, Any], system_prompt: str) -> str:
        response = requests.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "stream": False,
                "temperature": 0.05,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
        try:
            text = str(response.json()["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("智谱规划响应无法解析") from exc
        if not text:
            raise RuntimeError("智谱规划返回空正文")
        return text
