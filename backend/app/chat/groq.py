from __future__ import annotations

import json
import os
from typing import Any

import requests


class GroqChatProvider:
    """Structured JSON planner used by clinical extraction and report services."""

    name = "groq"

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("已选择 Groq，但未设置 GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.base_url = base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")

    def plan(self, payload: dict[str, Any], system_prompt: str) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "temperature": 0.05,
            "max_tokens": 700,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            ],
        }
        if self.model.startswith("openai/gpt-oss-"):
            body["reasoning_effort"] = "low"
        response = requests.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
        response.raise_for_status()
        try:
            text = str(response.json()["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Groq规划响应无法解析") from exc
        if not text:
            raise RuntimeError("Groq规划返回空正文")
        return text
