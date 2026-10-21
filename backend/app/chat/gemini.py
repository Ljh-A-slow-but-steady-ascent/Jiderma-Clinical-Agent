from __future__ import annotations

import json
import os
from typing import Any

import requests


class GeminiChatProvider:
    """Structured JSON planner used by clinical extraction and report services."""

    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("已选择 Gemini Flash，但未设置 GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self.base_url = base_url or os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/models")

    def plan(self, payload: dict[str, Any], system_prompt: str) -> str:
        response = requests.post(
            f"{self.base_url.rstrip('/')}/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json={
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]}],
                "generationConfig": {
                    "temperature": 0.05,
                    "maxOutputTokens": 700,
                    "responseMimeType": "application/json",
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        try:
            text = "".join(
                str(part.get("text", ""))
                for part in response.json()["candidates"][0]["content"]["parts"]
            ).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Gemini规划响应无法解析") from exc
        if not text:
            raise RuntimeError("Gemini规划返回空正文")
        return text
