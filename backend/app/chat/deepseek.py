from __future__ import annotations

import json
import os
import time
from typing import Any

import requests


class DeepSeekChatProvider:
    """Structured JSON planner used by clinical extraction and report services."""

    name = "deepseek"

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("已选择 DeepSeek，但未设置 DEEPSEEK_API_KEY")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions")

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        response = None
        for attempt in range(3):
            response = requests.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=90,
            )
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
                break
            try:
                delay = float(response.headers.get("Retry-After", attempt + 1))
            except (TypeError, ValueError):
                delay = attempt + 1
            time.sleep(min(max(delay, 0.2), 3.0))
        assert response is not None
        response.raise_for_status()
        return response.json()

    def plan(self, payload: dict[str, Any], system_prompt: str) -> str:
        body = self._post({
            "model": self.model,
            "stream": False,
            "temperature": 0.05,
            "max_tokens": 900,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            ],
        })
        try:
            content = body["choices"][0]["message"].get("content")
            text = "" if content is None else str(content).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("DeepSeek规划响应无法解析") from exc
        if not text:
            raise RuntimeError("DeepSeek规划返回空正文")
        return text
