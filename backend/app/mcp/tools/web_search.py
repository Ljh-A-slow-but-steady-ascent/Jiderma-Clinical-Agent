from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import quote

import requests
from pydantic import BaseModel, ConfigDict, Field

from backend.app.mcp.schemas import ToolContext
from backend.app.services.privacy import EgressGuard, EgressRequest, sanitize_text


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=3, ge=1, le=5)


class WebSearchTool:
    name = "web_search"
    description = "Search public web references for simple non-medical questions."

    def __init__(self, guard: EgressGuard | None = None) -> None:
        self.guard = guard or EgressGuard()

    def input_schema(self) -> dict[str, Any]:
        return Input.model_json_schema()

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        request = Input.model_validate(arguments)
        safe_query = sanitize_text(request.query)[0]
        decision = self.guard.enforce(EgressRequest(
            channel="http", destination="zh.wikipedia.org",
            payload={"query": safe_query}, tool_name=self.name,
        ))
        safe_query = str(decision.sanitized_payload["query"])
        response = requests.get(
            "https://zh.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search", "srsearch": safe_query,
                "srlimit": request.top_k, "utf8": 1, "format": "json",
            },
            headers={"User-Agent": "Jiderma-Demo/1.0 (public knowledge lookup)"},
            timeout=8,
        )
        response.raise_for_status()
        items = response.json().get("query", {}).get("search", [])
        results = []
        for item in items[: request.top_k]:
            title = str(item.get("title", "")).strip()
            snippet = re.sub(r"<[^>]+>", "", str(item.get("snippet", "")))
            results.append({
                "title": title,
                "url": f"https://zh.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                "snippet": html.unescape(snippet).strip(),
                "source": "Wikipedia",
            })
        return {"query": safe_query, "results": results}
