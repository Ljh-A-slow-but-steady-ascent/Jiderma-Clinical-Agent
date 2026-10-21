from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.app.mcp.schemas import ToolContext
from backend.app.services.knowledge_service import KnowledgeService


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500)
    case_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=5)


class SearchGuidelineTool:
    name = "search_guideline"
    description = "Search only the local triage and referral guideline knowledge base."

    def __init__(self, service: KnowledgeService) -> None:
        self.service = service

    def input_schema(self) -> dict[str, Any]:
        return Input.model_json_schema()

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        request = Input.model_validate(arguments)
        chunks = self.service.search(request.query, top_k=request.top_k)
        return {
            "chunks": chunks,
            "evidence_assembly": self.service.assemble_evidence(chunks),
        }
