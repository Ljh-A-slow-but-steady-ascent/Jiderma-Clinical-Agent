from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from backend.app.mcp.schemas import ToolContext
from backend.app.services.triage_service import TriagePolicyService


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str


class GetTriageResultTool:
    name = "get_triage_result"
    description = "Read the immutable deterministic TriagePolicy result for a case."

    def __init__(self, service: TriagePolicyService) -> None:
        self.service = service

    def input_schema(self) -> dict[str, Any]:
        return Input.model_json_schema()

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        request = Input.model_validate(arguments)
        return self.service.get(request.case_id).model_dump(mode="json")
