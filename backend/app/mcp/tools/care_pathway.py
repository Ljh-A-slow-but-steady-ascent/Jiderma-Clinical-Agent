from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from backend.app.mcp.schemas import ToolContext
from backend.app.services.care_service import CareService


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str


class GetCarePathwayTool:
    name = "get_care_pathway"
    description = "Read the policy-bound next-care pathway for a case."

    def __init__(self, service: CareService) -> None:
        self.service = service

    def input_schema(self) -> dict[str, Any]:
        return Input.model_json_schema()

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        request = Input.model_validate(arguments)
        return self.service.get(request.case_id).model_dump(mode="json")
