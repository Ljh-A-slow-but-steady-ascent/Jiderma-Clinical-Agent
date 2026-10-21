from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from backend.app.mcp.schemas import ToolContext
from backend.app.services.privacy_service import PrivacyService


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str


class GetPrivacyReportTool:
    name = "get_privacy_report"
    description = "Read audited facts about whether case data left the local trusted domain."

    def __init__(self, service: PrivacyService) -> None:
        self.service = service

    def input_schema(self) -> dict[str, Any]:
        return Input.model_json_schema()

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        request = Input.model_validate(arguments)
        return self.service.report(request.case_id).model_dump(mode="json")
