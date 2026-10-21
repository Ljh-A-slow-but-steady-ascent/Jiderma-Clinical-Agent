from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolContext(StrictModel):
    case_id: str | None = None
    channel: str = "mcp"
    destination: str = "local_mcp_client"
    for_cloud: bool = False


class ToolDefinition(StrictModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolInvocation(StrictModel):
    name: str
    arguments: dict[str, Any]
    context: ToolContext = Field(default_factory=ToolContext)


class ToolInvocationResult(StrictModel):
    tool_name: str
    result: dict[str, Any]
    latency_ms: float = Field(ge=0)
    success: bool
    egress_allowed: bool | None = None
    egress_latency_ms: float | None = Field(default=None, ge=0)
