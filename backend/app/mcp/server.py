from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from backend.app.mcp.factory import create_tool_registry
from backend.app.mcp.schemas import ToolContext
from backend.app.services.container import ServiceContainer, create_service_container


def create_mcp_server(services: ServiceContainer | None = None) -> FastMCP:
    services = services or create_service_container()
    registry = create_tool_registry(services)
    server = FastMCP(
        name="dermaguard-local-tools",
        instructions=(
            "DermaGuard local evidence, policy, care-pathway, guideline, and privacy tools. "
            "Never request or return raw medical image bytes or patient identity."
        ),
    )

    @server.tool(structured_output=True)
    async def get_visual_evidence(case_id: str) -> dict[str, Any]:
        """Read persisted local visual evidence; raw images are never returned."""
        call = await registry.invoke(
            "get_visual_evidence",
            {"case_id": case_id},
            ToolContext(case_id=case_id, channel="mcp", destination="mcp_client"),
        )
        return call.result

    @server.tool(structured_output=True)
    async def get_triage_result(case_id: str) -> dict[str, Any]:
        """Read the immutable deterministic TriagePolicy result."""
        call = await registry.invoke(
            "get_triage_result",
            {"case_id": case_id},
            ToolContext(case_id=case_id, channel="mcp", destination="mcp_client"),
        )
        return call.result

    @server.tool(structured_output=True)
    async def search_guideline(
        query: str, case_id: str | None = None, top_k: int = 3
    ) -> dict[str, Any]:
        """Search only the local triage and referral guideline collection."""
        call = await registry.invoke(
            "search_guideline",
            {"query": query, "case_id": case_id, "top_k": top_k},
            ToolContext(case_id=case_id, channel="mcp", destination="mcp_client"),
        )
        return call.result

    @server.tool(structured_output=True)
    async def get_care_pathway(case_id: str) -> dict[str, Any]:
        """Read the next-care pathway bound to the policy result."""
        call = await registry.invoke(
            "get_care_pathway",
            {"case_id": case_id},
            ToolContext(case_id=case_id, channel="mcp", destination="mcp_client"),
        )
        return call.result

    @server.tool(structured_output=True)
    async def get_privacy_report(case_id: str) -> dict[str, Any]:
        """Read audited facts about outbound case data and image locality."""
        call = await registry.invoke(
            "get_privacy_report",
            {"case_id": case_id},
            ToolContext(case_id=case_id, channel="mcp", destination="mcp_client"),
        )
        return call.result

    @server.tool(structured_output=True)
    async def web_search(query: str, top_k: int = 3) -> dict[str, Any]:
        """Search public references for simple non-medical questions only."""
        call = await registry.invoke(
            "web_search", {"query": query, "top_k": top_k},
            ToolContext(channel="http", destination="zh.wikipedia.org"),
        )
        return call.result

    return server


mcp = create_mcp_server()


if __name__ == "__main__":
    mcp.run(transport="stdio")
