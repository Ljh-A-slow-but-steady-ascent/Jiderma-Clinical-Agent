from __future__ import annotations

from backend.app.mcp.registry import ToolRegistry
from backend.app.mcp.tools import (
    GetCarePathwayTool,
    GetPrivacyReportTool,
    GetTriageResultTool,
    GetVisualEvidenceTool,
    SearchGuidelineTool,
    WebSearchTool,
)
from backend.app.services.container import ServiceContainer
from backend.app.services.privacy import EgressGuard


def create_tool_registry(services: ServiceContainer) -> ToolRegistry:
    registry = ToolRegistry(services.repository, EgressGuard(services.repository))
    registry.register(GetVisualEvidenceTool(services.visual))
    registry.register(GetTriageResultTool(services.triage))
    registry.register(SearchGuidelineTool(services.knowledge))
    registry.register(GetCarePathwayTool(services.care))
    registry.register(GetPrivacyReportTool(services.privacy))
    registry.register(WebSearchTool(EgressGuard(services.repository)))
    return registry
