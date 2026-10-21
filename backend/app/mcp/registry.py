from __future__ import annotations

import time
from typing import Any, Protocol

from jsonschema import Draft202012Validator

from backend.app.database import CaseRepository
from backend.app.mcp.schemas import ToolContext, ToolDefinition, ToolInvocationResult
from backend.app.services.privacy import EgressGuard, EgressRequest, sanitize_for_storage


class RegisteredTool(Protocol):
    name: str
    description: str

    def input_schema(self) -> dict[str, Any]: ...

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]: ...


class ToolRegistry:
    def __init__(
        self,
        repository: CaseRepository | None = None,
        egress_guard: EgressGuard | None = None,
    ) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self.repository = repository
        self.egress_guard = egress_guard

    def register(self, tool: RegisteredTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def list_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name=tool.name, description=tool.description, input_schema=tool.input_schema())
            for tool in self._tools.values()
        ]

    def validate_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if tool_name not in self._tools:
            raise KeyError(f"unknown tool: {tool_name}")
        Draft202012Validator(self._tools[tool_name].input_schema()).validate(arguments)

    async def invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolInvocationResult:
        if tool_name not in self._tools:
            raise KeyError(f"unknown tool: {tool_name}")
        self.validate_arguments(tool_name, arguments)
        context = context or ToolContext()
        started = time.perf_counter()
        success = False
        result: dict[str, Any] = {}
        try:
            result = await self._tools[tool_name].invoke(arguments, context)
            egress_allowed = None
            egress_latency = None
            if context.for_cloud:
                guard = self.egress_guard or EgressGuard(self.repository)
                decision = guard.enforce(
                    EgressRequest(
                        channel="llm",
                        destination=context.destination,
                        payload=result,
                        case_id=context.case_id or arguments.get("case_id"),
                        tool_name=tool_name,
                    )
                )
                result = decision.sanitized_payload
                egress_allowed = decision.allowed
                egress_latency = decision.latency_ms
            success = True
            return ToolInvocationResult(
                tool_name=tool_name,
                result=result,
                latency_ms=(time.perf_counter() - started) * 1000,
                success=True,
                egress_allowed=egress_allowed,
                egress_latency_ms=egress_latency,
            )
        finally:
            if self.repository is not None:
                latency = (time.perf_counter() - started) * 1000
                self.repository.add_tool_call(
                    case_id=context.case_id or arguments.get("case_id"),
                    tool_name=tool_name,
                    arguments=sanitize_for_storage(arguments),
                    result=sanitize_for_storage(result) if success else {"error": "tool invocation failed"},
                    success=success,
                    latency_ms=latency,
                )
