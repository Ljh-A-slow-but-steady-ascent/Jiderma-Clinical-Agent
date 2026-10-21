from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, START, StateGraph

if TYPE_CHECKING:
    from backend.app.harness.core import ClinicalAgentHarness


class HarnessGraphState(TypedDict, total=False):
    run_id: str
    user_id: int
    message: str
    case_id: str | None
    model: str
    detail_level: str
    image_bytes: bytes | None
    image_content_type: str | None
    intake: Any
    conversation_history: list[dict[str, Any]]
    conversation_id: str
    intent: str
    action_kind: str
    response: Any


async def request_route(state: HarnessGraphState) -> str:
    # Keep the router async: LangGraph otherwise delegates a sync lambda to the
    # event loop's default executor, which can delay clean shutdown in ASGI tests.
    return state["intent"]


async def confirmation_route(state: HarnessGraphState) -> str:
    return state["action_kind"]


def build_request_graph(harness: ClinicalAgentHarness):
    """Compile the clinician request graph used for every new/resumed turn."""
    graph = StateGraph(HarnessGraphState)
    graph.add_node("route_request", harness.graph_route_request)
    graph.add_node("prepare_case", harness.graph_prepare_case)
    graph.add_node("collect_materials", harness.graph_collect_materials)
    graph.add_node("pause", harness.graph_pause)
    graph.add_node("prepare_update", harness.graph_prepare_update)
    graph.add_node("generate_report", harness.graph_generate_report)
    graph.add_node("analyze_case", harness.graph_analyze_case)
    graph.add_node("general_chat", harness.graph_general_chat)
    graph.add_node("clinical_chat", harness.graph_clinical_chat)
    graph.add_edge(START, "route_request")
    graph.add_conditional_edges(
        "route_request",
        request_route,
        {
            "prepare_case": "prepare_case",
            "collect_materials": "collect_materials",
            "pause": "pause",
            "prepare_update": "prepare_update",
            "generate_report": "generate_report",
            "analyze_case": "analyze_case",
            "general_chat": "general_chat",
            "clinical_chat": "clinical_chat",
        },
    )
    for node in (
        "prepare_case", "collect_materials", "pause", "prepare_update",
        "generate_report", "analyze_case", "general_chat", "clinical_chat",
    ):
        graph.add_edge(node, END)
    return graph.compile()


def build_confirmation_graph(harness: ClinicalAgentHarness):
    """Compile the confirmation gate; only approved mutations reach write nodes."""
    graph = StateGraph(HarnessGraphState)
    graph.add_node("route_confirmation", harness.graph_route_confirmation)
    graph.add_node("create_case", harness.graph_confirm_create_case)
    graph.add_node("update_case", harness.graph_confirm_update_case)
    graph.add_node("generate_report", harness.graph_confirm_generate_report)
    graph.add_node("delete_report", harness.graph_confirm_delete_report)
    graph.add_node("delete_document", harness.graph_confirm_delete_document)
    graph.add_edge(START, "route_confirmation")
    graph.add_conditional_edges(
        "route_confirmation",
        confirmation_route,
        {
            "CREATE_ANALYZE_CASE": "create_case",
            "UPDATE_CASE": "update_case",
            "GENERATE_REPORT": "generate_report",
            "DELETE_REPORT": "delete_report",
            "DELETE_DOCUMENT": "delete_document",
        },
    )
    graph.add_edge("create_case", END)
    graph.add_edge("update_case", END)
    graph.add_edge("generate_report", END)
    graph.add_edge("delete_report", END)
    graph.add_edge("delete_document", END)
    return graph.compile()
