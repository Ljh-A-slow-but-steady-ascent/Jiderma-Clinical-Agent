from pathlib import Path

import pytest

from backend.app.care.schemas import CandidatePrediction, VisualEvidence
from backend.app.database import CaseRepository, init_database
from backend.app.mcp.server import create_mcp_server
from backend.app.services.care_service import CareService
from backend.app.services.case_service import CaseService
from backend.app.services.container import create_service_container
from backend.app.services.knowledge_service import KnowledgeService
from backend.app.services.triage_service import TriagePolicyService


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def mcp_case(tmp_path):
    repository = CaseRepository(init_database(f"sqlite:///{tmp_path / 'mcp.db'}"))
    case_service = CaseService(repository, data_root=tmp_path / "cases")
    sample = Path("third_party/DermFM-Zero/examples/PAT_8_15_820.png").read_bytes()
    case_id = case_service.create(sample, "image/png")
    visual = VisualEvidence(
        case_id=case_id,
        model="local_multimodal_classifier",
        candidate_set="pad_clinical_multimodal_v1",
        top_predictions=[
            CandidatePrediction(label="basal cell carcinoma", score=0.9),
            CandidatePrediction(label="nevus", score=0.07),
            CandidatePrediction(label="actinic keratosis", score=0.03),
        ],
        image_quality=0.9,
        supports_triage=True,
        latency_ms=18.0,
    )
    repository.save_visual(case_id, visual)
    triage = TriagePolicyService(repository=repository).evaluate(visual)
    knowledge = KnowledgeService()
    chunks = knowledge.search("优先转诊", top_k=2)
    CareService(repository=repository).recommend(
        visual, triage, knowledge.references(chunks)
    )
    return case_id, repository


@pytest.mark.anyio
async def test_real_mcp_sdk_lists_and_calls_all_tools(mcp_case):
    case_id, repository = mcp_case
    server = create_mcp_server(create_service_container(repository))
    assert {item.name for item in await server.list_tools()} == {
        "get_visual_evidence",
        "get_triage_result",
        "search_guideline",
        "get_care_pathway",
        "get_privacy_report",
            "web_search",
    }
    calls = (
        ("get_visual_evidence", {"case_id": case_id}),
        ("get_triage_result", {"case_id": case_id}),
        ("search_guideline", {"query": "为什么优先转诊", "case_id": case_id}),
        ("get_care_pathway", {"case_id": case_id}),
        ("get_privacy_report", {"case_id": case_id}),
    )
    for name, arguments in calls:
        assert await server.call_tool(name, arguments)
