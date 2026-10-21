from pathlib import Path

import pytest

from backend.app.care.schemas import ClinicalMetadata
from backend.app.database import CaseRepository, init_database
from backend.app.mcp.factory import create_tool_registry
from backend.app.mcp.schemas import ToolContext
from backend.app.services.container import create_service_container


class FakeRunner:
    def predict(self, image_path: Path, clinical_metadata: ClinicalMetadata) -> dict:
        assert clinical_metadata.grew is True
        return {
            "model_name": "DermFM-Zero-Open",
            "top_predictions": [
                {"label": "basal cell carcinoma", "score": 0.91},
                {"label": "nevus", "score": 0.06},
                {"label": "actinic keratosis", "score": 0.03},
            ],
            "latency_ms": 17.0,
        }


CLINICAL = ClinicalMetadata(
    age=65, gender="MALE", region="FOREARM", itch=True, hurt=False,
    bleed=True, grew=True, changed=True, diameter_1=10, diameter_2=8,
    skin_cancer_history=False, cancer_history=False,
)


@pytest.fixture
def ready_case(tmp_path):
    repository = CaseRepository(init_database("sqlite+pysqlite:///:memory:"))
    services = create_service_container(repository)
    services.cases.data_root = tmp_path / "cases"
    services.visual.runner = FakeRunner()
    sample = Path("third_party/DermFM-Zero/examples/PAT_8_15_820.png").read_bytes()
    case_id = services.cases.create(sample, "image/png")
    visual = services.visual.analyze(case_id, CLINICAL)
    triage = services.triage.evaluate(visual)
    chunks = services.knowledge.search("皮肤专科 优先转诊", top_k=2)
    services.care.recommend(visual, triage, services.knowledge.references(chunks))
    return case_id, services, create_tool_registry(services)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_mcp_list_tools(ready_case):
    _, _, registry = ready_case
    assert {item.name for item in registry.list_tools()} == {
        "get_visual_evidence",
        "get_triage_result",
        "search_guideline",
        "get_care_pathway",
        "get_privacy_report",
            "web_search",
    }


@pytest.mark.anyio
async def test_mcp_visual_tool_uses_existing_service_and_has_no_raw_image(ready_case):
    case_id, services, registry = ready_case
    called = False
    original = services.visual.get

    def tracked(value):
        nonlocal called
        called = True
        return original(value)

    services.visual.get = tracked
    response = await registry.invoke(
        "get_visual_evidence", {"case_id": case_id}, ToolContext(case_id=case_id)
    )
    assert called is True
    assert response.result["model"] == "DermFM-Zero-Open"
    serialized = str(response.result).lower()
    assert "local_image_path" not in serialized
    assert "base64" not in serialized
    assert "data:image" not in serialized


@pytest.mark.anyio
async def test_mcp_triage_tool(ready_case):
    case_id, _, registry = ready_case
    response = await registry.invoke("get_triage_result", {"case_id": case_id})
    assert response.result["level"] == "PRIORITY_REFERRAL"
    assert response.result["policy_version"]


@pytest.mark.anyio
async def test_mcp_guideline_tool(ready_case):
    _, _, registry = ready_case
    response = await registry.invoke("search_guideline", {"query": "下一步转诊"})
    assert response.result["chunks"]
    assert {"id", "title", "content", "source", "score"} <= set(response.result["chunks"][0])
    assert response.result["chunks"][0]["citation"]
    assert response.result["evidence_assembly"]["answer_policy"]


@pytest.mark.anyio
async def test_mcp_care_pathway_tool(ready_case):
    case_id, _, registry = ready_case
    response = await registry.invoke("get_care_pathway", {"case_id": case_id})
    assert response.result["level"] == "PRIORITY_REFERRAL"
    assert "确诊" not in str(response.result)


@pytest.mark.anyio
async def test_mcp_privacy_tool(ready_case):
    case_id, _, registry = ready_case
    response = await registry.invoke("get_privacy_report", {"case_id": case_id})
    assert response.result["raw_image_sent_to_cloud"] is False
    assert response.result["patient_identity_sent_to_cloud"] is False
