from pathlib import Path
import json

import httpx
import pytest

from backend.app.api.v2_routes import create_v2_runtime, get_runtime
from backend.app.care.schemas import ClinicalMetadata
from backend.app.database import CaseRepository, init_database
from backend.app.main import app
from backend.app.services.container import create_service_container


class FakeFrozenRunner:
    def predict(self, image_path: Path, clinical_metadata: ClinicalMetadata) -> dict:
        assert image_path.is_file()
        assert clinical_metadata.bleed is True
        return {
            "model_name": "DermFM-Zero-Open",
            "top_predictions": [
                {"label": "basal cell carcinoma", "score": 0.94},
                {"label": "squamous cell carcinoma", "score": 0.04},
                {"label": "actinic keratosis", "score": 0.02},
            ],
            "latency_ms": 18.2,
        }


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client(tmp_path):
    repository = CaseRepository(init_database("sqlite+pysqlite:///:memory:"))
    services = create_service_container(repository)
    services.cases.data_root = tmp_path / "cases"
    services.visual.runner = FakeFrozenRunner()
    test_runtime = create_v2_runtime(services)
    async def override_runtime():
        return test_runtime

    app.dependency_overrides[get_runtime] = override_runtime
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as value:
        yield value
    app.dependency_overrides.clear()


async def _create_structured_case(client, image: bytes, outpatient_no: str) -> dict:
    patient = await client.post("/v1/patients", json={
        "name": "测试患者", "sex": "男", "age": 65,
        "outpatient_no": outpatient_no, "phone": "",
        "medical_history": "", "allergy_history": "",
    })
    assert patient.status_code == 200, patient.text
    metadata = {
        "age": 65, "gender": "MALE", "region": "FOREARM",
        "itch": True, "hurt": False, "bleed": True, "grew": True,
        "changed": True, "diameter_1": 10, "diameter_2": 8,
        "skin_cancer_history": False, "cancer_history": False,
    }
    return await client.post(
        "/v1/cases",
        files={"image": ("lesion.png", image, "image/png")},
        data={
            "mode": "clinician", "retention_mode": "session",
            "patient_id": patient.json()["patient_id"], "lesion_site": "左前臂",
            "chief_complaint": "近期增大并出血",
            "clinical_metadata_json": json.dumps(metadata),
        },
    )


@pytest.mark.anyio
async def test_complete_v2_decision_interaction_mcp_privflow_flow(client):
    image = Path("third_party/DermFM-Zero/examples/PAT_8_15_820.png").read_bytes()
    created = await _create_structured_case(client, image, "E2E-001")
    assert created.status_code == 200
    case_id = created.json()["case_id"]
    assert created.json()["raw_image_local_only"] is True

    analyzed = await client.post(f"/v1/cases/{case_id}/analyze")
    assert analyzed.status_code == 200, analyzed.text
    analysis = analyzed.json()
    assert analysis["visual"]["clinical_context"]["bleed"] is True
    assert len(analysis["visual"]["top_predictions"]) == 3
    assert analysis["triage"]["level"] == "PRIORITY_REFERRAL"
    assert analysis["care_pathway"]["level"] == analysis["triage"]["level"]
    assert analysis["guidelines"]
    assert analysis["performance"]["total_pretriage_latency_ms"] >= 0

    models = await client.get("/v1/chat/models")
    assert models.status_code == 200
    assert any(item["id"] == "glm-5.3-flash" for item in models.json())

    preview = await client.get(f"/v1/cases/{case_id}/cloud-payload")
    assert preview.status_code == 200
    serialized = preview.text.lower()
    assert preview.json()["raw_image_included"] is False
    assert "local_image_path" not in serialized
    assert "data:image" not in serialized

    privacy = (await client.get(f"/v1/cases/{case_id}/privacy")).json()
    assert privacy["raw_image_sent_to_cloud"] is False
    assert privacy["patient_identity_sent_to_cloud"] is False
    assert privacy["visual_evidence_sent"] is True

    audit = (await client.get("/v1/audit", params={"case_id": case_id})).json()
    assert audit
    audit_text = str(audit).lower()
    assert "local_image_path" not in audit_text
    assert "data:image" not in audit_text
    assert "/data/cases/" not in audit_text
