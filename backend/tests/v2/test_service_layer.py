from pathlib import Path

from backend.app.care import CarePathwayLevel
from backend.app.care.schemas import ClinicalMetadata
from backend.app.database import CaseRepository, init_database
from backend.app.services.care_service import CareService
from backend.app.services.case_service import CaseService
from backend.app.services.knowledge_service import KnowledgeService
from backend.app.services.triage_service import TriagePolicyService
from backend.app.services.visual_service import VisualService


class FakeFrozenRunner:
    def predict(self, image_path: Path, clinical_metadata: ClinicalMetadata) -> dict:
        assert image_path.is_file()
        assert clinical_metadata.region == "FOREARM"
        return {
            "model_name": "DermFM-Zero-Open",
            "top_predictions": [
                {"label": "basal cell carcinoma", "score": 0.82},
                {"label": "nevus", "score": 0.12},
                {"label": "actinic keratosis", "score": 0.06},
            ],
            "latency_ms": 19.5,
        }


CLINICAL = ClinicalMetadata(
    age=65, gender="MALE", region="FOREARM", itch=True, hurt=False,
    bleed=True, grew=True, changed=True, diameter_1=10, diameter_2=8,
    skin_cancer_history=False, cancer_history=False,
)


def services(tmp_path):
    repository = CaseRepository(init_database("sqlite+pysqlite:///:memory:"))
    case_service = CaseService(repository, data_root=tmp_path / "cases")
    visual_service = VisualService(repository, case_service, FakeFrozenRunner())
    return repository, case_service, visual_service


def test_rest_and_mcp_ready_services_share_persisted_results(tmp_path):
    repository, case_service, visual_service = services(tmp_path)
    sample = Path("third_party/DermFM-Zero/examples/PAT_8_15_820.png").read_bytes()
    case_id = case_service.create(sample, "image/png")
    visual = visual_service.analyze(case_id, CLINICAL)
    assert visual.clinical_context == CLINICAL
    assert len(visual.top_predictions) == 3
    triage = TriagePolicyService(repository=repository).evaluate(visual)
    chunks = KnowledgeService().search("皮肤病灶需要转诊进一步评估", top_k=2)
    pathway = CareService(repository=repository).recommend(
        visual,
        triage,
        KnowledgeService().references(chunks),
    )

    assert visual_service.get(case_id) == visual
    assert TriagePolicyService(repository=repository).get(case_id) == triage
    assert CareService(repository=repository).get(case_id) == pathway
    assert triage.level == CarePathwayLevel.PRIORITY_REFERRAL


def test_database_never_stores_raw_image_blob_or_path_in_visual_result(tmp_path):
    repository, case_service, visual_service = services(tmp_path)
    sample = Path("third_party/DermFM-Zero/examples/PAT_8_15_820.png").read_bytes()
    case_id = case_service.create(sample, "image/png")
    visual_service.analyze(case_id, CLINICAL)
    raw = repository.get_visual(case_id)
    serialized = str(raw).lower()
    assert "image_path" not in serialized
    assert "base64" not in serialized
    assert "data:image" not in serialized


def test_knowledge_service_returns_local_guideline_schema():
    result = KnowledgeService().search("优先转诊 皮肤专科", top_k=2)
    assert result
    assert {"id", "title", "content", "source", "score"} <= set(result[0])
    assert result[0]["citation"]
