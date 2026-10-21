from pathlib import Path

from backend.app.auth import hash_password, login, token_hash, verify_password
from backend.app.database import CaseRepository, init_database
from backend.app.services.report_service import ReportService
from backend.app.services.privacy import EgressGuard


def test_password_session_patient_and_report_lifecycle(tmp_path: Path) -> None:
    repository = CaseRepository(init_database("sqlite+pysqlite:///:memory:"))
    encoded = hash_password("doctor-password")
    assert verify_password("doctor-password", encoded)
    assert not verify_password("wrong", encoded)
    user = repository.create_user("doctor", encoded, "测试医生")
    logged_in = login(repository, "doctor", "doctor-password")
    assert logged_in is not None
    token, public_user = logged_in
    assert public_user["role"] == "clinician"
    assert repository.get_session_user(token_hash(token), __import__("datetime").datetime.utcnow()) is not None

    patient = repository.create_patient(
        name="测试患者", sex="女", age=42, outpatient_no="OP-001", phone="13800138000",
        medical_history="无", allergy_history="无", created_by=user.id,
    )
    image = tmp_path / "lesion.png"
    image.write_bytes(b"image")
    repository.create_case("CASE_TEST", str(image), "abc", "local", "clinician")
    repository.link_case_patient(
        "CASE_TEST", patient.patient_id, "左前臂", "近期增大",
        {
            "age": 42, "gender": "FEMALE", "region": "FOREARM",
            "itch": False, "hurt": False, "bleed": False, "grew": True,
            "changed": True, "diameter_1": 10, "diameter_2": 8,
            "skin_cancer_history": False, "cancer_history": False,
        },
    )

    generated = ReportService(EgressGuard(repository)).generate(
        case_id="CASE_TEST",
        patient=repository.get_case_patient("CASE_TEST"),
        clinical_payload={
            "visual": {"top_predictions": [{"label": "nevus", "score": 0.8}]},
            "triage": {"level": "LOCAL_REVIEW"},
            "care_pathway": {"action": "医生继续评估"},
        },
        model_id="local-template",
    )
    assert "测试患者" in generated.content
    assert "色素痣" in generated.content
    report = repository.create_report(
        case_id="CASE_TEST", patient_id=patient.patient_id, content=generated.content,
        structured=generated.structured, provider=generated.provider, model=generated.model,
        created_by=user.id,
    )
    saved = repository.update_report(report.report_id, generated.content + "\n医生已审核", user.id)
    assert saved["status"] == "final"
    assert repository.delete_report(report.report_id)
    assert repository.get_case("CASE_TEST") is not None
    assert image.exists()


def test_cloud_report_provider_only_receives_masked_structured_payload(monkeypatch) -> None:
    captured = {}

    class FakeProvider:
        name = "fake-cloud"
        model = "fake-model"

        def plan(self, payload, system_prompt):
            captured["payload"] = payload
            captured["system_prompt"] = system_prompt
            return __import__("json").dumps({
                "clinical_summary": "结构化主诉摘要",
                "visual_summary": "结构化视觉摘要",
                "risk_interpretation": "结构化风险摘要",
                "evidence_summary": "结构化证据摘要",
                "recommendations": "建议医生复核",
                "follow_up": "按计划随访",
            }, ensure_ascii=False)

    monkeypatch.setattr(
        "backend.app.services.report_service.create_chat_provider", lambda model_id: FakeProvider()
    )
    patient = {
        "name": "隐私患者", "outpatient_no": "SECRET-001", "phone": "13800138000",
        "age": 45, "sex": "女", "lesion_site": "左前臂",
        "chief_complaint": "隐私患者近期变化，联系电话13800138000",
    }
    result = ReportService(EgressGuard()).generate(
        case_id="CASE_PRIVATE", patient=patient,
        clinical_payload={"visual": {}, "triage": {}, "care_pathway": {}},
        model_id="fake-model",
    )
    serialized = __import__("json").dumps(captured["payload"], ensure_ascii=False)
    assert "隐私患者" not in serialized
    assert "SECRET-001" not in serialized
    assert "13800138000" not in serialized
    assert "[REDACTED_IDENTITY]" in serialized
    assert "隐私患者" in result.content
