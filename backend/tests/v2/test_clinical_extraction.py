from __future__ import annotations

import json

from backend.app.services.clinical_extraction_service import ClinicalExtractionService
from backend.app.services.privacy import EgressGuard


def test_offline_extraction_preserves_explicit_positive_and_negative_facts() -> None:
    result = ClinicalExtractionService(EgressGuard()).extract(
        narrative=(
            "70岁男性，面部色素性病灶约9×9毫米，近期明显增大且颜色发生变化，"
            "无疼痛、瘙痒和出血，无皮肤癌病史及其他癌症病史。"
        ),
        age=70,
        gender="MALE",
        model_id="auto",
        identity_values=[],
    )
    assert result.missing_fields == []
    assert result.metadata == {
        "age": 70, "gender": "MALE", "region": "FACE",
        "itch": False, "hurt": False, "bleed": False,
        "grew": True, "changed": True,
        "diameter_1": 9.0, "diameter_2": 9.0,
        "skin_cancer_history": False, "cancer_history": False,
    }


def test_cloud_extraction_masks_identity_and_keeps_unknown_fields(monkeypatch) -> None:
    captured: dict = {}

    class FakeProvider:
        name = "fake-cloud"
        model = "fake-model"

        def plan(self, payload, system_prompt):
            captured["payload"] = payload
            captured["system_prompt"] = system_prompt
            return json.dumps({
                "metadata": {
                    "region": "FOREARM", "itch": True, "hurt": None,
                    "bleed": False, "grew": True, "changed": True,
                    "diameter_1": 10, "diameter_2": 8,
                    "skin_cancer_history": False, "cancer_history": False,
                },
                "uncertain_fields": ["hurt"],
            })

    monkeypatch.setattr(
        "backend.app.chat.mock.create_chat_provider", lambda model_id: FakeProvider()
    )
    result = ClinicalExtractionService(EgressGuard()).extract(
        narrative="患者李某，电话13800138000，左前臂病灶近期增大并出现变化。",
        age=45,
        gender="FEMALE",
        model_id="glm-5.3-flash",
        identity_values=["李某", "OP-001", "13800138000"],
    )
    serialized = json.dumps(captured["payload"], ensure_ascii=False)
    assert "李某" not in serialized
    assert "13800138000" not in serialized
    assert "[REDACTED_IDENTITY]" in serialized
    assert result.metadata["age"] == 45
    assert result.metadata["gender"] == "FEMALE"
    assert result.metadata["hurt"] is None
    assert result.missing_fields == ["hurt"]
    assert result.uncertain_fields == ["hurt"]


def test_agent_intake_is_built_from_one_natural_language_message() -> None:
    result = ClinicalExtractionService(EgressGuard()).extract_agent_intake(
        narrative=(
            "患者张三，男，45岁，门诊号 OP001。左前臂病灶约8×6毫米，"
            "无瘦痒，无疼痛，无出血，近期无增大，无颜色变化，"
            "无皮肤癌病史，无其他肿瘤史。"
        ),
        model_id="auto",
    )
    assert result.missing_fields == []
    assert result.intake["patient_name"] == "张三"
    assert result.intake["outpatient_no"] == "OP001"
    assert result.intake["clinical_metadata"]["region"] == "FOREARM"
    assert result.intake["clinical_metadata"]["bleed"] is False


def test_agent_intake_reads_uploaded_pad_style_markdown_fields() -> None:
    result = ClinicalExtractionService(EgressGuard()).extract_agent_intake(
        narrative=(
            "展示病例：黑色素瘤\n年龄：70 岁\n性别：男\n病灶部位：面部\n"
            "病灶尺寸：9.0 mm × 9.0 mm\n瘙痒：否\n疼痛：否\n出血：否\n"
            "近期增大：是\n近期变化：是\n皮肤癌病史：否\n其他癌症病史：否"
        ),
        model_id="auto",
    )
    assert result.intake["age"] == 70
    assert result.intake["sex"] == "男"
    assert result.intake["lesion_site"] == "面部"
    metadata = result.intake["clinical_metadata"]
    assert metadata["diameter_1"] == 9.0
    assert metadata["diameter_2"] == 9.0
    assert metadata["itch"] is False
    assert metadata["hurt"] is False
    assert metadata["bleed"] is False
    assert metadata["grew"] is True
    assert metadata["changed"] is True
    assert result.missing_fields == ["patient_name", "outpatient_no"]
