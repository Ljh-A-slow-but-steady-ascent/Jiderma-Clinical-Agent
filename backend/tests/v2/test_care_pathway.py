import pytest

from backend.app.care import CarePathwayLevel
from backend.app.care.schemas import CandidatePrediction, GuidelineReference, VisualEvidence
from backend.app.services.care_service import CareService
from backend.app.services.triage_service import TriagePolicyService


def evidence(label: str = "nevus", quality: float = 0.9, rest: list[str] | None = None):
    labels = [label, *(rest or ["seborrheic keratosis", "actinic keratosis"])]
    return VisualEvidence(
        case_id="CASE_TEST",
        model="DermFM-Zero-Open",
        candidate_set="pad6",
        top_predictions=[
            CandidatePrediction(label=item, score=max(0.01, 0.8 - index * 0.2))
            for index, item in enumerate(labels)
        ],
        image_quality=quality,
        supports_triage=True,
        latency_ms=18.0,
    )


def test_low_quality_requires_retake():
    assert TriagePolicyService().evaluate(evidence(quality=0.2)).level == CarePathwayLevel.RETAKE


@pytest.mark.parametrize("label", ["melanoma", "basal cell carcinoma", "squamous cell carcinoma"])
def test_high_risk_top1_requires_priority_referral(label):
    result = TriagePolicyService().evaluate(evidence(label))
    assert result.level == CarePathwayLevel.PRIORITY_REFERRAL


def test_high_risk_in_top3_requires_referral_not_priority():
    result = TriagePolicyService().evaluate(
        evidence("nevus", rest=["melanoma", "seborrheic keratosis"])
    )
    assert result.level == CarePathwayLevel.REFERRAL


def test_multimodal_model_ignores_low_score_high_risk_tail():
    visual = evidence("nevus", rest=["melanoma", "seborrheic keratosis"])
    visual.top_predictions[0].score = 0.86
    visual.top_predictions[1].score = 0.13
    visual.top_predictions[2].score = 0.01
    result = TriagePolicyService().evaluate(visual)
    assert result.level == CarePathwayLevel.LOCAL_REVIEW


def test_benign_top3_keeps_local_review_language_safe():
    visual = evidence("nevus", rest=["seborrheic keratosis", "nevus"])
    triage = TriagePolicyService().evaluate(visual)
    pathway = CareService().recommend(
        visual,
        triage,
        [GuidelineReference(id="GUIDE_1", title="基层复核", source="demo")],
    )
    assert triage.level == CarePathwayLevel.LOCAL_REVIEW
    assert "基层医生" in pathway.action
    assert "无需就医" not in pathway.model_dump_json()


def test_care_pathway_cannot_change_policy_level():
    visual = evidence("melanoma")
    triage = TriagePolicyService().evaluate(visual)
    pathway = CareService().recommend(visual, triage, [])
    assert pathway.level == triage.level == CarePathwayLevel.PRIORITY_REFERRAL
    assert "确诊" not in pathway.model_dump_json()
