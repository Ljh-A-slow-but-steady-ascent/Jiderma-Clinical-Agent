from __future__ import annotations

from backend.app.care.schemas import CarePathwayLevel, TriageDecision, VisualEvidence
from backend.app.database import CaseRepository


class TriagePolicyService:
    """Deterministic V2 policy. Scores are candidate-set matches, not disease probabilities."""

    POLICY_VERSION = "dermaguard-v2-demo-policy-1"
    PRIORITY_TOP1 = frozenset(
        {
            "melanoma",
            "basal cell carcinoma",
            "squamous cell carcinoma",
        }
    )
    REFERRAL_TOP1 = frozenset({"actinic keratosis"})

    def __init__(self, min_quality: float = 0.4, repository: CaseRepository | None = None) -> None:
        self.min_quality = min_quality
        self.repository = repository

    def evaluate(self, evidence: VisualEvidence) -> TriageDecision:
        labels = [item.label.strip().lower() for item in evidence.top_predictions]
        top1 = labels[0] if labels else None
        metadata: dict[str, float | str | bool | None] = {
            "image_quality": evidence.image_quality,
            "top1_label": top1,
            "top1_score": evidence.top_predictions[0].score if evidence.top_predictions else None,
            "scores_are_calibrated_probabilities": False,
            "prediction_mode": evidence.prediction_mode,
        }
        if evidence.image_quality < self.min_quality:
            level = CarePathwayLevel.RETAKE
            reasons = ["IMAGE_QUALITY_BELOW_THRESHOLD"]
        elif not evidence.supports_triage or not labels:
            level = CarePathwayLevel.RETAKE
            reasons = ["VISUAL_EVIDENCE_UNAVAILABLE"]
        elif top1 in self.PRIORITY_TOP1:
            level = CarePathwayLevel.PRIORITY_REFERRAL
            reasons = ["HIGH_RISK_PATTERN_TOP1", f"CANDIDATE_{top1.upper().replace(' ', '_')}"]
        elif top1 in self.REFERRAL_TOP1:
            level = CarePathwayLevel.REFERRAL
            reasons = ["SPECIALIST_DIFFERENTIATION_RECOMMENDED", "CANDIDATE_ACTINIC_KERATOSIS"]
        elif any(
            item.label.strip().lower() in self.PRIORITY_TOP1
            and item.score >= 0.25
            for item in evidence.top_predictions[:3]
        ):
            level = CarePathwayLevel.REFERRAL
            reasons = ["HIGH_RISK_PATTERN_IN_TOP3", "SPECIALIST_DIFFERENTIATION_RECOMMENDED"]
        else:
            level = CarePathwayLevel.LOCAL_REVIEW
            reasons = ["NO_PRIORITY_PATTERN_IN_CURATED_TOP3", "CLINICAL_REVIEW_STILL_REQUIRED"]
        result = TriageDecision(
            case_id=evidence.case_id,
            level=level,
            reason_codes=reasons,
            policy_version=self.POLICY_VERSION,
            confidence_metadata=metadata,
        )
        if self.repository is not None:
            self.repository.save_triage(evidence.case_id, result)
        return result

    def get(self, case_id: str) -> TriageDecision:
        if self.repository is None:
            raise RuntimeError("repository is required to load a triage result")
        raw = self.repository.get_triage(case_id)
        if raw is None:
            raise KeyError(f"triage result not found: {case_id}")
        return TriageDecision.model_validate(raw)
