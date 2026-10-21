from __future__ import annotations

from typing import Protocol, Sequence

from backend.app.care.schemas import (
    CarePathway,
    CarePathwayLevel,
    GuidelineReference,
    RequestContext,
    TriageDecision,
    VisualEvidence,
)


class CarePathwayProvider(Protocol):
    def recommend(
        self,
        visual_evidence: VisualEvidence,
        triage_result: TriageDecision,
        guideline_context: Sequence[GuidelineReference],
        request_context: RequestContext,
    ) -> CarePathway: ...


class DeterministicCarePathwayProvider:
    """Map immutable policy output to a safe next-care path; no LLM is involved."""

    _PATHS = {
        CarePathwayLevel.RETAKE: {
            "action": "请重新拍摄清晰、光线均匀的病灶图片后再分析。",
            "destination_type": "local_image_retake",
            "urgency": "retake_required",
            "steps": ["在自然光或均匀照明下拍摄", "保持镜头稳定并对焦病灶", "避免滤镜和过度裁剪"],
        },
        CarePathwayLevel.LOCAL_REVIEW: {
            "action": "建议由基层医生结合病史和现场检查继续评估。",
            "destination_type": "primary_care",
            "urgency": "routine_review",
            "steps": ["由基层医生核对病史和症状", "结合现场检查判断是否需要专科意见", "如病灶变化或症状加重应及时复评"],
        },
        CarePathwayLevel.REFERRAL: {
            "action": "建议转诊至具备皮肤专科能力的上级医疗机构进一步鉴别。",
            "destination_type": "dermatology_capable_facility",
            "urgency": "referral",
            "steps": ["携带本次筛查摘要就诊", "接受皮肤专科评估", "由医生决定是否需要皮肤镜或其他检查"],
        },
        CarePathwayLevel.PRIORITY_REFERRAL: {
            "action": "建议优先转诊至具备皮肤专科及进一步检查条件的医疗机构。",
            "destination_type": "advanced_dermatology_facility",
            "urgency": "priority_referral",
            "steps": ["尽快联系具备皮肤专科能力的上级机构", "接受专科和皮肤镜评估", "由医生判断是否需要活检或病理检查"],
        },
    }

    def recommend(
        self,
        visual_evidence: VisualEvidence,
        triage_result: TriageDecision,
        guideline_context: Sequence[GuidelineReference],
        request_context: RequestContext,
    ) -> CarePathway:
        if visual_evidence.case_id != triage_result.case_id:
            raise ValueError("visual evidence and triage result belong to different cases")
        definition = self._PATHS[triage_result.level]
        return CarePathway(
            case_id=triage_result.case_id,
            level=triage_result.level,
            action=definition["action"],
            destination_type=definition["destination_type"],
            urgency=definition["urgency"],
            recommended_next_steps=list(definition["steps"]),
            reason_codes=triage_result.reason_codes,
            guideline_references=list(guideline_context),
        )
