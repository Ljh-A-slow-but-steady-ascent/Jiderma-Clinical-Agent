from __future__ import annotations

import time
from typing import Any

from backend.app.care.schemas import ClinicalMetadata
from backend.app.services.care_service import CareService
from backend.app.services.case_service import CaseService
from backend.app.services.knowledge_service import KnowledgeService
from backend.app.services.privacy_service import PrivacyService
from backend.app.services.triage_service import TriagePolicyService
from backend.app.services.visual_service import VisualService
from backend.app.database import CaseRepository


def positive_symptoms(profile: ClinicalMetadata) -> list[str]:
    flags = {
        "瘙痒": profile.itch,
        "疼痛": profile.hurt,
        "出血": profile.bleed,
        "近期增大": profile.grew,
        "外观变化": profile.changed,
        "皮肤肿瘤史": profile.skin_cancer_history,
        "其他肿瘤史": profile.cancer_history,
    }
    return [name for name, active in flags.items() if active]


class CaseAnalysisService:
    """One authoritative image-text analysis workflow shared by HTTP and Harness."""

    def __init__(
        self,
        repository: CaseRepository,
        cases: CaseService,
        visual: VisualService,
        triage: TriagePolicyService,
        knowledge: KnowledgeService,
        care: CareService,
        privacy: PrivacyService,
    ) -> None:
        self.repository = repository
        self.cases = cases
        self.visual = visual
        self.triage = triage
        self.knowledge = knowledge
        self.care = care
        self.privacy = privacy

    def run(self, case_id: str) -> dict[str, Any]:
        total_started = time.perf_counter()
        case_patient = self.repository.get_case_patient(case_id)
        if case_patient is None:
            raise ValueError("多模态分析前必须关联患者和结构化病例")
        profile = ClinicalMetadata.model_validate(case_patient.get("clinical_profile"))

        visual_started = time.perf_counter()
        visual = self.visual.analyze(case_id, clinical_metadata=profile)
        visual_wall = (time.perf_counter() - visual_started) * 1000

        triage_started = time.perf_counter()
        triage = self.triage.evaluate(visual)
        triage_latency = (time.perf_counter() - triage_started) * 1000

        rag_started = time.perf_counter()
        guidelines = self.knowledge.search_patient_context(
            user_query=case_patient.get("chief_complaint", "皮肤病灶辅助鉴别"),
            visual_candidates=[item.label for item in visual.top_predictions[:3]],
            positive_symptoms=positive_symptoms(profile),
            risk_reasons=[triage.level.value, *triage.reason_codes],
            topic="支持依据 冲突依据 鉴别要点 检查依据 转诊依据",
            top_k=5,
        )
        rag_latency = (time.perf_counter() - rag_started) * 1000

        care_started = time.perf_counter()
        case_record = self.repository.get_case(case_id)
        care = self.care.recommend(
            visual,
            triage,
            self.knowledge.references(guidelines),
            mode="patient" if case_record is None else case_record.mode,
        )
        care_latency = (time.perf_counter() - care_started) * 1000
        privacy = self.privacy.report(case_id)

        return {
            "case_id": case_id,
            "visual": visual.model_dump(mode="json"),
            "triage": triage.model_dump(mode="json"),
            "care_pathway": care.model_dump(mode="json"),
            "guidelines": guidelines,
            "privacy": privacy.model_dump(mode="json"),
            "performance": {
                "vision_model_latency_ms": visual.latency_ms,
                "visual_stage_wall_ms": visual_wall,
                "triage_latency_ms": triage_latency,
                "rag_latency_ms": rag_latency,
                "care_pathway_latency_ms": care_latency,
                "total_pretriage_latency_ms": (time.perf_counter() - total_started) * 1000,
            },
        }
