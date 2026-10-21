from __future__ import annotations

from collections.abc import Sequence

from backend.app.care import DeterministicCarePathwayProvider
from backend.app.care.schemas import (
    CarePathway,
    GuidelineReference,
    RequestContext,
    TriageDecision,
    VisualEvidence,
)
from backend.app.database import CaseRepository


class CareService:
    def __init__(
        self,
        provider: DeterministicCarePathwayProvider | None = None,
        repository: CaseRepository | None = None,
    ) -> None:
        self.provider = provider or DeterministicCarePathwayProvider()
        self.repository = repository

    def recommend(
        self,
        evidence: VisualEvidence,
        triage: TriageDecision,
        guidelines: Sequence[GuidelineReference],
        mode: str = "patient",
    ) -> CarePathway:
        result = self.provider.recommend(
            evidence,
            triage,
            guidelines,
            RequestContext(mode=mode),
        )
        if self.repository is not None:
            self.repository.save_care_pathway(evidence.case_id, result)
        return result

    def get(self, case_id: str) -> CarePathway:
        if self.repository is None:
            raise RuntimeError("repository is required to load a care pathway")
        raw = self.repository.get_care_pathway(case_id)
        if raw is None:
            raise KeyError(f"care pathway not found: {case_id}")
        return CarePathway.model_validate(raw)
