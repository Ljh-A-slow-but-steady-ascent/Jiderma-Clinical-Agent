from __future__ import annotations

from dataclasses import dataclass

from backend.app.database import CaseRepository
from backend.app.services.clinical_extraction_service import ClinicalExtractionService
from backend.app.services.analysis_service import CaseAnalysisService
from backend.app.services.care_service import CareService
from backend.app.services.case_service import CaseService
from backend.app.services.knowledge_service import KnowledgeService
from backend.app.services.privacy import EgressGuard
from backend.app.services.privacy_service import PrivacyService
from backend.app.services.triage_service import TriagePolicyService
from backend.app.services.visual_service import VisualService


@dataclass
class ServiceContainer:
    repository: CaseRepository
    cases: CaseService
    visual: VisualService
    triage: TriagePolicyService
    knowledge: KnowledgeService
    care: CareService
    privacy: PrivacyService
    clinical_extraction: ClinicalExtractionService
    analysis: CaseAnalysisService


def create_service_container(repository: CaseRepository | None = None) -> ServiceContainer:
    repository = repository or CaseRepository()
    cases = CaseService(repository)
    knowledge = KnowledgeService()
    visual = VisualService(repository, cases)
    triage = TriagePolicyService(repository=repository)
    care = CareService(repository=repository)
    privacy = PrivacyService(repository)
    return ServiceContainer(
        repository=repository,
        cases=cases,
        visual=visual,
        triage=triage,
        knowledge=knowledge,
        care=care,
        privacy=privacy,
        clinical_extraction=ClinicalExtractionService(EgressGuard(repository)),
        analysis=CaseAnalysisService(
            repository, cases, visual, triage, knowledge, care, privacy
        ),
    )
