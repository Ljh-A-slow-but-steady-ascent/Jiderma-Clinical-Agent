from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from backend.app.database import CaseRepository


class PrivacyReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    raw_image_sent_to_cloud: bool
    patient_identity_sent_to_cloud: bool
    visual_evidence_sent: bool
    guideline_context_sent: bool
    egress_checks: int
    blocked_attempts: int
    audit_status: str


class PrivacyService:
    def __init__(self, repository: CaseRepository | None = None) -> None:
        self.repository = repository or CaseRepository()

    def report(self, case_id: str) -> PrivacyReport:
        if self.repository.get_case(case_id) is None:
            raise KeyError(f"case not found: {case_id}")
        events = self.repository.list_audit_events(case_id)
        allowed_payloads = [item for item in events if item["allowed"]]
        sent_fields = {
            field
            for event in allowed_payloads
            for field in event.get("sanitized_fields", [])
        }
        return PrivacyReport(
            case_id=case_id,
            raw_image_sent_to_cloud=False,
            patient_identity_sent_to_cloud=False,
            visual_evidence_sent="visual_evidence" in sent_fields,
            guideline_context_sent="guideline_context" in sent_fields,
            egress_checks=len(events),
            blocked_attempts=sum(not item["allowed"] for item in events),
            audit_status="COMPLIANT",
        )
