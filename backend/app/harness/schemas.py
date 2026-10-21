from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.care.schemas import ClinicalMetadata


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


RunStatus = Literal[
    "RUNNING", "WAITING_USER", "WAITING_CONFIRM", "PAUSED", "COMPLETED", "FAILED", "CANCELLED"
]


class AgentIntake(StrictModel):
    patient_name: str = Field(min_length=1, max_length=120)
    sex: Literal["男", "女"]
    age: int = Field(ge=0, le=130)
    outpatient_no: str = Field(min_length=1, max_length=80)
    phone: str = Field(default="", max_length=40)
    medical_history: str = Field(default="", max_length=3000)
    allergy_history: str = Field(default="", max_length=2000)
    lesion_site: str = Field(min_length=1, max_length=120)
    chief_complaint: str = Field(min_length=10, max_length=5000)
    clinical_metadata: ClinicalMetadata


class AgentEvent(StrictModel):
    sequence: int = Field(ge=1)
    event_type: str
    node: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class PendingAction(StrictModel):
    action_id: str
    kind: Literal[
        "CREATE_ANALYZE_CASE",
        "UPDATE_CASE",
        "GENERATE_REPORT",
        "DELETE_REPORT",
        "DELETE_DOCUMENT",
    ]
    title: str
    summary: str
    preview: dict[str, Any] = Field(default_factory=dict)
    destructive: bool = False


class HarnessResponse(StrictModel):
    run_id: str
    status: RunStatus
    current_node: str
    case_id: str | None = None
    message: str
    version: int = Field(ge=1)
    pending_action: PendingAction | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    events: list[AgentEvent] = Field(default_factory=list)

