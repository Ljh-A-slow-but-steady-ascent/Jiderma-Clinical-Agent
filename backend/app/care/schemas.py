from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CarePathwayLevel(str, Enum):
    RETAKE = "RETAKE"
    LOCAL_REVIEW = "LOCAL_REVIEW"
    REFERRAL = "REFERRAL"
    PRIORITY_REFERRAL = "PRIORITY_REFERRAL"


class CandidatePrediction(StrictModel):
    label: str
    score: float = Field(ge=0, le=1)


class ClinicalMetadata(StrictModel):
    """De-identified PAD-compatible facts shared by inference and retrieval."""

    age: int = Field(ge=0, le=130)
    gender: Literal["MALE", "FEMALE"]
    region: Literal[
        "ABDOMEN", "ARM", "BACK", "CHEST", "EAR", "FACE", "FOOT",
        "FOREARM", "HAND", "LIP", "NECK", "NOSE", "SCALP", "THIGH",
    ]
    itch: bool
    hurt: bool
    bleed: bool
    grew: bool
    changed: bool
    diameter_1: float = Field(gt=0, le=200)
    diameter_2: float = Field(gt=0, le=200)
    skin_cancer_history: bool
    cancer_history: bool


class VisualEvidence(StrictModel):
    case_id: str
    model: str
    candidate_set: str
    top_predictions: list[CandidatePrediction]
    image_quality: float = Field(ge=0, le=1)
    supports_triage: bool
    latency_ms: float = Field(ge=0)
    local_only: bool = True
    warnings: list[str] = Field(default_factory=list)
    prediction_mode: Literal["supervised_multimodal_pad"] = "supervised_multimodal_pad"
    input_modality: Literal["clinical_image+structured_text"] = "clinical_image+structured_text"
    clinical_context: ClinicalMetadata | None = None


class TriageDecision(StrictModel):
    case_id: str
    level: CarePathwayLevel
    reason_codes: list[str]
    policy_version: str
    confidence_metadata: dict[str, float | str | bool | None] = Field(default_factory=dict)


class RequestContext(StrictModel):
    mode: Literal["patient", "clinician"] = "patient"
    locale: str = "zh-CN"


class GuidelineReference(StrictModel):
    id: str
    title: str
    source: str


class CarePathway(StrictModel):
    case_id: str
    level: CarePathwayLevel
    action: str
    destination_type: str
    urgency: str
    recommended_next_steps: list[str]
    reason_codes: list[str]
    guideline_references: list[GuidelineReference]
    disclaimer: str = "本系统仅用于辅助筛查和分级转诊，不构成医学诊断。"
