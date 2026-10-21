from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Annotated, Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from backend.app.care.schemas import CarePathway, ClinicalMetadata, TriageDecision, VisualEvidence
from backend.app.auth import require_user
from backend.app.chat import chat_model_catalog
from backend.app.harness import AgentIntake, ClinicalAgentHarness, HarnessResponse
from backend.app.mcp.factory import create_tool_registry
from backend.app.services.container import ServiceContainer, create_service_container
from backend.app.services.clinical_extraction_service import field_labels
from backend.app.services.dermfm_candidate_sets import DISPLAY_LABELS_ZH
from backend.app.services.privacy import EgressGuard, EgressRequest
from backend.app.services.privacy_service import PrivacyReport
from backend.app.services.report_service import ReportService


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CaseCreated(StrictModel):
    case_id: str
    mode: Literal["patient", "clinician"]
    retention_mode: Literal["session", "local"]
    raw_image_local_only: bool = True


class PatientCreate(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    sex: Literal["男", "女", "其他"]
    age: int = Field(ge=0, le=130)
    outpatient_no: str = Field(min_length=1, max_length=80)
    phone: str = Field(default="", max_length=40)
    medical_history: str = Field(default="", max_length=3000)
    allergy_history: str = Field(default="", max_length=2000)


class ClinicalExtractionRequest(StrictModel):
    narrative: str = Field(min_length=10, max_length=5000)
    age: int = Field(ge=0, le=130)
    gender: Literal["MALE", "FEMALE"]
    patient_name: str = Field(default="", max_length=120)
    outpatient_no: str = Field(default="", max_length=80)
    phone: str = Field(default="", max_length=40)
    model: Literal[
        "auto", "deepseek-flash", "deepseek-v4-pro", "glm-5.3-flash", "glm-4-flash", "openai/gpt-oss-120b",
        "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "gemini-flash",
    ] = "auto"


class ClinicalExtractionResponse(StrictModel):
    metadata: dict[str, Any]
    missing_fields: list[str]
    missing_field_labels: list[str]
    uncertain_fields: list[str]
    provider: str
    model: str
    privacy_status: Literal["SAFE"] = "SAFE"
    sanitized_fields: list[str] = Field(default_factory=list)


class AgentIntakeExtractionRequest(StrictModel):
    narrative: str = Field(min_length=10, max_length=5000)
    model: str = Field(default="auto", max_length=120)


class AgentIntakeExtractionResponse(StrictModel):
    intake: dict[str, Any]
    complete: bool
    missing_fields: list[str]
    missing_field_labels: list[str]
    uncertain_fields: list[str]
    provider: str
    model: str
    privacy_status: Literal["SAFE"] = "SAFE"
    sanitized_fields: list[str] = Field(default_factory=list)


class AgentDocumentUpdateRequest(StrictModel):
    content: str = Field(min_length=1, max_length=60000)
    title: str = Field(default="", max_length=160)


class CaseUpdateRequest(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    sex: Literal["男", "女"] | None = None
    age: int | None = Field(default=None, ge=0, le=130)
    outpatient_no: str | None = Field(default=None, min_length=1, max_length=80)
    phone: str | None = Field(default=None, max_length=40)
    medical_history: str | None = Field(default=None, max_length=3000)
    allergy_history: str | None = Field(default=None, max_length=2000)
    lesion_site: str | None = Field(default=None, max_length=120)
    chief_complaint: str | None = Field(default=None, max_length=5000)
    clinical_profile: ClinicalMetadata | None = None


class CaseDeleteRequest(StrictModel):
    confirm_case_id: str


class FolderCreateRequest(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    parent_id: str | None = Field(default=None, max_length=64)


class WorkspaceMoveRequest(StrictModel):
    resource_type: Literal["case", "report", "document"]
    resource_id: str = Field(min_length=1, max_length=100)
    folder_id: str | None = Field(default=None, max_length=64)


class ReportGenerateRequest(StrictModel):
    model: str = Field(default="local-template", max_length=120)


class ReportUpdateRequest(StrictModel):
    content: str = Field(min_length=1, max_length=50_000)
    status: Literal["draft", "final"] = "final"


class AgentActionRequest(StrictModel):
    action_id: str = Field(min_length=1, max_length=80)


class AgentResumeRequest(StrictModel):
    message: str = Field(min_length=1, max_length=5000)
    model: str = Field(default="auto", max_length=120)
    detail_level: Literal["concise", "detailed"] = "concise"


class PerformanceMetrics(StrictModel):
    vision_model_latency_ms: float = Field(ge=0)
    visual_stage_wall_ms: float = Field(ge=0)
    triage_latency_ms: float = Field(ge=0)
    rag_latency_ms: float = Field(ge=0)
    care_pathway_latency_ms: float = Field(ge=0)
    total_pretriage_latency_ms: float = Field(ge=0)


class AnalysisResult(StrictModel):
    case_id: str
    visual: VisualEvidence
    triage: TriageDecision
    care_pathway: CarePathway
    guidelines: list[dict[str, Any]]
    privacy: PrivacyReport
    performance: PerformanceMetrics


@dataclass
class V2Runtime:
    services: ServiceContainer
    guard: EgressGuard
    harness: ClinicalAgentHarness
    tasks: set[asyncio.Task]


def create_v2_runtime(services: ServiceContainer | None = None) -> V2Runtime:
    services = services or create_service_container()
    guard = EgressGuard(services.repository)
    registry = create_tool_registry(services)
    return V2Runtime(
        services=services, guard=guard,
        harness=ClinicalAgentHarness(services, registry, guard),
        tasks=set(),
    )


runtime = create_v2_runtime()
router = APIRouter(prefix="/v1", dependencies=[Depends(require_user)])


async def get_runtime() -> V2Runtime:
    return runtime


Runtime = Annotated[V2Runtime, Depends(get_runtime)]


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=422, detail=str(exc))


def _conversation_history(raw: str) -> list[dict[str, str]]:
    if not raw.strip():
        return []
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError("对话上下文必须是列表")
    history: list[dict[str, str]] = []
    for item in value[-12:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            continue
        message = str(item.get("message", "")).strip()
        if message:
            history.append({"role": str(item["role"]), "message": message[:4000]})
    return history


@router.post("/clinical-metadata/extract", response_model=ClinicalExtractionResponse)
async def extract_clinical_metadata(
    payload: ClinicalExtractionRequest, app: Runtime
) -> ClinicalExtractionResponse:
    try:
        result = app.services.clinical_extraction.extract(
            narrative=payload.narrative.strip(),
            age=payload.age,
            gender=payload.gender,
            model_id=payload.model,
            identity_values=[payload.patient_name, payload.outpatient_no, payload.phone],
        )
        return ClinicalExtractionResponse(
            metadata=result.metadata,
            missing_fields=result.missing_fields,
            missing_field_labels=field_labels(result.missing_fields),
            uncertain_fields=result.uncertain_fields,
            provider=result.provider,
            model=result.model,
            privacy_status="SAFE",
            sanitized_fields=result.sanitized_fields,
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/agent/intake/extract", response_model=AgentIntakeExtractionResponse)
async def extract_agent_intake(
    payload: AgentIntakeExtractionRequest, app: Runtime
) -> AgentIntakeExtractionResponse:
    try:
        result = app.services.clinical_extraction.extract_agent_intake(
            narrative=payload.narrative, model_id=payload.model
        )
        return AgentIntakeExtractionResponse(
            intake=result.intake,
            complete=not result.missing_fields,
            missing_fields=result.missing_fields,
            missing_field_labels=field_labels(result.missing_fields),
            uncertain_fields=result.uncertain_fields,
            provider=result.provider,
            model=result.model,
            sanitized_fields=result.sanitized_fields,
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/cases", response_model=CaseCreated)
async def create_case(
    app: Runtime,
    image: UploadFile = File(...),
    mode: Literal["patient", "clinician"] = Form("clinician"),
    retention_mode: Literal["session", "local"] = Form("local"),
    patient_id: str | None = Form(default=None),
    lesion_site: str = Form(default="未填写"),
    chief_complaint: str = Form(default="未填写"),
    clinical_metadata_json: str = Form(default=""),
) -> CaseCreated:
    try:
        if not patient_id and os.getenv("DERMAGUARD_AUTH_DISABLED", "0") != "1":
            raise ValueError("图像分析前必须先关联患者信息")
        profile: ClinicalMetadata | None = None
        if patient_id:
            patient = app.services.repository.get_patient(patient_id)
            if patient is None:
                raise KeyError(f"patient not found: {patient_id}")
            if not clinical_metadata_json.strip():
                raise ValueError("请在图像分析前完整填写结构化病例信息")
            profile = ClinicalMetadata.model_validate(json.loads(clinical_metadata_json))
            expected_gender = {"男": "MALE", "女": "FEMALE"}.get(patient.sex)
            if expected_gender is None:
                raise ValueError("当前多模态模型的官方病例格式仅接受男性或女性")
            if profile.age != patient.age or profile.gender != expected_gender:
                raise ValueError("结构化病例的年龄或性别与患者档案不一致")
        content = await image.read(10 * 1024 * 1024 + 1)
        case_id = app.services.cases.create(
            content,
            image.content_type or "application/octet-stream",
            mode=mode,
            retention_mode=retention_mode,
        )
        if patient_id:
            assert profile is not None
            app.services.repository.link_case_patient(
                case_id,
                patient_id,
                lesion_site.strip() or "未填写",
                chief_complaint.strip() or "未填写",
                profile.model_dump(mode="json"),
            )
        return CaseCreated(case_id=case_id, mode=mode, retention_mode=retention_mode)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/patients")
async def create_patient(payload: PatientCreate, app: Runtime, user: dict = Depends(require_user)) -> dict[str, Any]:
    try:
        patient = app.services.repository.create_patient(
            name=payload.name.strip(), sex=payload.sex, age=payload.age,
            outpatient_no=payload.outpatient_no.strip(), phone=payload.phone.strip() or None,
            medical_history=payload.medical_history.strip(), allergy_history=payload.allergy_history.strip(),
            created_by=int(user["id"]),
        )
        return app.services.repository.patient_public(patient)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/patients")
async def list_patients(app: Runtime, query: str = Query(default="", max_length=120)) -> list[dict[str, Any]]:
    return app.services.repository.list_patients(query=query)


@router.get("/cases")
async def list_cases(
    app: Runtime, user: dict = Depends(require_user),
    query: str = Query(default="", max_length=120),
    sex: str = Query(default="", max_length=20),
    lesion_site: str = Query(default="", max_length=120),
    min_age: int | None = Query(default=None, ge=0, le=130),
    max_age: int | None = Query(default=None, ge=0, le=130),
    diagnosis: str = Query(default="", max_length=120),
    triage_level: str = Query(default="", max_length=40),
    folder_id: str = Query(default="", max_length=64),
) -> list[dict[str, Any]]:
    normalized_diagnosis = next(
        (key for key, label in DISPLAY_LABELS_ZH.items() if diagnosis.strip() and diagnosis.strip() in label),
        diagnosis,
    )
    items = app.services.repository.list_cases(
        query=query, sex=sex, lesion_site=lesion_site, min_age=min_age,
        max_age=max_age, diagnosis=normalized_diagnosis, triage_level=triage_level,
        folder_id=folder_id, user_id=int(user["id"]),
    )
    locations = app.services.repository.workspace_locations(int(user["id"]), "case")
    return [{**item, "folder_id": locations.get(item["case_id"])} for item in items]


@router.get("/cases/{case_id}/patient")
async def get_case_patient(case_id: str, app: Runtime) -> dict[str, Any]:
    patient = app.services.repository.get_case_patient(case_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="该病例尚未关联患者")
    return patient


@router.put("/cases/{case_id}")
async def update_case(case_id: str, payload: CaseUpdateRequest, app: Runtime) -> dict[str, Any]:
    try:
        return app.services.repository.update_case_record(
            case_id, payload.model_dump(exclude_unset=True, mode="json")
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.delete("/cases/{case_id}")
async def delete_case(
    case_id: str, payload: CaseDeleteRequest, app: Runtime,
    user: dict = Depends(require_user),
) -> dict[str, Any]:
    if payload.confirm_case_id != case_id:
        raise HTTPException(status_code=409, detail="病例编号确认不匹配")
    try:
        app.services.cases.delete(case_id, int(user["id"]))
        return {"status": "deleted", "case_id": case_id, "reports_and_image_deleted": True}
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/cases/{case_id}/image")
async def get_case_image(
    case_id: str, app: Runtime, user: dict = Depends(require_user),
) -> Response:
    try:
        if not app.services.repository.case_belongs_to_user(case_id, int(user["id"])):
            raise KeyError("病例不存在或当前账号无权访问")
        path = app.services.cases.get_local_image_path(case_id)
        media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "application/octet-stream")
        return Response(path.read_bytes(), media_type=media_type)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/cases/{case_id}/analyze", response_model=AnalysisResult)
async def analyze_case(case_id: str, app: Runtime) -> AnalysisResult:
    try:
        return AnalysisResult.model_validate(app.services.analysis.run(case_id))
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/cases/{case_id}/visual", response_model=VisualEvidence)
async def get_visual(case_id: str, app: Runtime) -> VisualEvidence:
    try:
        return app.services.visual.get(case_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/cases/{case_id}/triage", response_model=TriageDecision)
async def get_triage(case_id: str, app: Runtime) -> TriageDecision:
    try:
        return app.services.triage.get(case_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/cases/{case_id}/care-pathway", response_model=CarePathway)
async def get_care_pathway(case_id: str, app: Runtime) -> CarePathway:
    try:
        return app.services.care.get(case_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/cases/{case_id}/privacy", response_model=PrivacyReport)
async def get_privacy(case_id: str, app: Runtime) -> PrivacyReport:
    try:
        return app.services.privacy.report(case_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/cases/{case_id}/cloud-payload")
async def view_cloud_payload(case_id: str, app: Runtime) -> dict[str, Any]:
    try:
        payload = {
            "visual_evidence": app.services.visual.get(case_id).model_dump(mode="json"),
            "triage_result": app.services.triage.get(case_id).model_dump(mode="json"),
            "care_pathway": app.services.care.get(case_id).model_dump(mode="json"),
        }
        patient = app.services.repository.get_case_patient(case_id)
        if patient and patient.get("clinical_profile"):
            payload["clinical_profile"] = patient["clinical_profile"]
        decision = app.guard.enforce(
            EgressRequest(
                channel="llm",
                destination="cloud_payload_preview",
                payload=payload,
                case_id=case_id,
            )
        )
        return {
            "payload": decision.sanitized_payload,
            "raw_image_included": False,
            "patient_identity_included": False,
            "egress_latency_ms": decision.latency_ms,
        }
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/agent/runs", response_model=HarnessResponse)
async def create_agent_run(
    app: Runtime,
    user: dict = Depends(require_user),
    message: str = Form(default=""),
    case_id: str = Form(default=""),
    model: str = Form(default="auto"),
    detail_level: Literal["concise", "detailed"] = Form(default="concise"),
    intake_json: str = Form(default=""),
    conversation_id: str = Form(default=""),
    history_json: str = Form(default=""),
    image: UploadFile | None = File(default=None),
) -> HarnessResponse:
    try:
        intake = AgentIntake.model_validate_json(intake_json) if intake_json.strip() else None
        image_bytes = await image.read(10 * 1024 * 1024 + 1) if image else None
        return await app.harness.start(
            user_id=int(user["id"]), message=message,
            case_id=case_id.strip() or None, model=model,
            detail_level=detail_level, image_bytes=image_bytes,
            image_content_type=image.content_type if image else None, intake=intake,
            conversation_id=conversation_id.strip() or None,
            conversation_history=_conversation_history(history_json),
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/agent/runs/dispatch", response_model=HarnessResponse)
async def dispatch_agent_run(
    app: Runtime,
    user: dict = Depends(require_user),
    message: str = Form(default=""),
    case_id: str = Form(default=""),
    model: str = Form(default="auto"),
    detail_level: Literal["concise", "detailed"] = Form(default="concise"),
    intake_json: str = Form(default=""),
    conversation_id: str = Form(default=""),
    history_json: str = Form(default=""),
    image: UploadFile | None = File(default=None),
) -> HarnessResponse:
    """Return a run id immediately; the UI polls persisted graph events."""
    try:
        intake = AgentIntake.model_validate_json(intake_json) if intake_json.strip() else None
        image_bytes = await image.read(10 * 1024 * 1024 + 1) if image else None
        normalized_case_id = case_id.strip() or None
        initial = app.harness.create_run(
            user_id=int(user["id"]), case_id=normalized_case_id, message=message
        )
        task = asyncio.create_task(app.harness.execute(
            run_id=initial.run_id, user_id=int(user["id"]), message=message,
            case_id=normalized_case_id, model=model, detail_level=detail_level,
            image_bytes=image_bytes,
            image_content_type=image.content_type if image else None, intake=intake,
            conversation_id=conversation_id.strip() or initial.run_id,
            conversation_history=_conversation_history(history_json),
        ))
        app.tasks.add(task)
        task.add_done_callback(app.tasks.discard)
        task.add_done_callback(lambda finished: finished.exception() if not finished.cancelled() else None)
        return initial
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/agent/runs/{run_id}", response_model=HarnessResponse)
async def get_agent_run(
    run_id: str, app: Runtime, user: dict = Depends(require_user)
) -> HarnessResponse:
    try:
        record = app.services.repository.get_agent_run_record(run_id)
        if record is None:
            raise KeyError("Agent任务不存在")
        if record.user_id != int(user["id"]):
            raise PermissionError("无权读取其他医生的Agent任务")
        node_labels = {
            "intake": "正在读取请求…", "route_request": "正在选择执行路径…",
            "general_knowledge_retrieval": "正在选择知识源并检索…",
            "medical_knowledge_retrieval": "正在检索本地医学证据…",
            "public_web_search": "正在检索公开网络资料…",
            "clinical_chat": "正在读取病例并规划检索…",
            "multimodal_analysis": "正在执行本地联合分析…",
            "confirm_case_creation": "病例资料已准备，等待医生确认。",
            "confirm_report_generation": "报告生成任务已准备，等待医生确认。",
            "confirm_agent_action": "Agent 已生成操作预览，等待医生确认。",
            "completed": "任务已完成。", "failed": "任务执行失败。",
        }
        current = app.services.repository.get_agent_run(run_id) or {}
        message = node_labels.get(current.get("current_node"), "正在处理任务…")
        last_result = current.get("state", {}).get("last_result", {})
        if current.get("status") == "COMPLETED" and last_result.get("kind") == "chat":
            message = last_result.get("chat", {}).get("response") or message
        elif (
            current.get("status") == "WAITING_CONFIRM"
            and last_result.get("kind") == "chat"
        ):
            message = last_result.get("chat", {}).get("response") or message
        elif current.get("status") == "WAITING_CONFIRM":
            message = "资料已准备完成，请核对后确认执行。"
        return app.harness.response(run_id, message)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/agent/runs/{run_id}/events")
async def get_agent_events(
    run_id: str, app: Runtime, user: dict = Depends(require_user)
) -> list[dict[str, Any]]:
    record = app.services.repository.get_agent_run_record(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Agent任务不存在")
    if record.user_id != int(user["id"]):
        raise HTTPException(status_code=403, detail="无权读取其他医生的Agent任务")
    return app.services.repository.list_agent_events(run_id)


@router.post("/agent/runs/{run_id}/confirm", response_model=HarnessResponse)
async def confirm_agent_action(
    run_id: str, payload: AgentActionRequest, app: Runtime,
    user: dict = Depends(require_user),
) -> HarnessResponse:
    try:
        return await app.harness.confirm(run_id, int(user["id"]), payload.action_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/agent/runs/{run_id}/reject", response_model=HarnessResponse)
async def reject_agent_action(
    run_id: str, app: Runtime, user: dict = Depends(require_user)
) -> HarnessResponse:
    try:
        return app.harness.reject(run_id, int(user["id"]))
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/agent/runs/{run_id}/pause", response_model=HarnessResponse)
async def pause_agent_run(
    run_id: str, app: Runtime, user: dict = Depends(require_user)
) -> HarnessResponse:
    try:
        return app.harness.pause(run_id, int(user["id"]))
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/agent/runs/{run_id}/resume", response_model=HarnessResponse)
async def resume_agent_run(
    run_id: str, payload: AgentResumeRequest, app: Runtime,
    user: dict = Depends(require_user),
) -> HarnessResponse:
    try:
        return await app.harness.resume(
            run_id, int(user["id"]), payload.message, payload.model, payload.detail_level
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/chat/models")
async def chat_models() -> list[dict[str, Any]]:
    return chat_model_catalog()


@router.get("/agent/documents")
async def list_agent_documents(
    app: Runtime, user: dict = Depends(require_user),
    conversation_id: str = Query(default="", max_length=100),
    folder_id: str = Query(default="", max_length=64),
) -> list[dict[str, Any]]:
    items = app.services.repository.list_agent_documents(
        user_id=int(user["id"]), conversation_id=conversation_id or None
    )
    if folder_id:
        ids = set(app.services.repository.folder_resource_ids(int(user["id"]), folder_id, "document"))
        items = [item for item in items if item["document_id"] in ids]
    locations = app.services.repository.workspace_locations(int(user["id"]), "document")
    return [{**item, "folder_id": locations.get(item["document_id"])} for item in items]


@router.post("/agent/documents/upload")
async def upload_agent_document(
    app: Runtime, user: dict = Depends(require_user),
    file: UploadFile = File(...), conversation_id: str = Form(default="workspace"),
    case_id: str = Form(default=""), folder_id: str = Form(default=""),
) -> dict[str, Any]:
    allowed = {".md", ".txt", ".json", ".csv"}
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in allowed:
        raise HTTPException(status_code=422, detail="仅支持 Markdown、TXT、JSON 和 CSV 文本文档")
    raw = await file.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="文档不能超过 2 MB")
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="文档必须使用 UTF-8 编码") from exc
    item = app.services.repository.create_agent_document(
        user_id=int(user["id"]), conversation_id=conversation_id or "workspace",
        case_id=case_id or None, title=os.path.splitext(file.filename or "未命名文档")[0],
        filename=file.filename or "document.md", content=content,
    )
    if folder_id:
        app.services.repository.move_workspace_item(int(user["id"]), "document", item["document_id"], folder_id)
    return item


@router.put("/agent/documents/{document_id}")
async def update_agent_document(
    document_id: str, request: AgentDocumentUpdateRequest,
    app: Runtime, user: dict = Depends(require_user),
) -> dict[str, Any]:
    try:
        return app.services.repository.update_agent_document(
            document_id, int(user["id"]), content=request.content,
            title=request.title.strip() or None,
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.delete("/agent/documents/{document_id}")
async def delete_agent_document(
    document_id: str, app: Runtime, user: dict = Depends(require_user),
) -> dict[str, Any]:
    if not app.services.repository.delete_agent_document(document_id, int(user["id"])):
        raise HTTPException(status_code=404, detail="Agent 文档不存在")
    return {"status": "deleted", "document_id": document_id, "case_and_report_retained": True}


@router.get("/agent/documents/{document_id}/download")
async def download_agent_document(
    document_id: str, app: Runtime, user: dict = Depends(require_user)
) -> Response:
    item = app.services.repository.get_agent_document(document_id, int(user["id"]))
    if item is None:
        raise HTTPException(status_code=404, detail="Agent 文档不存在")
    encoded = quote(item["filename"])
    return Response(
        item["content"].encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded}"},
    )


@router.get("/cases/{case_id}/chat-history")
async def chat_history(case_id: str, app: Runtime) -> list[dict[str, Any]]:
    if app.services.repository.get_case(case_id) is None:
        raise HTTPException(status_code=404, detail="病例不存在")
    return app.services.repository.list_chat_messages(case_id)


@router.delete("/cases/{case_id}/chat-history")
async def clear_chat_history(case_id: str, app: Runtime) -> dict[str, int]:
    if app.services.repository.get_case(case_id) is None:
        raise HTTPException(status_code=404, detail="病例不存在")
    return {"deleted": app.services.repository.clear_chat_messages(case_id)}


@router.post("/cases/{case_id}/reports")
async def generate_report(
    case_id: str,
    request: ReportGenerateRequest,
    app: Runtime,
    user: dict = Depends(require_user),
) -> dict[str, Any]:
    try:
        patient = app.services.repository.get_case_patient(case_id)
        if patient is None:
            raise KeyError("该病例尚未关联患者")
        clinical_payload = {
            "visual": app.services.repository.get_visual(case_id) or {},
            "triage": app.services.repository.get_triage(case_id) or {},
            "care_pathway": app.services.repository.get_care_pathway(case_id) or {},
            "clinical_profile": patient.get("clinical_profile", {}),
        }
        generated = ReportService(app.guard).generate(
            case_id=case_id,
            patient=patient,
            clinical_payload=clinical_payload,
            model_id=request.model,
        )
        record = app.services.repository.create_report(
            case_id=case_id,
            patient_id=patient["patient_id"],
            content=generated.content,
            structured=generated.structured,
            provider=generated.provider,
            model=generated.model,
            created_by=int(user["id"]),
        )
        report = app.services.repository.get_report(record.report_id)
        if report is None:
            raise RuntimeError("报告保存失败")
        return report
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/reports")
async def list_reports(
    app: Runtime, user: dict = Depends(require_user),
    query: str = Query(default="", max_length=120),
    folder_id: str = Query(default="", max_length=64),
) -> list[dict[str, Any]]:
    items = app.services.repository.list_reports(query=query, user_id=int(user["id"]))
    if folder_id:
        ids = set(app.services.repository.folder_resource_ids(int(user["id"]), folder_id, "report"))
        items = [item for item in items if item["report_id"] in ids]
    locations = app.services.repository.workspace_locations(int(user["id"]), "report")
    return [{**item, "folder_id": locations.get(item["report_id"])} for item in items]


@router.get("/workspace/folders")
async def list_workspace_folders(app: Runtime, user: dict = Depends(require_user)) -> list[dict[str, Any]]:
    return app.services.repository.list_folders(int(user["id"]))


@router.post("/workspace/folders")
async def create_workspace_folder(payload: FolderCreateRequest, app: Runtime, user: dict = Depends(require_user)) -> dict[str, Any]:
    try:
        return app.services.repository.create_folder(int(user["id"]), payload.name.strip(), payload.parent_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.delete("/workspace/folders/{folder_id}")
async def delete_workspace_folder(folder_id: str, app: Runtime, user: dict = Depends(require_user)) -> dict[str, Any]:
    if not app.services.repository.delete_folder(int(user["id"]), folder_id):
        raise HTTPException(status_code=404, detail="文件夹不存在")
    return {"status": "deleted", "folder_id": folder_id, "resources_retained": True}


@router.put("/workspace/items")
async def move_workspace_item(payload: WorkspaceMoveRequest, app: Runtime, user: dict = Depends(require_user)) -> dict[str, Any]:
    try:
        return app.services.repository.move_workspace_item(
            int(user["id"]), payload.resource_type, payload.resource_id, payload.folder_id
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/reports/{report_id}")
async def get_report(report_id: str, app: Runtime, user: dict = Depends(require_user)) -> dict[str, Any]:
    report = app.services.repository.get_report_for_user(report_id, int(user["id"]))
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    return report


@router.put("/reports/{report_id}")
async def update_report(
    report_id: str,
    request: ReportUpdateRequest,
    app: Runtime,
    user: dict = Depends(require_user),
) -> dict[str, Any]:
    try:
        return app.services.repository.update_report(
            report_id, request.content, int(user["id"]), request.status
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.delete("/reports/{report_id}")
async def delete_report(report_id: str, app: Runtime, user: dict = Depends(require_user)) -> dict[str, Any]:
    if not app.services.repository.delete_report(report_id, int(user["id"])):
        raise HTTPException(status_code=404, detail="报告不存在")
    return {"status": "deleted", "report_id": report_id, "case_and_image_retained": True}


@router.get("/reports/{report_id}/image")
async def get_report_image(report_id: str, app: Runtime, user: dict = Depends(require_user)) -> Response:
    report = app.services.repository.get_report_for_user(report_id, int(user["id"]))
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    path = app.services.cases.get_local_image_path(report["case_id"])
    media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "application/octet-stream")
    return Response(path.read_bytes(), media_type=media_type)


@router.get("/reports/{report_id}/download")
async def download_report(report_id: str, app: Runtime, user: dict = Depends(require_user)) -> PlainTextResponse:
    report = app.services.repository.get_report_for_user(report_id, int(user["id"]))
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    return PlainTextResponse(
        report["content"],
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={report['report_id']}.md"},
    )


@router.get("/audit")
async def audit(app: Runtime, case_id: str | None = Query(default=None)) -> list[dict[str, Any]]:
    return app.services.repository.list_audit_events(case_id)
