from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Type
from uuid import uuid4

from pydantic import BaseModel
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import sessionmaker

from backend.app.database.models import (
    AgentDocumentRecord,
    AgentEventRecord,
    AgentRunRecord,
    AuthSessionRecord,
    AuditEventRecord,
    CarePathwayRecord,
    CaseRecord,
    CasePatientRecord,
    ChatMessageRecord,
    PatientRecord,
    ReportRecord,
    ReportRevisionRecord,
    ToolCallRecord,
    TriageResultRecord,
    UserRecord,
    VisualResultRecord,
    WorkspaceFolderRecord,
    WorkspaceItemRecord,
)
from backend.app.database.repository import init_database


class CaseRepository:
    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine or init_database()
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    @staticmethod
    def _json(value: Any) -> str:
        if isinstance(value, BaseModel):
            return value.model_dump_json()
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _with_case_image(content: str, filename: str, case_id: str | None) -> str:
        if not case_id or not filename.lower().endswith(".md"):
            return content
        image_ref = f"/v1/cases/{case_id}/image"
        if image_ref in content:
            return content
        return f"## 患者病灶图像\n\n![患者病灶图片]({image_ref})\n\n{content}"

    def create_case(
        self,
        case_id: str,
        local_image_path: str,
        image_sha256: str,
        retention_mode: str,
        mode: str,
    ) -> None:
        with self.sessions.begin() as db:
            db.add(
                CaseRecord(
                    case_id=case_id,
                    local_image_path=local_image_path,
                    image_sha256=image_sha256,
                    retention_mode=retention_mode,
                    mode=mode,
                )
            )

    @staticmethod
    def user_public(user: UserRecord) -> dict[str, Any]:
        return {
            "id": user.id,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role,
        }

    def get_user_by_username(self, username: str) -> UserRecord | None:
        with self.sessions() as db:
            return db.scalar(select(UserRecord).where(UserRecord.username == username))

    def create_user(self, username: str, password_hash: str, full_name: str) -> UserRecord:
        with self.sessions.begin() as db:
            user = UserRecord(
                username=username,
                password_hash=password_hash,
                full_name=full_name,
                role="clinician",
            )
            db.add(user)
            db.flush()
            return user

    def create_auth_session(self, token_hash: str, user_id: int, expires_at: datetime) -> None:
        with self.sessions.begin() as db:
            db.add(AuthSessionRecord(token_hash=token_hash, user_id=user_id, expires_at=expires_at))

    def get_session_user(self, token_hash: str, now: datetime) -> UserRecord | None:
        with self.sessions() as db:
            session = db.get(AuthSessionRecord, token_hash)
            if session is None or session.expires_at <= now:
                return None
            user = db.get(UserRecord, session.user_id)
            return user if user is not None and user.active else None

    def delete_auth_session(self, token_hash: str) -> None:
        with self.sessions.begin() as db:
            db.execute(delete(AuthSessionRecord).where(AuthSessionRecord.token_hash == token_hash))

    def create_patient(
        self,
        *,
        name: str,
        sex: str,
        age: int,
        outpatient_no: str,
        phone: str | None,
        medical_history: str,
        allergy_history: str,
        created_by: int,
    ) -> PatientRecord:
        patient_id = f"PAT_{uuid4().hex[:12]}"
        with self.sessions.begin() as db:
            existing = db.scalar(select(PatientRecord).where(PatientRecord.outpatient_no == outpatient_no))
            if existing is not None:
                existing.name = name
                existing.sex = sex
                existing.age = age
                existing.phone = phone
                existing.medical_history = medical_history
                existing.allergy_history = allergy_history
                return existing
            patient = PatientRecord(
                patient_id=patient_id,
                name=name,
                sex=sex,
                age=age,
                outpatient_no=outpatient_no,
                phone=phone,
                medical_history=medical_history,
                allergy_history=allergy_history,
                created_by=created_by,
            )
            db.add(patient)
            db.flush()
            return patient

    def get_patient(self, patient_id: str) -> PatientRecord | None:
        with self.sessions() as db:
            return db.get(PatientRecord, patient_id)

    def list_patients(self, query: str = "", limit: int = 100) -> list[dict[str, Any]]:
        with self.sessions() as db:
            statement = select(PatientRecord).order_by(PatientRecord.updated_at.desc()).limit(limit)
            if query.strip():
                pattern = f"%{query.strip()}%"
                statement = statement.where(
                    PatientRecord.name.like(pattern) | PatientRecord.outpatient_no.like(pattern)
                )
            return [self.patient_public(item) for item in db.scalars(statement).all()]

    @staticmethod
    def patient_public(item: PatientRecord) -> dict[str, Any]:
        return {
            "patient_id": item.patient_id,
            "name": item.name,
            "sex": item.sex,
            "age": item.age,
            "outpatient_no": item.outpatient_no,
            "phone": item.phone or "",
            "medical_history": item.medical_history,
            "allergy_history": item.allergy_history,
            "created_at": item.created_at.isoformat(),
            "updated_at": item.updated_at.isoformat(),
        }

    def link_case_patient(
        self,
        case_id: str,
        patient_id: str,
        lesion_site: str,
        chief_complaint: str,
        clinical_profile: dict[str, Any],
    ) -> None:
        with self.sessions.begin() as db:
            db.add(CasePatientRecord(
                case_id=case_id,
                patient_id=patient_id,
                lesion_site=lesion_site,
                chief_complaint=chief_complaint,
                clinical_profile_json=self._json(clinical_profile),
            ))

    def get_case_patient(self, case_id: str) -> dict[str, Any] | None:
        with self.sessions() as db:
            link = db.get(CasePatientRecord, case_id)
            if link is None:
                return None
            patient = db.get(PatientRecord, link.patient_id)
            if patient is None:
                return None
            result = self.patient_public(patient)
            result.update({
                "case_id": case_id,
                "lesion_site": link.lesion_site,
                "chief_complaint": link.chief_complaint,
                "clinical_profile": json.loads(link.clinical_profile_json or "{}"),
            })
            return result

    def update_case_patient(
        self,
        case_id: str,
        *,
        lesion_site: str | None = None,
        chief_complaint: str | None = None,
        clinical_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply an allow-listed patch without replacing patient identity."""
        with self.sessions.begin() as db:
            link = db.get(CasePatientRecord, case_id)
            if link is None:
                raise KeyError(f"case patient not found: {case_id}")
            if lesion_site is not None:
                link.lesion_site = lesion_site
            if chief_complaint is not None:
                link.chief_complaint = chief_complaint
            if clinical_profile is not None:
                link.clinical_profile_json = self._json(clinical_profile)
        updated = self.get_case_patient(case_id)
        if updated is None:
            raise KeyError(f"case patient not found: {case_id}")
        return updated

    def create_report(
        self,
        *, case_id: str, patient_id: str, content: str, structured: dict[str, Any],
        provider: str, model: str, created_by: int,
    ) -> ReportRecord:
        report_id = f"RPT_{uuid4().hex[:12]}"
        filename = f"{patient_id}-{report_id[-6:]}-辅助诊断报告.md"
        with self.sessions.begin() as db:
            report = ReportRecord(
                report_id=report_id,
                case_id=case_id,
                patient_id=patient_id,
                filename=filename,
                content=content,
                structured_json=self._json(structured),
                provider=provider,
                model=model,
                created_by=created_by,
            )
            db.add(report)
            db.add(ReportRevisionRecord(
                report_id=report_id, content=content, editor_user_id=created_by
            ))
            db.flush()
            return report

    def list_reports(
        self, query: str = "", limit: int = 200, user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        with self.sessions() as db:
            statement = select(ReportRecord).order_by(ReportRecord.updated_at.desc()).limit(limit)
            if user_id is not None:
                statement = statement.where(ReportRecord.created_by == user_id)
            records = db.scalars(statement).all()
            output = []
            for item in records:
                patient = db.get(PatientRecord, item.patient_id)
                if query.strip() and patient is not None:
                    needle = query.strip().lower()
                    if needle not in patient.name.lower() and needle not in patient.outpatient_no.lower() and needle not in item.filename.lower():
                        continue
                output.append(self._report_public(item, patient))
            return output

    @staticmethod
    def _report_public(item: ReportRecord, patient: PatientRecord | None = None) -> dict[str, Any]:
        return {
            "report_id": item.report_id,
            "case_id": item.case_id,
            "patient_id": item.patient_id,
            "patient_name": patient.name if patient else "",
            "outpatient_no": patient.outpatient_no if patient else "",
            "filename": item.filename,
            "content": CaseRepository._with_case_image(item.content, item.filename, item.case_id),
            "structured": json.loads(item.structured_json or "{}"),
            "provider": item.provider,
            "model": item.model,
            "status": item.status,
            "created_at": item.created_at.isoformat(),
            "updated_at": item.updated_at.isoformat(),
        }

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        with self.sessions() as db:
            item = db.get(ReportRecord, report_id)
            if item is None:
                return None
            return self._report_public(item, db.get(PatientRecord, item.patient_id))

    def get_report_for_user(self, report_id: str, user_id: int) -> dict[str, Any] | None:
        """Return a report only when it belongs to the authenticated clinician."""
        with self.sessions() as db:
            item = db.get(ReportRecord, report_id)
            if item is None or item.created_by != user_id:
                return None
            return self._report_public(item, db.get(PatientRecord, item.patient_id))

    def patient_identifiers_for_user(self, user_id: int) -> list[dict[str, str]]:
        """Identifiers used locally to pseudonymize archive queries before egress."""
        with self.sessions() as db:
            patients = db.scalars(select(PatientRecord).where(
                PatientRecord.created_by == user_id
            ).order_by(PatientRecord.created_at)).all()
            return [{
                "patient_id": item.patient_id,
                "name": item.name,
                "outpatient_no": item.outpatient_no,
                "phone": item.phone or "",
            } for item in patients]

    def search_clinical_archive(
        self, user_id: int, query: str = "", limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search clinician-owned reports and documents inside the trusted database."""
        needle = query.strip().lower()
        output: list[dict[str, Any]] = []
        with self.sessions() as db:
            reports = db.scalars(select(ReportRecord).where(
                ReportRecord.created_by == user_id
            ).order_by(ReportRecord.updated_at.desc()).limit(200)).all()
            for item in reports:
                patient = db.get(PatientRecord, item.patient_id)
                haystack = " ".join([
                    item.report_id, item.filename, item.content,
                    patient.name if patient else "",
                    patient.outpatient_no if patient else "",
                ]).lower()
                if needle and needle not in haystack:
                    continue
                output.append({
                    "resource_type": "report", "resource_id": item.report_id,
                    "case_id": item.case_id, "title": item.filename,
                    "content": item.content, "updated_at": item.updated_at.isoformat(),
                })
            documents = db.scalars(select(AgentDocumentRecord).where(
                AgentDocumentRecord.user_id == user_id
            ).order_by(AgentDocumentRecord.updated_at.desc()).limit(200)).all()
            for item in documents:
                haystack = " ".join([
                    item.document_id, item.title, item.filename, item.content,
                ]).lower()
                if needle and needle not in haystack:
                    continue
                output.append({
                    "resource_type": "document", "resource_id": item.document_id,
                    "case_id": item.case_id, "title": item.title,
                    "filename": item.filename, "content": item.content,
                    "updated_at": item.updated_at.isoformat(),
                })
        output.sort(key=lambda item: item["updated_at"], reverse=True)
        return output[:limit]

    def read_clinical_archive_resource(
        self, user_id: int, resource_type: str, resource_id: str,
    ) -> dict[str, Any] | None:
        if resource_type == "report":
            return self.get_report_for_user(resource_id, user_id)
        if resource_type == "document":
            return self.get_agent_document(resource_id, user_id)
        return None

    def update_report(self, report_id: str, content: str, editor_user_id: int, status: str = "final") -> dict[str, Any]:
        with self.sessions.begin() as db:
            item = db.get(ReportRecord, report_id)
            if item is None or item.created_by != editor_user_id:
                raise KeyError(f"report not found: {report_id}")
            item.content = content
            item.status = status
            item.updated_at = datetime.utcnow()
            db.add(ReportRevisionRecord(report_id=report_id, content=content, editor_user_id=editor_user_id))
            db.flush()
            return self._report_public(item, db.get(PatientRecord, item.patient_id))

    @staticmethod
    def _agent_document_public(item: AgentDocumentRecord) -> dict[str, Any]:
        return {
            "document_id": item.document_id,
            "conversation_id": item.conversation_id,
            "case_id": item.case_id,
            "title": item.title,
            "filename": item.filename,
            "content": CaseRepository._with_case_image(item.content, item.filename, item.case_id),
            "revision": item.revision,
            "created_at": item.created_at.isoformat(),
            "updated_at": item.updated_at.isoformat(),
        }

    def create_agent_document(
        self, *, user_id: int, conversation_id: str, case_id: str | None,
        title: str, filename: str, content: str,
    ) -> dict[str, Any]:
        document_id = f"DOC_{uuid4().hex[:12]}"
        with self.sessions.begin() as db:
            content = self._with_case_image(content, filename, case_id)
            item = AgentDocumentRecord(
                document_id=document_id, user_id=user_id,
                conversation_id=conversation_id, case_id=case_id,
                title=title, filename=filename, content=content,
            )
            db.add(item)
            db.flush()
            return self._agent_document_public(item)

    def list_agent_documents(
        self, *, user_id: int, conversation_id: str | None = None, limit: int = 200,
    ) -> list[dict[str, Any]]:
        with self.sessions() as db:
            statement = (
                select(AgentDocumentRecord)
                .where(AgentDocumentRecord.user_id == user_id)
                .order_by(AgentDocumentRecord.updated_at.desc())
                .limit(limit)
            )
            if conversation_id:
                statement = statement.where(
                    AgentDocumentRecord.conversation_id == conversation_id
                )
            return [self._agent_document_public(item) for item in db.scalars(statement).all()]

    def get_agent_document(self, document_id: str, user_id: int) -> dict[str, Any] | None:
        with self.sessions() as db:
            item = db.get(AgentDocumentRecord, document_id)
            if item is None or item.user_id != user_id:
                return None
            return self._agent_document_public(item)

    def update_agent_document(
        self, document_id: str, user_id: int, *, content: str,
        title: str | None = None,
    ) -> dict[str, Any]:
        with self.sessions.begin() as db:
            item = db.get(AgentDocumentRecord, document_id)
            if item is None or item.user_id != user_id:
                raise KeyError("Agent 文档不存在")
            item.content = self._with_case_image(content, item.filename, item.case_id)
            if title:
                item.title = title
            item.revision += 1
            item.updated_at = datetime.utcnow()
            db.flush()
            return self._agent_document_public(item)

    def copy_agent_document(
        self, document_id: str, user_id: int, conversation_id: str | None = None,
    ) -> dict[str, Any]:
        source = self.get_agent_document(document_id, user_id)
        if source is None:
            raise KeyError("Agent 文档不存在")
        filename = source["filename"]
        if "." in filename:
            stem, suffix = filename.rsplit(".", 1)
            filename = f"{stem}_副本.{suffix}"
        else:
            filename = f"{filename}_副本.md"
        return self.create_agent_document(
            user_id=user_id,
            conversation_id=conversation_id or source["conversation_id"],
            case_id=source.get("case_id"),
            title=f"{source['title']}（副本）",
            filename=filename,
            content=source["content"],
        )

    def copy_report(self, report_id: str, user_id: int) -> dict[str, Any]:
        source = self.get_report_for_user(report_id, user_id)
        if source is None:
            raise KeyError("报告不存在")
        record = self.create_report(
            case_id=source["case_id"], patient_id=source["patient_id"],
            content=source["content"], structured=source["structured"],
            provider=source["provider"], model=source["model"], created_by=user_id,
        )
        copied = self.get_report_for_user(record.report_id, user_id)
        if copied is None:
            raise RuntimeError("报告复制失败")
        return copied

    def delete_agent_document(self, document_id: str, user_id: int) -> bool:
        """Delete one clinician-owned document and its folder placement."""
        with self.sessions.begin() as db:
            item = db.get(AgentDocumentRecord, document_id)
            if item is None or item.user_id != user_id:
                return False
            db.execute(delete(WorkspaceItemRecord).where(
                WorkspaceItemRecord.user_id == user_id,
                WorkspaceItemRecord.resource_type == "document",
                WorkspaceItemRecord.resource_id == document_id,
            ))
            db.delete(item)
            return True

    def delete_report(self, report_id: str, user_id: int | None = None) -> bool:
        with self.sessions.begin() as db:
            report = db.get(ReportRecord, report_id)
            if report is None or (user_id is not None and report.created_by != user_id):
                return False
            db.execute(delete(WorkspaceItemRecord).where(
                WorkspaceItemRecord.resource_type == "report",
                WorkspaceItemRecord.resource_id == report_id,
                *([WorkspaceItemRecord.user_id == user_id] if user_id is not None else []),
            ))
            db.execute(delete(ReportRevisionRecord).where(ReportRevisionRecord.report_id == report_id))
            result = db.execute(delete(ReportRecord).where(ReportRecord.report_id == report_id))
            return bool(result.rowcount)

    def get_case(self, case_id: str) -> CaseRecord | None:
        with self.sessions() as db:
            return db.get(CaseRecord, case_id)

    def list_cases(
        self, query: str = "", limit: int = 200, *, sex: str = "",
        lesion_site: str = "", min_age: int | None = None,
        max_age: int | None = None, diagnosis: str = "", triage_level: str = "",
        folder_id: str = "", user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return case workspace summaries without exposing local file paths."""
        needle = query.strip().lower()
        with self.sessions() as db:
            records = db.scalars(
                select(CaseRecord).order_by(CaseRecord.created_at.desc()).limit(limit)
            ).all()
            output: list[dict[str, Any]] = []
            folder_items: set[str] | None = None
            if folder_id and user_id is not None:
                folder_items = set(db.scalars(select(WorkspaceItemRecord.resource_id).where(
                    WorkspaceItemRecord.user_id == user_id,
                    WorkspaceItemRecord.folder_id == folder_id,
                    WorkspaceItemRecord.resource_type == "case",
                )).all())
            for item in records:
                if folder_items is not None and item.case_id not in folder_items:
                    continue
                link = db.get(CasePatientRecord, item.case_id)
                patient = db.get(PatientRecord, link.patient_id) if link else None
                if user_id is not None and (patient is None or patient.created_by != user_id):
                    continue
                searchable = " ".join(
                    [
                        item.case_id,
                        patient.name if patient else "",
                        patient.outpatient_no if patient else "",
                        link.lesion_site if link else "",
                    ]
                ).lower()
                if needle and needle not in searchable:
                    continue
                if sex and (patient is None or patient.sex != sex):
                    continue
                if min_age is not None and (patient is None or patient.age < min_age):
                    continue
                if max_age is not None and (patient is None or patient.age > max_age):
                    continue
                if lesion_site and (link is None or lesion_site.lower() not in link.lesion_site.lower()):
                    continue
                visual = db.scalar(
                    select(VisualResultRecord).where(
                        VisualResultRecord.case_id == item.case_id
                    )
                )
                triage = db.scalar(
                    select(TriageResultRecord).where(
                        TriageResultRecord.case_id == item.case_id
                    )
                )
                visual_data = json.loads(visual.result_json) if visual else {}
                triage_data = json.loads(triage.result_json) if triage else {}
                predictions = visual_data.get("top_predictions", [])
                top_label = predictions[0].get("label", "") if predictions else ""
                level = triage_data.get("level", "")
                if diagnosis and diagnosis.lower() not in top_label.lower():
                    continue
                if triage_level and level != triage_level:
                    continue
                output.append(
                    {
                        "case_id": item.case_id,
                        "patient_name": patient.name if patient else "",
                        "outpatient_no": patient.outpatient_no if patient else "",
                        "lesion_site": link.lesion_site if link else "",
                        "chief_complaint": link.chief_complaint if link else "",
                        "has_analysis": visual is not None and triage is not None,
                        "top_diagnosis": top_label,
                        "triage_level": level,
                        "created_at": item.created_at.isoformat(),
                    }
                )
            return output

    def owned_case_ids(self, user_id: int) -> list[str]:
        """Return every case owned by one clinician through its patient record."""
        with self.sessions() as db:
            rows = db.execute(
                select(CaseRecord.case_id)
                .join(CasePatientRecord, CasePatientRecord.case_id == CaseRecord.case_id)
                .join(PatientRecord, PatientRecord.patient_id == CasePatientRecord.patient_id)
                .where(PatientRecord.created_by == user_id)
                .order_by(CaseRecord.created_at.desc())
            ).all()
            return [str(row[0]) for row in rows]

    def case_belongs_to_user(self, case_id: str, user_id: int) -> bool:
        with self.sessions() as db:
            link = db.get(CasePatientRecord, case_id)
            if link is None:
                return False
            patient = db.get(PatientRecord, link.patient_id)
            return patient is not None and patient.created_by == user_id

    def update_case_record(self, case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.sessions.begin() as db:
            link = db.get(CasePatientRecord, case_id)
            if link is None:
                raise KeyError("病例尚未关联患者")
            patient = db.get(PatientRecord, link.patient_id)
            if patient is None:
                raise KeyError("患者不存在")
            for field in ("name", "sex", "age", "phone", "medical_history", "allergy_history"):
                if field in payload and payload[field] is not None:
                    setattr(patient, field, payload[field])
            if payload.get("outpatient_no"):
                duplicate = db.scalar(select(PatientRecord).where(
                    PatientRecord.outpatient_no == payload["outpatient_no"],
                    PatientRecord.patient_id != patient.patient_id,
                ))
                if duplicate is not None:
                    raise ValueError("门诊号已被其他患者使用")
                patient.outpatient_no = payload["outpatient_no"]
            if payload.get("lesion_site") is not None:
                link.lesion_site = payload["lesion_site"]
            if payload.get("chief_complaint") is not None:
                link.chief_complaint = payload["chief_complaint"]
            if payload.get("clinical_profile") is not None:
                link.clinical_profile_json = self._json(payload["clinical_profile"])
            patient.updated_at = datetime.utcnow()
        updated = self.get_case_patient(case_id)
        if updated is None:
            raise KeyError("病例更新失败")
        return updated

    def delete_case_record(self, case_id: str, user_id: int | None = None) -> str:
        """Delete one case and its dependent records; return its image path."""
        with self.sessions.begin() as db:
            case = db.get(CaseRecord, case_id)
            if case is None:
                raise KeyError("病例不存在")
            link = db.get(CasePatientRecord, case_id)
            patient = db.get(PatientRecord, link.patient_id) if link else None
            if user_id is not None and (patient is None or patient.created_by != user_id):
                raise PermissionError("无权删除其他医生的病例")
            image_path = case.local_image_path
            report_ids = list(db.scalars(select(ReportRecord.report_id).where(ReportRecord.case_id == case_id)).all())
            document_ids = list(db.scalars(select(AgentDocumentRecord.document_id).where(AgentDocumentRecord.case_id == case_id)).all())
            if report_ids:
                db.execute(delete(ReportRevisionRecord).where(ReportRevisionRecord.report_id.in_(report_ids)))
            run_ids = list(db.scalars(select(AgentRunRecord.run_id).where(AgentRunRecord.case_id == case_id)).all())
            if run_ids:
                db.execute(delete(AgentEventRecord).where(AgentEventRecord.run_id.in_(run_ids)))
            for model in (ReportRecord, ChatMessageRecord, ToolCallRecord,
                          VisualResultRecord, TriageResultRecord, CarePathwayRecord,
                          CasePatientRecord, AgentDocumentRecord):
                db.execute(delete(model).where(model.case_id == case_id))
            db.execute(delete(AgentRunRecord).where(AgentRunRecord.case_id == case_id))
            db.execute(delete(WorkspaceItemRecord).where(
                (WorkspaceItemRecord.resource_type == "case") &
                (WorkspaceItemRecord.resource_id == case_id)
            ))
            if report_ids:
                db.execute(delete(WorkspaceItemRecord).where(
                    WorkspaceItemRecord.resource_type == "report",
                    WorkspaceItemRecord.resource_id.in_(report_ids),
                ))
            if document_ids:
                db.execute(delete(WorkspaceItemRecord).where(
                    WorkspaceItemRecord.resource_type == "document",
                    WorkspaceItemRecord.resource_id.in_(document_ids),
                ))
            db.delete(case)
            return image_path

    def create_folder(self, user_id: int, name: str, parent_id: str | None = None) -> dict[str, Any]:
        folder_id = f"FLD_{uuid4().hex[:12]}"
        with self.sessions.begin() as db:
            if parent_id:
                parent = db.get(WorkspaceFolderRecord, parent_id)
                if parent is None or parent.user_id != user_id:
                    raise KeyError("上级文件夹不存在")
            item = WorkspaceFolderRecord(folder_id=folder_id, user_id=user_id, name=name, parent_id=parent_id)
            db.add(item); db.flush()
            return {"folder_id": item.folder_id, "name": item.name, "parent_id": item.parent_id, "created_at": item.created_at.isoformat()}

    def list_folders(self, user_id: int) -> list[dict[str, Any]]:
        with self.sessions() as db:
            folders = db.scalars(select(WorkspaceFolderRecord).where(WorkspaceFolderRecord.user_id == user_id).order_by(WorkspaceFolderRecord.name)).all()
            return [{"folder_id": item.folder_id, "name": item.name, "parent_id": item.parent_id, "created_at": item.created_at.isoformat()} for item in folders]

    def delete_folder(self, user_id: int, folder_id: str) -> bool:
        with self.sessions.begin() as db:
            folder = db.get(WorkspaceFolderRecord, folder_id)
            if folder is None or folder.user_id != user_id:
                return False
            db.execute(delete(WorkspaceItemRecord).where(WorkspaceItemRecord.folder_id == folder_id, WorkspaceItemRecord.user_id == user_id))
            db.execute(delete(WorkspaceFolderRecord).where(WorkspaceFolderRecord.parent_id == folder_id, WorkspaceFolderRecord.user_id == user_id))
            db.delete(folder); return True

    def move_workspace_item(self, user_id: int, resource_type: str, resource_id: str, folder_id: str | None) -> dict[str, Any]:
        with self.sessions.begin() as db:
            db.execute(delete(WorkspaceItemRecord).where(
                WorkspaceItemRecord.user_id == user_id,
                WorkspaceItemRecord.resource_type == resource_type,
                WorkspaceItemRecord.resource_id == resource_id,
            ))
            if folder_id:
                folder = db.get(WorkspaceFolderRecord, folder_id)
                if folder is None or folder.user_id != user_id:
                    raise KeyError("文件夹不存在")
                db.add(WorkspaceItemRecord(user_id=user_id, folder_id=folder_id, resource_type=resource_type, resource_id=resource_id))
        return {"resource_type": resource_type, "resource_id": resource_id, "folder_id": folder_id}

    def folder_resource_ids(self, user_id: int, folder_id: str, resource_type: str) -> list[str]:
        with self.sessions() as db:
            return list(db.scalars(select(WorkspaceItemRecord.resource_id).where(
                WorkspaceItemRecord.user_id == user_id,
                WorkspaceItemRecord.folder_id == folder_id,
                WorkspaceItemRecord.resource_type == resource_type,
            )).all())

    def workspace_locations(self, user_id: int, resource_type: str) -> dict[str, str]:
        """Return resource id to folder id mappings for workspace presentation."""
        with self.sessions() as db:
            records = db.scalars(select(WorkspaceItemRecord).where(
                WorkspaceItemRecord.user_id == user_id,
                WorkspaceItemRecord.resource_type == resource_type,
            )).all()
            return {item.resource_id: item.folder_id for item in records}

    def _upsert_result(self, model: Type, case_id: str, result: BaseModel) -> None:
        with self.sessions.begin() as db:
            record = db.scalar(select(model).where(model.case_id == case_id))
            if record is None:
                record = model(case_id=case_id, result_json=result.model_dump_json())
                db.add(record)
            else:
                record.result_json = result.model_dump_json()

    def _get_result(self, model: Type, case_id: str) -> dict[str, Any] | None:
        with self.sessions() as db:
            record = db.scalar(select(model).where(model.case_id == case_id))
            return None if record is None else json.loads(record.result_json)

    def save_visual(self, case_id: str, result: BaseModel) -> None:
        self._upsert_result(VisualResultRecord, case_id, result)

    def get_visual(self, case_id: str) -> dict[str, Any] | None:
        return self._get_result(VisualResultRecord, case_id)

    def save_triage(self, case_id: str, result: BaseModel) -> None:
        self._upsert_result(TriageResultRecord, case_id, result)

    def get_triage(self, case_id: str) -> dict[str, Any] | None:
        return self._get_result(TriageResultRecord, case_id)

    def save_care_pathway(self, case_id: str, result: BaseModel) -> None:
        self._upsert_result(CarePathwayRecord, case_id, result)

    def get_care_pathway(self, case_id: str) -> dict[str, Any] | None:
        return self._get_result(CarePathwayRecord, case_id)

    def add_chat_message(self, case_id: str, role: str, message: str, tool_calls: list[dict]) -> None:
        with self.sessions.begin() as db:
            db.add(
                ChatMessageRecord(
                    case_id=case_id,
                    role=role,
                    sanitized_message=message,
                    tool_calls_json=self._json(tool_calls),
                )
            )

    def list_chat_messages(self, case_id: str) -> list[dict[str, Any]]:
        with self.sessions() as db:
            records = db.scalars(
                select(ChatMessageRecord)
                .where(ChatMessageRecord.case_id == case_id)
                .order_by(ChatMessageRecord.id)
            ).all()
            return [
                {
                    "role": item.role,
                    "message": item.sanitized_message,
                    "tool_calls": json.loads(item.tool_calls_json),
                    "timestamp": item.created_at.isoformat(),
                }
                for item in records
            ]

    def clear_chat_messages(self, case_id: str) -> int:
        with self.sessions.begin() as db:
            result = db.execute(
                delete(ChatMessageRecord).where(ChatMessageRecord.case_id == case_id)
            )
            return int(result.rowcount or 0)

    def add_tool_call(
        self,
        case_id: str | None,
        tool_name: str,
        arguments: dict,
        result: dict,
        success: bool,
        latency_ms: float,
    ) -> None:
        with self.sessions.begin() as db:
            db.add(
                ToolCallRecord(
                    case_id=case_id,
                    tool_name=tool_name,
                    arguments_json=self._json(arguments),
                    result_json=self._json(result),
                    success=success,
                    latency_ms=round(latency_ms),
                )
            )

    @staticmethod
    def _public_agent_state(value: dict[str, Any]) -> dict[str, Any]:
        """Keep local asset paths and raw intake identity out of API responses."""
        return {
            key: value
            for key, value in value.items()
            if key not in {"asset_path", "private_intake", "request"}
        }

    def create_agent_run(
        self,
        *,
        run_id: str,
        user_id: int,
        case_id: str | None,
        status: str,
        current_node: str,
        state: dict[str, Any],
        pending_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.sessions.begin() as db:
            record = AgentRunRecord(
                run_id=run_id,
                user_id=user_id,
                case_id=case_id,
                status=status,
                current_node=current_node,
                state_json=self._json(state),
                pending_action_json=self._json(pending_action or {}),
            )
            db.add(record)
            db.flush()
            return self._agent_run_public(record)

    def get_agent_run_record(self, run_id: str) -> AgentRunRecord | None:
        with self.sessions() as db:
            return db.get(AgentRunRecord, run_id)

    def get_agent_run(self, run_id: str) -> dict[str, Any] | None:
        with self.sessions() as db:
            record = db.get(AgentRunRecord, run_id)
            return None if record is None else self._agent_run_public(record)

    def update_agent_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        current_node: str | None = None,
        case_id: str | None = None,
        state: dict[str, Any] | None = None,
        pending_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.sessions.begin() as db:
            record = db.get(AgentRunRecord, run_id)
            if record is None:
                raise KeyError(f"agent run not found: {run_id}")
            if status is not None:
                record.status = status
            if current_node is not None:
                record.current_node = current_node
            if case_id is not None:
                record.case_id = case_id
            if state is not None:
                record.state_json = self._json(state)
            if pending_action is not None:
                record.pending_action_json = self._json(pending_action)
            record.version += 1
            record.updated_at = datetime.utcnow()
            db.flush()
            return self._agent_run_public(record)

    def _agent_run_public(self, record: AgentRunRecord) -> dict[str, Any]:
        state = json.loads(record.state_json or "{}")
        return {
            "run_id": record.run_id,
            "case_id": record.case_id,
            "user_id": record.user_id,
            "status": record.status,
            "current_node": record.current_node,
            "state": self._public_agent_state(state),
            "pending_action": json.loads(record.pending_action_json or "{}"),
            "version": record.version,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }

    def add_agent_event(
        self,
        run_id: str,
        *,
        event_type: str,
        node: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.sessions.begin() as db:
            count = len(
                db.scalars(
                    select(AgentEventRecord).where(AgentEventRecord.run_id == run_id)
                ).all()
            )
            record = AgentEventRecord(
                run_id=run_id,
                sequence=count + 1,
                event_type=event_type,
                node=node,
                message=message,
                payload_json=self._json(payload or {}),
            )
            db.add(record)
            db.flush()
            return self._agent_event_public(record)

    @staticmethod
    def _agent_event_public(record: AgentEventRecord) -> dict[str, Any]:
        return {
            "sequence": record.sequence,
            "event_type": record.event_type,
            "node": record.node,
            "message": record.message,
            "payload": json.loads(record.payload_json or "{}"),
            "timestamp": record.created_at.isoformat(),
        }

    def list_agent_events(self, run_id: str) -> list[dict[str, Any]]:
        with self.sessions() as db:
            records = db.scalars(
                select(AgentEventRecord)
                .where(AgentEventRecord.run_id == run_id)
                .order_by(AgentEventRecord.sequence)
            ).all()
            return [self._agent_event_public(item) for item in records]

    def add_audit_event(self, event: dict[str, Any]) -> None:
        with self.sessions.begin() as db:
            db.add(
                AuditEventRecord(
                    event_type=event["event_type"],
                    case_id=event.get("case_id"),
                    channel=event["channel"],
                    tool_name=event.get("tool_name"),
                    destination=event["destination"],
                    privacy_action=event["privacy_action"],
                    allowed=event["allowed"],
                    sanitized_fields_json=self._json(event.get("sanitized_fields", [])),
                    latency_ms=round(event.get("latency_ms", 0)),
                )
            )

    def list_audit_events(self, case_id: str | None = None) -> list[dict[str, Any]]:
        with self.sessions() as db:
            query = select(AuditEventRecord).order_by(AuditEventRecord.id)
            if case_id:
                query = query.where(AuditEventRecord.case_id == case_id)
            return [
                {
                    "event_type": item.event_type,
                    "case_id": item.case_id,
                    "channel": item.channel,
                    "tool_name": item.tool_name,
                    "destination": item.destination,
                    "privacy_action": item.privacy_action,
                    "allowed": item.allowed,
                    "sanitized_fields": json.loads(item.sanitized_fields_json),
                    "latency_ms": item.latency_ms,
                    "timestamp": item.created_at.isoformat(),
                }
                for item in db.scalars(query).all()
            ]
