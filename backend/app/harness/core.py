from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.harness.assets import LocalAssetStore
from backend.app.harness.graph import build_confirmation_graph, build_request_graph
from backend.app.harness.open_source_agent import OpenSourceAgentRuntime
from backend.app.harness.schemas import AgentIntake, HarnessResponse, PendingAction
from backend.app.mcp.registry import ToolRegistry
from backend.app.services.container import ServiceContainer
from backend.app.services.report_service import ReportService
from backend.app.services.privacy import EgressGuard


CREATE_HINT = "请点击图片按钮选择病灶图片，并在对话框中自然描述患者身份、部位、症状与变化。"

VISUAL_MODEL_DISCLOSURE_PATTERNS = (
    r"DermFM(?:[\s‑-]*Zero(?:[\s‑-]*Open)?)?",
    r"HAM7(?:[\s‑-]*(?:Linear[\s‑-]*Head|线性任务头|任务头))?",
    r"supervised_linear_probe",
    r"supervised_multimodal_pad",
    r"dermfm_extended_[A-Za-z0-9_.-]+",
    r"ham7_supervised_[A-Za-z0-9_.-]+",
    r"pad_clinical_multimodal_[A-Za-z0-9_.-]+",
)


def redact_visual_model_details(text: str) -> str:
    result = str(text)
    for pattern in VISUAL_MODEL_DISCLOSURE_PATTERNS:
        result = re.sub(pattern, "本地视觉模型", result, flags=re.I)
    return result


class ClinicalAgentHarness:
    """Durable, clinician-controlled task harness around existing trusted services.

    The LLM is never allowed to mutate clinical state directly. Mutations are
    represented as pending actions, persisted, previewed, and executed only
    after an authenticated clinician confirms the exact action id.
    """

    def __init__(
        self,
        services: ServiceContainer,
        registry: ToolRegistry,
        guard: EgressGuard,
        assets: LocalAssetStore | None = None,
    ) -> None:
        self.services = services
        self.repository = services.repository
        self.registry = registry
        self.guard = guard
        self.assets = assets or LocalAssetStore()
        self.open_agent = OpenSourceAgentRuntime(registry, guard)
        self.request_graph = build_request_graph(self)
        self.confirmation_graph = build_confirmation_graph(self)

    def _event(
        self,
        run_id: str,
        event_type: str,
        node: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.repository.add_agent_event(
            run_id, event_type=event_type, node=node, message=message, payload=payload
        )

    def _response(self, run_id: str, message: str, result: dict[str, Any] | None = None) -> HarnessResponse:
        run = self.repository.get_agent_run(run_id)
        if run is None:
            raise KeyError(f"agent run not found: {run_id}")
        pending = run.get("pending_action") or None
        return HarnessResponse(
            run_id=run_id,
            status=run["status"],
            current_node=run["current_node"],
            case_id=run.get("case_id"),
            message=message,
            version=run["version"],
            pending_action=PendingAction.model_validate(pending) if pending else None,
            result=result or run.get("state", {}).get("last_result", {}),
            events=self.repository.list_agent_events(run_id),
        )

    def _new_run(self, user_id: int, case_id: str | None, message: str) -> str:
        run_id = f"RUN_{uuid4().hex[:12]}"
        self.repository.create_agent_run(
            run_id=run_id,
            user_id=user_id,
            case_id=case_id,
            status="RUNNING",
            current_node="intake",
            state={"request": message, "tool_budget": 8, "tools_used": 0},
        )
        self._event(run_id, "status", "intake", "已创建Agent任务")
        return run_id

    @staticmethod
    def _safe_intake_preview(intake: AgentIntake, asset_id: str) -> dict[str, Any]:
        return {
            "患者": intake.patient_name,
            "患者编号": intake.outpatient_no,
            "性别": intake.sex,
            "年龄": intake.age,
            "病灶部位": intake.lesion_site,
            "病例描述": intake.chief_complaint,
            "图片资产": asset_id,
            "结构化病例": intake.clinical_metadata.model_dump(mode="json"),
        }

    async def start(
        self,
        *,
        user_id: int,
        message: str,
        case_id: str | None,
        model: str,
        detail_level: str,
        image_bytes: bytes | None = None,
        image_content_type: str | None = None,
        intake: AgentIntake | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        conversation_id: str | None = None,
    ) -> HarnessResponse:
        message = message.strip()
        if case_id and self.repository.get_case(case_id) is None:
            raise KeyError(f"case not found: {case_id}")
        run_id = self._new_run(user_id, case_id, message)
        return await self.execute(
            run_id=run_id, user_id=user_id, message=message, case_id=case_id,
            model=model, detail_level=detail_level, image_bytes=image_bytes,
            image_content_type=image_content_type, intake=intake,
            conversation_history=conversation_history,
            conversation_id=conversation_id,
        )

    def create_run(self, *, user_id: int, case_id: str | None, message: str) -> HarnessResponse:
        if case_id and self.repository.get_case(case_id) is None:
            raise KeyError(f"case not found: {case_id}")
        run_id = self._new_run(user_id, case_id, message.strip())
        return self._response(run_id, "任务已接收，正在规划执行路径。")

    async def execute(
        self, *, run_id: str, user_id: int, message: str, case_id: str | None,
        model: str, detail_level: str, image_bytes: bytes | None = None,
        image_content_type: str | None = None, intake: AgentIntake | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        conversation_id: str | None = None,
    ) -> HarnessResponse:
        try:
            final = await self.request_graph.ainvoke({
                "run_id": run_id,
                "user_id": user_id,
                "message": message,
                "case_id": case_id,
                "model": model,
                "detail_level": detail_level,
                "image_bytes": image_bytes,
                "image_content_type": image_content_type,
                "intake": intake,
                "conversation_history": (conversation_history or [])[-12:],
                "conversation_id": conversation_id or f"run:{run_id}",
            })
            return final["response"]
        except Exception:
            self.repository.update_agent_run(run_id, status="FAILED", current_node="failed")
            self._event(run_id, "error", "failed", "LangGraph任务执行失败")
            raise

    def response(self, run_id: str, message: str = "已同步当前任务状态。") -> HarnessResponse:
        return self._response(run_id, message)

    async def graph_route_request(self, state: dict[str, Any]) -> dict[str, Any]:
        run_id = state["run_id"]
        message = state.get("message", "")
        has_image = state.get("image_bytes") is not None
        has_intake = state.get("intake") is not None
        if has_image:
            intent = "prepare_case" if has_intake else "collect_materials"
        elif message in {"暂停", "暂停任务", "先暂停"}:
            intent = "pause"
        elif state.get("case_id") and self._parse_case_patch(message):
            intent = "prepare_update"
        elif state.get("case_id") and self._is_report_command(message):
            intent = "generate_report"
        elif state.get("case_id") and self._is_analysis_command(message):
            intent = "analyze_case"
        elif not state.get("case_id"):
            intent = "collect_materials" if self._needs_case_materials(message) else "general_chat"
        elif not self._looks_medical(message) and not self._looks_case_context(message):
            intent = "general_chat"
        else:
            intent = "clinical_chat"
        self.repository.update_agent_run(run_id, current_node="route_request")
        self._event(
            run_id, "transition", "route_request", "LangGraph已完成任务路由",
            {"next_node": intent},
        )
        return {"intent": intent}

    async def graph_prepare_case(self, state: dict[str, Any]) -> dict[str, Any]:
        run_id = state["run_id"]
        intake = AgentIntake.model_validate(state["intake"])
        asset = self.assets.stage(state["image_bytes"], state.get("image_content_type") or "")
        action = PendingAction(
            action_id=f"ACT_{uuid4().hex[:12]}",
            kind="CREATE_ANALYZE_CASE",
            title="创建病例并执行本地多模态分析",
            summary="确认后将写入患者与病例信息，并调用本地多模态模型。",
            preview=self._safe_intake_preview(intake, asset.asset_id),
        )
        private_state = {
            "request": state.get("message", ""), "tool_budget": 8, "tools_used": 0,
            "asset_path": str(asset.path), "asset_content_type": asset.content_type,
            "asset_id": asset.asset_id,
            "private_intake": intake.model_dump(mode="json"),
        }
        self.repository.update_agent_run(
            run_id, status="WAITING_CONFIRM", current_node="confirm_case_creation",
            state=private_state, pending_action=action.model_dump(mode="json"),
        )
        self._event(
            run_id, "confirmation", "confirm_case_creation",
            "等待医生确认病例写入和本地分析", {"action_id": action.action_id},
        )
        return {"response": self._response(run_id, "资料已准备完成，请核对后确认创建病例并开始分析。")}

    async def graph_collect_materials(self, state: dict[str, Any]) -> dict[str, Any]:
        run_id = state["run_id"]
        partial = state.get("image_bytes") is not None or state.get("intake") is not None
        self.repository.update_agent_run(
            run_id, status="WAITING_USER", current_node="collect_case_materials"
        )
        self._event(run_id, "waiting", "collect_case_materials", "等待图片和病例资料")
        prefix = "病例资料和病灶图片需要同时提交。" if partial else ""
        return {"response": self._response(run_id, prefix + CREATE_HINT)}

    async def graph_pause(self, state: dict[str, Any]) -> dict[str, Any]:
        run_id = state["run_id"]
        self.repository.update_agent_run(run_id, status="PAUSED", current_node="paused")
        self._event(run_id, "paused", "paused", "任务已由医生暂停")
        return {"response": self._response(run_id, "当前任务已暂停，运行状态和事件已保存。")}

    async def graph_prepare_update(self, state: dict[str, Any]) -> dict[str, Any]:
        run_id = state["run_id"]
        patch = self._parse_case_patch(state.get("message", ""))
        action = PendingAction(
            action_id=f"ACT_{uuid4().hex[:12]}", kind="UPDATE_CASE",
            title="修改病例记录", summary="确认后写入病例并重新生成分析证据。",
            preview=patch,
        )
        current = self.repository.get_agent_run(run_id) or {}
        private_state = {**current.get("state", {}), "case_patch": patch}
        self.repository.update_agent_run(
            run_id, status="WAITING_CONFIRM", current_node="confirm_case_update",
            state=private_state, pending_action=action.model_dump(mode="json"),
        )
        self._event(run_id, "confirmation", "confirm_case_update", "等待医生确认病例修改")
        return {"response": self._response(run_id, "已生成病例修改预览，请确认后写入。")}

    async def graph_generate_report(self, state: dict[str, Any]) -> dict[str, Any]:
        run_id = state["run_id"]
        action = PendingAction(
            action_id=f"ACT_{uuid4().hex[:12]}", kind="GENERATE_REPORT",
            title="生成并保存辅助诊断报告",
            summary="确认后将基于当前病例的结构化证据生成报告草稿并写入报告库。",
            preview={"病例编号": state["case_id"], "模型": state.get("model", "auto")},
        )
        self.repository.update_agent_run(
            run_id, status="WAITING_CONFIRM", current_node="confirm_report_generation",
            state={"report_model": state.get("model", "auto")},
            pending_action=action.model_dump(mode="json"),
        )
        self._event(run_id, "confirmation", "confirm_report_generation", "等待医生确认生成报告")
        return {"response": self._response(run_id, "报告生成任务已准备，请核对后确认执行。")}

    async def graph_analyze_case(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"response": self._run_analysis(state["run_id"], state["case_id"])}

    async def graph_general_chat(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run every knowledge or general conversation through Deep Agents."""
        run_id = state["run_id"]
        if not self.open_agent.available(state.get("model", "auto")):
            self._event(
                run_id, "error", "open_source_agent",
                "Deep Agent模型不可用，请检查所选模型与服务密钥。",
            )
            raise RuntimeError("Deep Agent模型不可用")
        return {"response": await self._run_open_source_agent(state)}

    async def graph_clinical_chat(self, state: dict[str, Any]) -> dict[str, Any]:
        run_id = state["run_id"]
        if not self.open_agent.available(state.get("model", "auto")):
            self._event(
                run_id, "error", "open_source_agent",
                "Deep Agent模型不可用，请检查所选模型与服务密钥。",
            )
            raise RuntimeError("Deep Agent模型不可用")
        return {"response": await self._run_open_source_agent(state)}

    async def _run_open_source_agent(self, state: dict[str, Any]) -> HarnessResponse:
        run_id = state["run_id"]
        case_id = state.get("case_id")
        self.repository.update_agent_run(run_id, current_node="open_source_agent")
        self._event(run_id, "plan", "open_source_agent", "Deep Agent 正在理解目标、拆解任务并选择工具")

        def on_event(node: str, message: str) -> None:
            event_type = "plan" if node in {"agent_context", "agent_planning", "task_plan"} else "tool"
            self._event(run_id, event_type, node, message)

        output = await self.open_agent.run(
            message=state.get("message", ""), model_id=state.get("model", "auto"),
            user_id=int(state["user_id"]),
            conversation_id=state.get("conversation_id") or f"run:{run_id}",
            case_id=case_id, history=state.get("conversation_history") or [],
            on_event=on_event,
        )
        answer = redact_visual_model_details(output["response"])
        output["response"] = answer
        result = {"kind": "chat", "chat": output}
        proposal = output.get("proposed_action")
        if proposal:
            action = self._pending_from_agent_proposal(proposal, case_id, state)
            self.repository.update_agent_run(
                run_id, status="WAITING_CONFIRM", current_node="confirm_agent_action",
                state={
                    "request": state.get("message", ""),
                    "agent_action": proposal,
                    "report_model": state.get("model", "auto"),
                    "last_result": result,
                    "tools_used": len(output.get("tool_calls", [])),
                },
                pending_action=action.model_dump(mode="json"),
            )
            self._event(
                run_id, "confirmation", "confirm_agent_action",
                "Agent 已完成任务拆解，关键写操作等待医生确认",
                {"action_kind": action.kind, "action_id": action.action_id},
            )
            return self._response(run_id, answer, result)
        self.repository.update_agent_run(
            run_id, status="COMPLETED", current_node="completed",
            state={"last_result": result, "tools_used": len(output.get("tool_calls", []))},
        )
        if case_id:
            self.repository.add_chat_message(case_id, "user", state.get("message", ""), [])
            self.repository.add_chat_message(case_id, "assistant", answer, output.get("tool_calls", []))
        self._event(run_id, "completed", "completed", "Deep Agent 已完成计划、工具执行与结果整合")
        return self._response(run_id, answer, result)

    @staticmethod
    def _pending_from_agent_proposal(
        proposal: dict[str, Any], case_id: str | None, state: dict[str, Any]
    ) -> PendingAction:
        kind = str(proposal.get("kind", ""))
        if not case_id and kind not in {"DELETE_DOCUMENT", "DELETE_REPORT"}:
            raise ValueError("待确认操作必须绑定病例")
        payload = proposal.get("payload") or {}
        definitions = {
            "UPDATE_CASE": (
                "修改病例记录", "确认后将更新当前病例并重新生成分析证据。", False,
            ),
            "GENERATE_REPORT": (
                "生成并保存辅助诊断报告",
                "确认后将基于当前结构化证据生成报告草稿并写入报告库。", False,
            ),
            "DELETE_REPORT": (
                "删除已保存报告", "确认后只删除指定报告，病例和原始图像将保留。", True,
            ),
            "DELETE_DOCUMENT": (
                "删除已存文档", "确认后将删除指定文档及其文件夹归档关系，关联病例和报告不受影响。", True,
            ),
        }
        if kind not in definitions:
            raise ValueError("不支持的 Agent 写操作")
        title, summary, destructive = definitions[kind]
        preview = ({"病例编号": case_id} if case_id else {}) | payload
        if kind == "GENERATE_REPORT":
            preview["模型"] = state.get("model", "auto")
        return PendingAction(
            action_id=f"ACT_{uuid4().hex[:12]}", kind=kind,
            title=title, summary=summary, preview=preview, destructive=destructive,
        )

    async def graph_route_confirmation(self, state: dict[str, Any]) -> dict[str, Any]:
        kind = state.get("action_kind", "")
        if kind not in {
            "CREATE_ANALYZE_CASE", "UPDATE_CASE", "GENERATE_REPORT", "DELETE_REPORT",
            "DELETE_DOCUMENT",
        }:
            raise ValueError("不支持的待确认操作")
        self._event(
            state["run_id"], "transition", "route_confirmation",
            "LangGraph已进入受控写操作节点", {"action_kind": kind},
        )
        return {"action_kind": kind}

    async def graph_confirm_create_case(self, state: dict[str, Any]) -> dict[str, Any]:
        run_id, user_id = state["run_id"], state["user_id"]
        record = self.repository.get_agent_run_record(run_id)
        if record is None:
            raise KeyError(f"agent run not found: {run_id}")
        private = self._private_state(record.state_json)
        intake = AgentIntake.model_validate(private["private_intake"])
        path = Path(private["asset_path"])
        patient = self.repository.create_patient(
            name=intake.patient_name, sex=intake.sex, age=intake.age,
            outpatient_no=intake.outpatient_no, phone=intake.phone or None,
            medical_history=intake.medical_history,
            allergy_history=intake.allergy_history, created_by=user_id,
        )
        self._event(run_id, "tool", "create_patient", "患者档案已写入本地数据库")
        case_id = self.services.cases.create(
            path.read_bytes(), private["asset_content_type"], mode="clinician", retention_mode="local"
        )
        self.repository.link_case_patient(
            case_id, patient.patient_id, intake.lesion_site, intake.chief_complaint,
            intake.clinical_metadata.model_dump(mode="json"),
        )
        self.assets.discard(path)
        self.repository.update_agent_run(run_id, case_id=case_id, current_node="multimodal_analysis")
        self._event(run_id, "tool", "multimodal_analysis", "正在执行本地图像—病例联合分析")
        return {"case_id": case_id, "response": self._finish_analysis(
            run_id, case_id, "病例已创建，多模态分析已完成。"
        )}

    async def graph_confirm_update_case(self, state: dict[str, Any]) -> dict[str, Any]:
        run_id = state["run_id"]
        case_id = state.get("case_id")
        if not case_id:
            raise ValueError("当前任务未绑定病例")
        record = self.repository.get_agent_run_record(run_id)
        if record is None:
            raise KeyError(f"agent run not found: {run_id}")
        private = self._private_state(record.state_json)
        patch = private.get("case_patch") or (
            (private.get("agent_action") or {}).get("payload") or {}
        )
        patch = {
            key: value for key, value in patch.items()
            if key in {"chief_complaint", "lesion_site"}
        }
        if not patch:
            raise ValueError("待确认的病例修改内容为空")
        self.repository.update_case_patient(case_id, **patch)
        self._event(run_id, "tool", "update_case", "病例修改已写入并保存操作轨迹")
        return {"response": self._finish_analysis(
            run_id, case_id, "病例已更新，并重新生成分析与医学证据。"
        )}

    async def graph_confirm_generate_report(self, state: dict[str, Any]) -> dict[str, Any]:
        record = self.repository.get_agent_run_record(state["run_id"])
        if record is None or not state.get("case_id"):
            raise KeyError("Agent 任务或病例不存在")
        private = self._private_state(record.state_json)
        return {"response": self._generate_report(
            state["run_id"], state["user_id"], state["case_id"],
            str(private.get("report_model") or "auto"),
        )}

    async def graph_confirm_delete_report(self, state: dict[str, Any]) -> dict[str, Any]:
        run_id, case_id = state["run_id"], state.get("case_id")
        record = self.repository.get_agent_run_record(run_id)
        if record is None:
            raise KeyError("Agent 任务不存在")
        proposal = self._private_state(record.state_json).get("agent_action") or {}
        report_id = str((proposal.get("payload") or {}).get("report_id", ""))
        report = self.repository.get_report_for_user(report_id, state["user_id"])
        if report is None or (case_id and report.get("case_id") != case_id):
            raise ValueError("待删除报告不存在或当前账号无权访问")
        case_id = report.get("case_id")
        if not self.repository.delete_report(report_id, state["user_id"]):
            raise RuntimeError("报告删除失败")
        result = {
            "kind": "operation", "operation": "delete_report",
            "report_id": report_id, "case_id": case_id,
        }
        self.repository.update_agent_run(
            run_id, status="COMPLETED", current_node="completed",
            state={"last_result": result, "tools_used": 1}, pending_action={},
        )
        self._event(run_id, "completed", "completed", "指定报告已删除，病例与原始图像已保留")
        return {"response": self._response(run_id, "报告已删除，病例和原始图像未受影响。", result)}

    async def graph_confirm_delete_document(self, state: dict[str, Any]) -> dict[str, Any]:
        run_id, user_id = state["run_id"], state["user_id"]
        record = self.repository.get_agent_run_record(run_id)
        if record is None:
            raise KeyError("Agent 任务不存在")
        proposal = self._private_state(record.state_json).get("agent_action") or {}
        document_id = str((proposal.get("payload") or {}).get("document_id", ""))
        document = self.repository.get_agent_document(document_id, user_id)
        if document is None:
            raise ValueError("待删除文档不存在或当前账号无权访问")
        if not self.repository.delete_agent_document(document_id, user_id):
            raise RuntimeError("文档删除失败")
        result = {
            "kind": "operation", "operation": "delete_document",
            "document_id": document_id, "case_id": document.get("case_id"),
        }
        self.repository.update_agent_run(
            run_id, status="COMPLETED", current_node="completed",
            state={"last_result": result, "tools_used": 1}, pending_action={},
        )
        self._event(run_id, "completed", "completed", "指定文档已删除，关联病例和报告已保留")
        return {"response": self._response(
            run_id, "文档已删除，关联病例和报告未受影响。", result
        )}

    async def confirm(self, run_id: str, user_id: int, action_id: str) -> HarnessResponse:
        record = self.repository.get_agent_run_record(run_id)
        if record is None:
            raise KeyError(f"agent run not found: {run_id}")
        if record.user_id != user_id:
            raise PermissionError("无权操作其他医生的Agent任务")
        run = self.repository.get_agent_run(run_id)
        assert run is not None
        action = run.get("pending_action") or {}
        if run["status"] != "WAITING_CONFIRM" or action.get("action_id") != action_id:
            raise ValueError("待确认操作已失效，请重新发起")
        kind = action.get("kind")
        self.repository.update_agent_run(run_id, status="RUNNING", current_node="execute_confirmed_action")
        self._event(run_id, "confirmed", "execute_confirmed_action", "医生已确认操作")
        try:
            final = await self.confirmation_graph.ainvoke({
                "run_id": run_id,
                "user_id": user_id,
                "case_id": run.get("case_id"),
                "action_kind": kind,
            })
            return final["response"]
        except Exception:
            self.repository.update_agent_run(run_id, status="FAILED", current_node="failed")
            self._event(run_id, "error", "failed", "确认后的操作执行失败")
            raise

    def reject(self, run_id: str, user_id: int) -> HarnessResponse:
        record = self.repository.get_agent_run_record(run_id)
        if record is None:
            raise KeyError(f"agent run not found: {run_id}")
        if record.user_id != user_id:
            raise PermissionError("无权操作其他医生的Agent任务")
        private = self._private_state(record.state_json)
        self.assets.discard(private.get("asset_path"))
        self.repository.update_agent_run(
            run_id, status="CANCELLED", current_node="cancelled", pending_action={}
        )
        self._event(run_id, "cancelled", "cancelled", "医生已取消待执行操作")
        return self._response(run_id, "操作已取消，未写入病例。")

    def pause(self, run_id: str, user_id: int) -> HarnessResponse:
        record = self.repository.get_agent_run_record(run_id)
        if record is None or record.user_id != user_id:
            raise KeyError("Agent任务不存在")
        if record.status in {"COMPLETED", "FAILED", "CANCELLED"}:
            raise ValueError("当前任务已经结束")
        self.repository.update_agent_run(run_id, status="PAUSED", current_node="paused")
        self._event(run_id, "paused", "paused", "任务已由医生暂停")
        return self._response(run_id, "任务已暂停，当前状态和事件记录已保存。")

    async def resume(
        self, run_id: str, user_id: int, message: str,
        model: str = "auto", detail_level: str = "concise",
    ) -> HarnessResponse:
        record = self.repository.get_agent_run_record(run_id)
        if record is None:
            raise KeyError("Agent任务不存在")
        if record.user_id != user_id:
            raise PermissionError("无权操作其他医生的Agent任务")
        if record.status not in {"PAUSED", "WAITING_USER"}:
            raise ValueError("只能恢复已暂停或等待补充信息的任务")
        if not record.case_id:
            raise ValueError("该任务尚未绑定病例，请通过聊天框附加图片和病例资料")
        self.repository.update_agent_run(run_id, status="RUNNING", current_node="resume")
        self._event(run_id, "resumed", "resume", "医生已恢复任务并提交新指令")
        try:
            final = await self.request_graph.ainvoke({
                "run_id": run_id, "user_id": user_id, "message": message.strip(),
                "case_id": record.case_id, "model": model, "detail_level": detail_level,
                "image_bytes": None, "image_content_type": None, "intake": None,
            })
            return final["response"]
        except Exception:
            self.repository.update_agent_run(run_id, status="FAILED", current_node="failed")
            self._event(run_id, "error", "failed", "恢复后的LangGraph任务执行失败")
            raise

    @staticmethod
    def _private_state(raw: str) -> dict[str, Any]:
        import json
        return json.loads(raw or "{}")

    def _finish_analysis(self, run_id: str, case_id: str, message: str) -> HarnessResponse:
        analysis = self.services.analysis.run(case_id)
        result = {"kind": "analysis", "analysis": analysis}
        self.repository.update_agent_run(
            run_id, status="COMPLETED", current_node="completed", case_id=case_id,
            state={"last_result": result, "tools_used": 4}, pending_action={},
        )
        labels = [item["label"] for item in analysis["visual"]["top_predictions"][:3]]
        self._event(
            run_id, "completed", "completed", "多模态分析、风险分级和医学检索已完成",
            {"candidate_labels": labels, "triage_level": analysis["triage"]["level"]},
        )
        return self._response(run_id, message, result)

    def _run_analysis(self, run_id: str, case_id: str) -> HarnessResponse:
        self.repository.update_agent_run(run_id, current_node="multimodal_analysis")
        self._event(run_id, "tool", "multimodal_analysis", "正在重新执行本地多模态分析")
        return self._finish_analysis(run_id, case_id, "多模态分析已完成。")

    def _generate_report(self, run_id: str, user_id: int, case_id: str, model: str) -> HarnessResponse:
        patient = self.repository.get_case_patient(case_id)
        if patient is None:
            raise KeyError("该病例尚未关联患者")
        self.repository.update_agent_run(run_id, current_node="report_generation")
        self._event(run_id, "tool", "report_generation", "正在基于脱敏结构化证据生成报告草稿")
        clinical_payload = {
            "visual": self.repository.get_visual(case_id) or {},
            "triage": self.repository.get_triage(case_id) or {},
            "care_pathway": self.repository.get_care_pathway(case_id) or {},
            "clinical_profile": patient.get("clinical_profile", {}),
        }
        model_id = "local-template" if model == "auto" else model
        generated = ReportService(self.guard).generate(
            case_id=case_id, patient=patient,
            clinical_payload=clinical_payload, model_id=model_id,
        )
        record = self.repository.create_report(
            case_id=case_id, patient_id=patient["patient_id"],
            content=generated.content, structured=generated.structured,
            provider=generated.provider, model=generated.model, created_by=user_id,
        )
        report = self.repository.get_report(record.report_id)
        if report is None:
            raise RuntimeError("报告保存失败")
        result = {"kind": "report", "report": report}
        self.repository.update_agent_run(
            run_id, status="COMPLETED", current_node="completed",
            state={"last_result": result, "tools_used": 1}, pending_action={},
        )
        self._event(run_id, "completed", "completed", "报告草稿已生成并保存到病例库")
        return self._response(run_id, "报告草稿已生成，可在病例报告库继续编辑。", result)

    @staticmethod
    def _parse_case_patch(message: str) -> dict[str, str]:
        patterns = {
            "chief_complaint": r"(?:把|将)?(?:当前)?(?:病例描述|主诉)(?:修改|更新|改)(?:为|成)?[：:\s]*(.+)",
            "lesion_site": r"(?:把|将)?(?:病灶)?部位(?:修改|更新|改)(?:为|成)?[：:\s]*(.+)",
        }
        for field, pattern in patterns.items():
            match = re.search(pattern, message.strip())
            if match:
                value = match.group(1).strip("。；; \n")
                if value:
                    return {field: value[:5000]}
        return {}

    @staticmethod
    def _is_report_command(message: str) -> bool:
        return any(term in message for term in ("生成报告", "创建报告", "写一份报告", "报告初稿"))

    @staticmethod
    def _is_analysis_command(message: str) -> bool:
        return any(term in message for term in ("重新分析", "开始分析", "执行分析", "检测这张", "识别这张"))

    @staticmethod
    def _needs_case_materials(message: str) -> bool:
        return any(term in message for term in (
            "创建病例", "新建病例", "分析这张", "识别这张", "检测这张",
            "上传图片", "病灶图片", "联合分析",
        ))

    @staticmethod
    def _looks_medical(message: str) -> bool:
        return any(term in message for term in (
            "皮肤", "病灶", "症状", "疾病", "痣", "疹", "斑", "瘤", "癌",
            "痤", "疣", "炎", "癣", "痒", "疼", "出血", "药", "治疗", "诊断",
            "医生", "医学", "转诊", "隐私", "原始图片", "上传到云",
            "血压", "发热", "咳嗽", "感染", "头痛", "腹痛", "糖尿病", "心脏",
            "肾脏", "疫苗", "手术", "医学检查", "用药", "剂量",
        ))

    @staticmethod
    def _looks_case_context(message: str) -> bool:
        return any(term in message for term in (
            "当前病例", "候选", "Top-", "top-", "分析结果", "风险分级",
            "医学证据", "指南依据", "检查依据", "转诊依据",
        ))

