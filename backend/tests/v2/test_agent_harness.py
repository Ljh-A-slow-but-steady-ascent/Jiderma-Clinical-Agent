from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from backend.app.api.v2_routes import create_v2_runtime, get_runtime
from backend.app.care.schemas import ClinicalMetadata
from backend.app.database import CaseRepository, init_database
from backend.app.main import app
from backend.app.harness.core import ClinicalAgentHarness
from backend.app.services.container import create_service_container


class FakeMultimodalRunner:
    def predict(self, image_path: Path, clinical_metadata: ClinicalMetadata) -> dict:
        assert image_path.is_file()
        assert clinical_metadata.bleed is True
        return {
            "model_name": "local_multimodal_classifier",
            "top_predictions": [
                {"label": "nevus", "score": 0.76},
                {"label": "melanoma", "score": 0.19},
                {"label": "basal cell carcinoma", "score": 0.05},
            ],
            "latency_ms": 12.0,
        }


class FakeDeepAgentRuntime:
    @staticmethod
    def available(model_id: str) -> bool:
        return True

    async def run(self, *, message: str, on_event, **kwargs) -> dict:
        on_event("agent_context", "正在结合上下文识别任务目标")
        if "色素痣" in message:
            on_event("search_guideline", "已完成本地医学知识检索")
            response = "色素痣是常见的色素性皮肤病灶。"
            tools = [{"tool_name": "search_guideline", "status": "success", "latency_ms": 0}]
        elif "1+1" in message:
            on_event("calculator", "已完成本地计算")
            response = "计算结果：2"
            tools = [{"tool_name": "calculator", "status": "success", "latency_ms": 0}]
        else:
            response = "我已结合当前对话继续处理。"
            tools = []
        return {
            "response": response,
            "tool_calls": tools,
            "provider": "deepagents",
            "model": "test-agent-model",
            "task_plan": [],
            "artifacts": [],
            "agent_trace": [{
                "step": 1, "intent": "deep_agent_runtime",
                "decision": "MODEL_TOOL_LOOP_COMPLETED",
            }],
            "proposed_action": None,
        }


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def harness_client(tmp_path):
    repository = CaseRepository(init_database("sqlite+pysqlite:///:memory:"))
    services = create_service_container(repository)
    services.cases.data_root = tmp_path / "cases"
    services.visual.runner = FakeMultimodalRunner()
    runtime = create_v2_runtime(services)
    runtime.harness.assets.root = tmp_path / "agent_uploads"
    runtime.harness.open_agent = FakeDeepAgentRuntime()

    async def override_runtime():
        return runtime

    app.dependency_overrides[get_runtime] = override_runtime
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
    app.dependency_overrides.clear()


def intake() -> dict:
    return {
        "patient_name": "测试患者", "sex": "男", "age": 65,
        "outpatient_no": "HARNESS-001", "phone": "", "medical_history": "",
        "allergy_history": "", "lesion_site": "左前臂",
        "chief_complaint": "左前臂色素性病灶近期增大并偶有出血",
        "clinical_metadata": {
            "age": 65, "gender": "MALE", "region": "FOREARM",
            "itch": False, "hurt": False, "bleed": True, "grew": True,
            "changed": True, "diameter_1": 9, "diameter_2": 7,
            "skin_cancer_history": False, "cancer_history": False,
        },
    }


@pytest.mark.anyio
async def test_langgraph_harness_requires_confirmation_then_runs_analysis(harness_client):
    image = Path("third_party/DermFM-Zero/examples/PAT_8_15_820.png").read_bytes()
    prepared = await harness_client.post(
        "/v1/agent/runs",
        files={"image": ("lesion.png", image, "image/png")},
        data={
            "message": "创建病例并分析", "model": "auto",
            "detail_level": "concise", "intake_json": json.dumps(intake(), ensure_ascii=False),
        },
    )
    assert prepared.status_code == 200, prepared.text
    pending = prepared.json()
    assert pending["status"] == "WAITING_CONFIRM"
    assert pending["pending_action"]["kind"] == "CREATE_ANALYZE_CASE"
    assert any(event["node"] == "route_request" for event in pending["events"])
    assert "asset_path" not in prepared.text
    assert "private_intake" not in prepared.text

    confirmed = await harness_client.post(
        f'/v1/agent/runs/{pending["run_id"]}/confirm',
        json={"action_id": pending["pending_action"]["action_id"]},
    )
    assert confirmed.status_code == 200, confirmed.text
    completed = confirmed.json()
    assert completed["status"] == "COMPLETED"
    assert completed["case_id"]
    assert completed["result"]["kind"] == "analysis"
    assert completed["result"]["analysis"]["visual"]["top_predictions"][0]["label"] == "nevus"
    assert any(event["node"] == "route_confirmation" for event in completed["events"])

    restored = await harness_client.get(f'/v1/agent/runs/{pending["run_id"]}')
    assert restored.status_code == 200
    assert restored.json()["status"] == "COMPLETED"

    rerun = await harness_client.post(
        "/v1/agent/runs",
        data={"message": "开始分析", "case_id": completed["case_id"], "model": "auto"},
    )
    assert rerun.status_code == 200, rerun.text
    assert rerun.json()["status"] == "COMPLETED"
    assert rerun.json()["current_node"] == "completed"
    assert rerun.json()["result"]["kind"] == "analysis"

    report_request = await harness_client.post(
        "/v1/agent/runs",
        data={
            "message": "请生成报告", "case_id": completed["case_id"],
            "model": "auto", "detail_level": "concise",
        },
    )
    assert report_request.status_code == 200, report_request.text
    report_pending = report_request.json()
    assert report_pending["status"] == "WAITING_CONFIRM"
    assert report_pending["pending_action"]["kind"] == "GENERATE_REPORT"

    report_confirmed = await harness_client.post(
        f'/v1/agent/runs/{report_pending["run_id"]}/confirm',
        json={"action_id": report_pending["pending_action"]["action_id"]},
    )
    assert report_confirmed.status_code == 200, report_confirmed.text
    report_result = report_confirmed.json()
    assert report_result["status"] == "COMPLETED"
    assert report_result["result"]["kind"] == "report"
    assert report_result["result"]["report"]["case_id"] == completed["case_id"]
    assert report_result["pending_action"] is None


@pytest.mark.anyio
async def test_harness_waits_for_materials_without_a_case(harness_client):
    response = await harness_client.post(
        "/v1/agent/runs", data={"message": "请创建新病例并联合分析"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "WAITING_USER"
    assert payload["current_node"] == "collect_case_materials"


@pytest.mark.anyio
async def test_harness_answers_general_medical_question_without_forcing_case(harness_client):
    response = await harness_client.post(
        "/v1/agent/runs", data={"message": "色素痣是什么？", "model": "auto"}
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "COMPLETED"
    assert payload["case_id"] is None
    assert "色素痣" in payload["message"]
    assert payload["result"]["kind"] == "chat"
    assert payload["result"]["chat"]["tool_calls"][0]["tool_name"] == "search_guideline"
    assert any(event["node"] == "open_source_agent" for event in payload["events"])


@pytest.mark.anyio
async def test_dispatch_returns_running_then_exposes_real_events(harness_client):
    dispatched = await harness_client.post(
        "/v1/agent/runs/dispatch", data={"message": "计算 1+1", "model": "auto"}
    )
    assert dispatched.status_code == 200, dispatched.text
    initial = dispatched.json()
    assert initial["status"] == "RUNNING"

    payload = initial
    for _ in range(30):
        await asyncio.sleep(0.01)
        response = await harness_client.get(f'/v1/agent/runs/{initial["run_id"]}')
        payload = response.json()
        if payload["status"] != "RUNNING":
            break
    assert payload["status"] == "COMPLETED"
    assert "2" in payload["message"]
    assert any(event["event_type"] == "transition" for event in payload["events"])


@pytest.mark.anyio
async def test_dispatch_accepts_conversation_thread_and_recent_history(harness_client):
    response = await harness_client.post(
        "/v1/agent/runs/dispatch",
        data={
            "message": "为什么？", "model": "auto", "conversation_id": "conv-test-1",
            "history_json": json.dumps([
                {"role": "user", "message": "请解释 LangGraph"},
                {"role": "assistant", "message": "LangGraph 是状态图编排框架。"},
            ], ensure_ascii=False),
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "RUNNING"
