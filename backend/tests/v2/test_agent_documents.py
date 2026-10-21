from __future__ import annotations

import json

import httpx
import pytest

from backend.app.api.v2_routes import create_v2_runtime, get_runtime
from backend.app.database import CaseRepository, init_database
from backend.app.harness.open_source_agent import OpenSourceAgentRuntime
from backend.app.main import app
from backend.app.mcp.factory import create_tool_registry
from backend.app.services.container import create_service_container
from backend.app.services.privacy import EgressGuard


@pytest.mark.anyio
async def test_agent_document_tools_create_rewrite_and_list() -> None:
    repository = CaseRepository(init_database("sqlite+pysqlite:///:memory:"))
    services = create_service_container(repository)
    runtime = OpenSourceAgentRuntime(
        create_tool_registry(services), EgressGuard(repository)
    )
    events: list[tuple[str, str]] = []
    proposals: list[dict] = []
    tools = {
        tool.name: tool
        for tool in runtime._tools(
            7, "conversation-test", None,
            lambda node, message: events.append((node, message)), proposals,
        )
    }

    created = json.loads(await tools["create_document"].ainvoke({
        "title": "随访说明", "filename": "随访说明.md", "content": "第一版",
    }))
    document_id = created["document_id"]
    assert created["editable"] is True
    assert created["downloadable"] is True

    rewritten = json.loads(await tools["rewrite_document"].ainvoke({
        "document_id": document_id, "title": "随访说明", "content": "第二版",
    }))
    assert rewritten["revision"] == 2
    items = repository.list_agent_documents(
        user_id=7, conversation_id="conversation-test"
    )
    assert items[0]["content"] == "第二版"
    assert items[0]["revision"] == 2
    assert any(node == "create_document" for node, _ in events)
    assert any(node == "rewrite_document" for node, _ in events)

    moved = json.loads(await tools["move_document_to_folder"].ainvoke({
        "document_id": document_id, "folder_name": "随访病例",
    }))
    assert moved["moved"] is True
    assert moved["folder_created"] is True
    assert repository.folder_resource_ids(7, moved["folder_id"], "document") == [document_id]

    moved_again = json.loads(await tools["move_document_to_folder"].ainvoke({
        "document_id": document_id, "folder_name": "随访病例",
    }))
    assert moved_again["folder_id"] == moved["folder_id"]
    assert moved_again["folder_created"] is False

    copied = json.loads(await tools["copy_archive_file"].ainvoke({
        "resource_type": "document", "resource_id": document_id,
        "folder_name": "副本归档",
    }))
    assert copied["copied"] is True
    assert copied["copied_id"] != document_id
    assert repository.get_agent_document(copied["copied_id"], 7) is not None
    moved_generic = json.loads(await tools["move_archive_file"].ainvoke({
        "resource_type": "document", "resource_id": copied["copied_id"],
        "folder_name": "复核资料",
    }))
    assert moved_generic["moved"] is True
    assert repository.folder_resource_ids(
        7, moved_generic["folder_id"], "document"
    ) == [copied["copied_id"]]

    proposal = json.loads(await tools["propose_delete_document"].ainvoke({
        "document_id": document_id,
    }))
    assert proposal["requires_confirmation"] is True
    assert proposal["destructive"] is True
    assert proposals == [{
        "kind": "DELETE_DOCUMENT",
        "payload": {
            "document_id": document_id, "title": "随访说明",
            "filename": "随访说明.md", "case_id": None,
        },
    }]
    assert repository.get_agent_document(document_id, 7) is not None


def test_agent_documents_are_isolated_by_user_and_conversation() -> None:
    repository = CaseRepository(init_database("sqlite+pysqlite:///:memory:"))
    item = repository.create_agent_document(
        user_id=1, conversation_id="conversation-a", case_id=None,
        title="文档", filename="document.md", content="内容",
    )
    assert repository.get_agent_document(item["document_id"], 1) is not None
    assert repository.get_agent_document(item["document_id"], 2) is None
    assert repository.list_agent_documents(user_id=1, conversation_id="conversation-b") == []


@pytest.mark.anyio
async def test_agent_can_query_owned_archive_without_identity_egress() -> None:
    repository = CaseRepository(init_database("sqlite+pysqlite:///:memory:"))
    patient = repository.create_patient(
        name="小明", sex="男", age=48, outpatient_no="OP-PRIVATE-001",
        phone="13812345678", medical_history="", allergy_history="", created_by=7,
    )
    report = repository.create_report(
        case_id="CASE_PRIVATE", patient_id=patient.patient_id,
        content="小明的色素痣随访报告，门诊号 OP-PRIVATE-001，电话 13812345678。",
        structured={}, provider="local", model="template", created_by=7,
    )
    repository.create_agent_document(
        user_id=8, conversation_id="other", case_id=None,
        title="其他医生文档", filename="other.md", content="不应返回",
    )
    runtime = OpenSourceAgentRuntime(
        create_tool_registry(create_service_container(repository)), EgressGuard(repository)
    )
    tools = {tool.name: tool for tool in runtime._tools(
        7, "conversation-private", None, lambda *_: None, [],
        alias_to_identity={"[患者_01]": "小明", "[门诊号_01]": "OP-PRIVATE-001"},
    )}
    searched_raw = await tools["search_clinical_archive"].ainvoke({
        "query": "[患者_01]", "limit": 8,
    })
    searched = json.loads(searched_raw)
    assert [item["resource_id"] for item in searched["results"]] == [report.report_id]
    assert searched["results"][0]["stored_in_database"] is True
    assert searched["results"][0]["source_scope"] == "本地报告库"
    assert "小明" not in searched_raw
    assert "OP-PRIVATE-001" not in searched_raw
    assert "13812345678" not in searched_raw

    read_raw = await tools["read_clinical_archive_resource"].ainvoke({
        "resource_type": "report", "resource_id": report.report_id,
    })
    read = json.loads(read_raw)
    assert read["found"] is True
    assert read["stored_in_database"] is True
    assert "小明" not in read_raw
    assert "OP-PRIVATE-001" not in read_raw
    assert "13812345678" not in read_raw


@pytest.mark.anyio
async def test_document_delete_executes_only_after_owner_confirmation() -> None:
    repository = CaseRepository(init_database("sqlite+pysqlite:///:memory:"))
    runtime = create_v2_runtime(create_service_container(repository))
    document = repository.create_agent_document(
        user_id=7, conversation_id="delete-confirmation", case_id=None,
        title="待删除文档", filename="delete.md", content="内容",
    )
    initial = runtime.harness.create_run(user_id=7, case_id=None, message="删除这份文档")
    action = {
        "action_id": "ACT_DELETE_DOC", "kind": "DELETE_DOCUMENT",
        "title": "删除已存文档", "summary": "确认后删除。",
        "preview": {"document_id": document["document_id"]}, "destructive": True,
    }
    repository.update_agent_run(
        initial.run_id, status="WAITING_CONFIRM", current_node="confirm_agent_action",
        state={"agent_action": {
            "kind": "DELETE_DOCUMENT",
            "payload": {"document_id": document["document_id"]},
        }},
        pending_action=action,
    )
    assert repository.get_agent_document(document["document_id"], 7) is not None
    completed = await runtime.harness.confirm(initial.run_id, 7, "ACT_DELETE_DOC")
    assert completed.status == "COMPLETED"
    assert completed.result["operation"] == "delete_document"
    assert repository.get_agent_document(document["document_id"], 7) is None


@pytest.mark.anyio
async def test_agent_document_can_be_edited_and_downloaded_from_api() -> None:
    repository = CaseRepository(init_database("sqlite+pysqlite:///:memory:"))
    runtime = create_v2_runtime(create_service_container(repository))
    item = repository.create_agent_document(
        user_id=0, conversation_id="conversation-api", case_id=None,
        title="可编辑文档", filename="可编辑文档.md", content="初稿",
    )

    async def override_runtime():
        return runtime

    app.dependency_overrides[get_runtime] = override_runtime
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            updated = await client.put(
                f'/v1/agent/documents/{item["document_id"]}',
                json={"title": "可编辑文档", "content": "医生修改后的第二版"},
            )
            assert updated.status_code == 200, updated.text
            assert updated.json()["revision"] == 2
            downloaded = await client.get(
                f'/v1/agent/documents/{item["document_id"]}/download'
            )
            assert downloaded.status_code == 200
            assert downloaded.text == "医生修改后的第二版"
            assert "filename*=UTF-8''" in downloaded.headers["content-disposition"]
            deleted = await client.delete(f'/v1/agent/documents/{item["document_id"]}')
            assert deleted.status_code == 200
            assert repository.get_agent_document(item["document_id"], 0) is None
    finally:
        app.dependency_overrides.clear()


def test_case_linked_markdown_contains_secure_lesion_image_reference() -> None:
    repository = CaseRepository(init_database("sqlite+pysqlite:///:memory:"))
    item = repository.create_agent_document(
        user_id=7, conversation_id="case-doc", case_id="CASE_IMAGE_1",
        title="患者随访", filename="患者随访.md", content="# 随访记录\n\n病灶稳定。",
    )
    assert "![患者病灶图片](/v1/cases/CASE_IMAGE_1/image)" in item["content"]
    updated = repository.update_agent_document(
        item["document_id"], 7, content="# 第二版\n\n继续观察。"
    )
    assert updated["content"].count("/v1/cases/CASE_IMAGE_1/image") == 1


@pytest.mark.anyio
async def test_doctor_can_upload_text_document_and_file_it() -> None:
    repository = CaseRepository(init_database("sqlite+pysqlite:///:memory:"))
    runtime = create_v2_runtime(create_service_container(repository))
    folder = repository.create_folder(0, "门诊资料")

    async def override_runtime():
        return runtime

    app.dependency_overrides[get_runtime] = override_runtime
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            uploaded = await client.post(
                "/v1/agent/documents/upload",
                files={"file": ("复诊记录.md", "# 复诊\n病灶稳定".encode(), "text/markdown")},
                data={"conversation_id": "workspace", "folder_id": folder["folder_id"]},
            )
            assert uploaded.status_code == 200, uploaded.text
            document_id = uploaded.json()["document_id"]
            listed = await client.get(
                "/v1/agent/documents", params={"folder_id": folder["folder_id"]}
            )
            assert [item["document_id"] for item in listed.json()] == [document_id]
    finally:
        app.dependency_overrides.clear()
