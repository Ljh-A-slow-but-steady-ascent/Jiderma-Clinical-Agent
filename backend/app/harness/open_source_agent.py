from __future__ import annotations

import ast
import json
import operator
import os
import re
from typing import Any, Callable, Literal

from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, ConfigDict, Field

from backend.app.mcp.registry import ToolRegistry
from backend.app.mcp.schemas import ToolContext
from backend.app.services.privacy import EgressGuard, EgressRequest
from backend.app.services.privacy.egress import sanitize_text


AGENT_SYSTEM_PROMPT = """你是霁肤医生工作台的智能 Agent，面向临床工作人员。

你必须结合完整对话上下文理解指代、省略和追问，不得把“它、这个、为什么、继续”当成独立搜索词。
你拥有任务拆解、工具调用、子任务委派和虚拟工作区能力。你应自主完成目标，不要因为任务较长就缩减工作。
对于包含三个及以上步骤、需要多个证据源或需要执行操作的任务，先用 write_todos 建立任务清单，并在执行中持续更新；
可以在有助于并行调查或隔离上下文时使用 task 委派子任务，也可以用虚拟文件记录临时材料。
虚拟文件仅用于本次 Agent 工作，不代表服务器真实文件；不得声称访问了未通过工具提供的宿主机文件。
用户要求创建、保存、导出文档时，必须调用 create_document 生成可打开、可编辑、可下载的交付文档；
用户要求改写已有文档时，先调用 list_documents 或 read_document，再调用 rewrite_document，不要只在聊天正文中贴出修改稿。
用户要求把已有文档归档、移动到文件夹时，先调用 list_documents 确定 document_id，再调用 move_document_to_folder。
用户要求复制或移动报告、文档时，先检索并核对资源，再调用 copy_archive_file 或 move_archive_file；不得用改文件名冒充复制或移动。
不得用创建新文档、修改文件名或在文件名中加斜杠来冒充文件夹移动。workspace 文件夹是数据库中的真实归档结构，不是虚拟工作区路径。
用户要求删除文档时，先检索并读取目标，说明你理解的删除对象和影响，然后调用 propose_delete_document 生成待确认操作。未获得医生确认前不得删除，也不得声称已删除。
病例记录及病例库不得由 Agent 删除。若用户要求删除病例，明确引导其进入病例工作台并手动输入完整病例编号确认。

你自主判断是否需要工具，但必须遵守：
1. 普通问答直接回答，不调用搜索。
2. 用户明确说“联网、搜索、查最新”，或问题明显依赖当前信息时，才调用 web_search。
3. 医学知识与指南问题优先调用 search_guideline，不用通用网络结果替代本地医学证据。只要解释具体疾病、危险征象、检查或转诊依据，必须先调用 search_guideline。
4. 询问当前病例时，先读取相关病例工具，不得臆造候选、分数、分级或照护建议。
4.1 查询历史报告或数据库文档时，使用 search_clinical_archive 和 read_clinical_archive_resource；只能读取当前登录医生有权访问的脱敏内容。
5. 你不能调用多模态推理；只有外层工作流检测到当前消息附图后才能进入该流程。
6. 普通读取、检索、计算和临时规划无需确认；创建、写入、修改或删除病例及报告等关键业务操作，必须调用 propose_* 工具，由外层确认门执行。
7. 不得泄露模型权重、内部路径、密钥或患者身份信息。
8. 用户要求生成报告、修改病例或删除报告时，必须调用对应的 propose_* 工具生成待确认操作；每轮最多提议一个写操作。

请用自然、准确的中文 Markdown 回答，先回答当前问题，不套用固定报告模板。
如调用了 web_search，只引用工具实际返回的链接。
解释疾病、风险、检查或转诊依据时，每个关键结论都必须引用工具返回的来源。引用必须明确标注“本地知识库”、“本地报告库”、“本地文档库”或“外部公开资料”；不得将数据库内文件冒充成医学指南。
"""

TOOL_STEP_LABELS = {
    "write_todos": "拆解并更新任务计划",
    "task": "委派独立子任务",
    "ls": "查看虚拟工作区",
    "read_file": "读取虚拟工作区资料",
    "write_file": "写入虚拟工作区草稿",
    "edit_file": "修改虚拟工作区草稿",
    "glob": "匹配虚拟工作区文件",
    "grep": "检索虚拟工作区内容",
    "create_document": "创建可编辑交付文档",
    "list_documents": "读取当前会话文档",
    "read_document": "读取交付文档内容",
    "rewrite_document": "改写交付文档",
    "list_workspace_folders": "查看真实归档文件夹",
    "move_document_to_folder": "将文档移入真实文件夹",
    "search_clinical_archive": "脱敏检索本地报告和文档库",
    "read_clinical_archive_resource": "读取脱敏的库内文件",
    "propose_delete_document": "生成文档删除预览并等待确认",
    "copy_archive_file": "复制数据库中的报告或文档",
    "move_archive_file": "移动数据库中的报告或文档",
    "propose_delete_archive_file": "生成报告或文档删除预览并等待确认",
    "search_guideline": "检索本地医学指南",
    "web_search": "检索用户指定的公开资料",
    "calculator": "执行本地计算",
    "get_visual_evidence": "读取当前病例的联合分析证据",
    "get_triage_result": "读取当前病例的风险分级",
    "get_care_pathway": "读取当前病例的照护路径",
    "get_privacy_report": "核验当前病例的隐私状态",
    "list_case_reports": "核对当前病例已有报告",
    "propose_case_update": "生成病例修改预览",
    "propose_generate_report": "生成报告写入预览",
    "propose_delete_report": "生成报告删除预览",
}


class QueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500)


class CalculatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(min_length=1, max_length=120)


class CaseUpdateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chief_complaint: str = Field(default="", max_length=5000)
    lesion_site: str = Field(default="", max_length=120)


class ReportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    report_id: str = Field(min_length=1, max_length=80)


class DocumentCreateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=160)
    filename: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1, max_length=60000)


class DocumentIdInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(min_length=1, max_length=80)


class DocumentRewriteInput(DocumentIdInput):
    title: str = Field(default="", max_length=160)
    content: str = Field(min_length=1, max_length=60000)


class DocumentMoveInput(DocumentIdInput):
    folder_name: str = Field(min_length=1, max_length=120)


class ArchiveSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(default="", max_length=500)
    limit: int = Field(default=8, ge=1, le=20)


class ArchiveReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resource_type: Literal["report", "document"]
    resource_id: str = Field(min_length=1, max_length=100)


class ArchiveFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resource_type: Literal["report", "document"]
    resource_id: str = Field(min_length=1, max_length=100)


class ArchiveFileMoveInput(ArchiveFileInput):
    folder_name: str = Field(min_length=1, max_length=120)


class ArchiveFileCopyInput(ArchiveFileInput):
    folder_name: str = Field(default="", max_length=120)


def _base_url(value: str, default: str) -> str:
    url = (value or default).rstrip("/")
    suffix = "/chat/completions"
    return url[:-len(suffix)] if url.endswith(suffix) else url


def _safe_calculate(expression: str) -> float:
    operations = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operations:
            return operations[type(node.op)](evaluate(node.operand))
        raise ValueError("仅支持基本算术表达式")

    value = evaluate(ast.parse(expression, mode="eval"))
    if abs(value) > 1e100:
        raise ValueError("计算结果超出允许范围")
    return value


class OpenSourceAgentRuntime:
    """Deep Agents runtime around the project's audited clinical tools."""

    def __init__(self, registry: ToolRegistry, guard: EgressGuard) -> None:
        self.registry = registry
        self.guard = guard
        self.checkpointer = InMemorySaver()

    @staticmethod
    def available(model_id: str) -> bool:
        if model_id in {"deepseek-flash", "deepseek-v4-pro"}:
            return bool(os.getenv("DEEPSEEK_API_KEY"))
        if model_id in {"glm-5.3-flash", "glm-4-flash"}:
            return bool(os.getenv("ZHIPUAI_API_KEY"))
        if model_id in {"openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"}:
            return bool(os.getenv("GROQ_API_KEY"))
        if model_id == "gemini-flash":
            return False
        provider = os.getenv("DERMAGUARD_CHAT_PROVIDER", "mock").strip().lower()
        return (provider == "deepseek" and bool(os.getenv("DEEPSEEK_API_KEY"))) or (provider == "groq" and bool(os.getenv("GROQ_API_KEY"))) or (
            provider in {"zhipu", "zhipu_glm", "glm"} and bool(os.getenv("ZHIPUAI_API_KEY"))
        )

    @staticmethod
    def _model(model_id: str) -> ChatOpenAI:
        selected = model_id
        if selected == "auto":
            provider = os.getenv("DERMAGUARD_CHAT_PROVIDER", "mock").strip().lower()
            if provider == "deepseek":
                selected = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
            elif provider == "groq":
                selected = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
            else:
                selected = os.getenv("ZHIPUAI_MODEL", "glm-5.3-flash")
        if selected in {"deepseek-flash", "deepseek-v4-pro"}:
            return ChatOpenAI(
                model=selected, api_key=os.environ["DEEPSEEK_API_KEY"], temperature=0.25,
                base_url=_base_url(os.getenv("DEEPSEEK_BASE_URL", ""), "https://api.deepseek.com"),
                timeout=90, max_retries=2,
                extra_body={"thinking": {"type": "disabled"}},
            )
        if selected in {"glm-5.3-flash", "glm-4-flash"} or selected.startswith("glm-"):
            return ChatOpenAI(
                model=selected, api_key=os.environ["ZHIPUAI_API_KEY"], temperature=0.25,
                base_url=_base_url(os.getenv("ZHIPUAI_BASE_URL", ""), "https://open.bigmodel.cn/api/paas/v4"),
                timeout=60, max_retries=2,
            )
        return ChatOpenAI(
            model=selected, api_key=os.environ["GROQ_API_KEY"], temperature=0.25,
            base_url=_base_url(os.getenv("GROQ_BASE_URL", ""), "https://api.groq.com/openai/v1"),
            timeout=60, max_retries=2,
        )

    def _tools(
        self, user_id: int, conversation_id: str, case_id: str | None,
        on_event: Callable[[str, str], None],
        proposals: list[dict[str, Any]],
        alias_to_identity: dict[str, str] | None = None,
    ) -> list[StructuredTool]:
        aliases = alias_to_identity or {}

        def restore_local_query(value: str) -> str:
            result = value
            for alias, identity in aliases.items():
                result = result.replace(alias, identity)
            return result

        def mask_archive_text(value: Any) -> str:
            result = str(value or "")
            for alias, identity in sorted(aliases.items(), key=lambda item: len(item[1]), reverse=True):
                if identity:
                    result = result.replace(identity, alias)
            return sanitize_text(result)[0]
        def safe_tool_output(name: str, payload: dict[str, Any]) -> str:
            checked = self.guard.enforce(EgressRequest(
                channel="llm", destination="agent_tool_result", payload=payload,
                case_id=case_id, tool_name=name,
            ))
            return json.dumps(checked.sanitized_payload, ensure_ascii=False)

        async def invoke(name: str, arguments: dict[str, Any]) -> str:
            on_event(name, f"正在调用 {name}")
            call = await self.registry.invoke(
                name, arguments,
                ToolContext(case_id=case_id, channel="mcp", destination="langchain_agent"),
            )
            on_event(name, f"{name} 已完成")
            return safe_tool_output(name, call.result)

        async def search_guideline(query: str) -> str:
            """检索本地权威医学知识库。医学定义、鉴别、检查或转诊问题优先使用。"""
            return await invoke("search_guideline", {"query": query, "case_id": case_id, "top_k": 5})

        async def web_search(query: str) -> str:
            """搜索公开互联网资料。仅在用户明确要求联网/搜索，或问题必须依赖当前信息时使用。"""
            checked = self.guard.enforce(EgressRequest(
                channel="http", destination="zh.wikipedia.org", payload={"query": query},
                case_id=case_id, tool_name="web_search",
            ))
            return await invoke("web_search", {"query": checked.sanitized_payload["query"], "top_k": 3})

        async def calculator(expression: str) -> str:
            """执行可审计的基本算术计算。"""
            on_event("calculator", "正在执行本地计算")
            return str(_safe_calculate(expression))

        def clean_filename(value: str) -> str:
            filename = re.sub(r"[^\w\-.\u4e00-\u9fff]+", "_", value.strip()).strip("._")
            if not filename:
                filename = "agent-document.md"
            if "." not in filename:
                filename += ".md"
            return filename[:160]

        async def create_document(title: str, filename: str, content: str) -> str:
            """Create a persistent, editable and downloadable user document for this conversation."""
            item = self.registry.repository.create_agent_document(
                user_id=user_id, conversation_id=conversation_id, case_id=case_id,
                title=title.strip(), filename=clean_filename(filename), content=content,
            )
            on_event("create_document", f"已创建可编辑文档：{item['filename']}")
            return json.dumps({
                "document_id": item["document_id"], "title": item["title"],
                "filename": item["filename"], "revision": item["revision"],
                "editable": True, "downloadable": True,
            }, ensure_ascii=False)

        async def list_documents() -> str:
            """List editable documents already created in the current conversation."""
            items = self.registry.repository.list_agent_documents(
                user_id=user_id, conversation_id=conversation_id
            )
            on_event("list_documents", f"已读取当前会话的 {len(items)} 个文档")
            return json.dumps({"documents": [
                {key: item[key] for key in ("document_id", "title", "filename", "revision", "updated_at")}
                for item in items
            ]}, ensure_ascii=False)

        async def read_document(document_id: str) -> str:
            """Read one editable document before revising or discussing it."""
            item = self.registry.repository.get_agent_document(document_id, user_id)
            if item is None or item["conversation_id"] != conversation_id:
                return json.dumps({"found": False, "reason": "当前会话未找到该文档"}, ensure_ascii=False)
            on_event("read_document", f"已读取文档：{item['filename']}")
            return safe_tool_output("read_document", item)

        async def rewrite_document(document_id: str, content: str, title: str = "") -> str:
            """Rewrite an existing conversation document and create a new revision."""
            existing = self.registry.repository.get_agent_document(document_id, user_id)
            if existing is None or existing["conversation_id"] != conversation_id:
                return json.dumps({"updated": False, "reason": "当前会话未找到该文档"}, ensure_ascii=False)
            item = self.registry.repository.update_agent_document(
                document_id, user_id, content=content, title=title.strip() or None
            )
            on_event("rewrite_document", f"已改写文档：{item['filename']}（第 {item['revision']} 版）")
            return json.dumps({
                "updated": True, "document_id": item["document_id"],
                "filename": item["filename"], "revision": item["revision"],
            }, ensure_ascii=False)

        async def list_workspace_folders() -> str:
            """List real database-backed folders visible in the clinician workspace."""
            folders = self.registry.repository.list_folders(user_id)
            on_event("list_workspace_folders", f"已读取 {len(folders)} 个归档文件夹")
            return json.dumps({"folders": folders}, ensure_ascii=False)

        async def move_document_to_folder(document_id: str, folder_name: str) -> str:
            """Move an existing document into a real folder, creating the folder when absent."""
            document = self.registry.repository.get_agent_document(document_id, user_id)
            if document is None:
                return json.dumps({"moved": False, "reason": "未找到该文档"}, ensure_ascii=False)
            normalized = folder_name.strip().strip("/")
            folders = self.registry.repository.list_folders(user_id)
            folder = next((item for item in folders if item["name"] == normalized), None)
            created = False
            if folder is None:
                folder = self.registry.repository.create_folder(user_id, normalized)
                created = True
            self.registry.repository.move_workspace_item(
                user_id, "document", document_id, folder["folder_id"]
            )
            on_event(
                "move_document_to_folder",
                f"已将 {document['filename']} 移入文件夹“{folder['name']}”",
            )
            return json.dumps({
                "moved": True,
                "document_id": document_id,
                "filename": document["filename"],
                "folder_id": folder["folder_id"],
                "folder_name": folder["name"],
                "folder_created": created,
            }, ensure_ascii=False)

        async def search_clinical_archive(query: str = "", limit: int = 8) -> str:
            """Search clinician-owned reports and documents; results are pseudonymized before egress."""
            local_query = restore_local_query(query)
            items = self.registry.repository.search_clinical_archive(
                user_id, local_query, limit=limit
            )
            safe_items = []
            for item in items:
                resource_type = str(item["resource_type"])
                safe_items.append({
                    "resource_type": resource_type,
                    "resource_id": item["resource_id"],
                    "case_id": item.get("case_id"),
                    "title": mask_archive_text(item.get("title") or item.get("filename")),
                    "snippet": mask_archive_text(item.get("content"))[:900],
                    "updated_at": item.get("updated_at"),
                    "stored_in_database": True,
                    "source_scope": "本地报告库" if resource_type == "report" else "本地文档库",
                })
            on_event("search_clinical_archive", f"已在本地临床资料库检索到 {len(safe_items)} 项结果")
            return safe_tool_output("search_clinical_archive", {"results": safe_items})

        async def read_clinical_archive_resource(resource_type: str, resource_id: str) -> str:
            """Read one authorized archive resource without exposing identity or filesystem data."""
            item = self.registry.repository.read_clinical_archive_resource(
                user_id, resource_type, resource_id
            )
            if item is None:
                return json.dumps({"found": False, "reason": "资料不存在或当前账号无权访问"}, ensure_ascii=False)
            source_scope = "本地报告库" if resource_type == "report" else "本地文档库"
            payload = {
                "found": True,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "case_id": item.get("case_id"),
                "title": mask_archive_text(item.get("title") or item.get("filename")),
                "content": mask_archive_text(item.get("content"))[:12000],
                "revision": item.get("revision"),
                "updated_at": item.get("updated_at"),
                "stored_in_database": True,
                "source_scope": source_scope,
            }
            on_event("read_clinical_archive_resource", f"已读取脱敏的{source_scope}资料")
            return safe_tool_output("read_clinical_archive_resource", payload)

        async def propose_delete_document(document_id: str) -> str:
            """Prepare deletion of one owned document; execution requires clinician confirmation."""
            document = self.registry.repository.get_agent_document(document_id, user_id)
            if document is None:
                return json.dumps({"accepted": False, "reason": "文档不存在或当前账号无权访问"}, ensure_ascii=False)
            payload = {
                "document_id": document_id,
                "title": mask_archive_text(document.get("title")),
                "filename": mask_archive_text(document.get("filename")),
                "case_id": document.get("case_id"),
            }
            if not proposals:
                proposals.append({"kind": "DELETE_DOCUMENT", "payload": payload})
            on_event("propose_delete_document", "已理解删除目标，正在等待医生确认")
            return json.dumps({
                "accepted": True, "requires_confirmation": True,
                "destructive": True, "preview": payload,
            }, ensure_ascii=False)

        def archive_resource(resource_type: str, resource_id: str) -> dict[str, Any] | None:
            return self.registry.repository.read_clinical_archive_resource(
                user_id, resource_type, resource_id
            )

        def resolve_folder(folder_name: str) -> tuple[str | None, str]:
            normalized = folder_name.strip().strip("/")
            if not normalized:
                return None, "未归档"
            folders = self.registry.repository.list_folders(user_id)
            folder = next((item for item in folders if item["name"] == normalized), None)
            if folder is None:
                folder = self.registry.repository.create_folder(user_id, normalized)
            return folder["folder_id"], folder["name"]

        async def copy_archive_file(
            resource_type: str, resource_id: str, folder_name: str = "",
        ) -> str:
            """Copy an owned report or document and optionally place the copy in a folder."""
            source = archive_resource(resource_type, resource_id)
            if source is None:
                return json.dumps({"copied": False, "reason": "文件不存在或当前账号无权访问"}, ensure_ascii=False)
            if resource_type == "report":
                copied = self.registry.repository.copy_report(resource_id, user_id)
                copied_id = copied["report_id"]
            else:
                copied = self.registry.repository.copy_agent_document(
                    resource_id, user_id, conversation_id=conversation_id
                )
                copied_id = copied["document_id"]
            folder_id, resolved_name = resolve_folder(folder_name)
            if not folder_name.strip():
                folder_id = self.registry.repository.workspace_locations(
                    user_id, resource_type
                ).get(resource_id)
                resolved_name = next(
                    (item["name"] for item in self.registry.repository.list_folders(user_id)
                     if item["folder_id"] == folder_id), "未归档"
                )
            if folder_id:
                self.registry.repository.move_workspace_item(
                    user_id, resource_type, copied_id, folder_id
                )
            on_event("copy_archive_file", f"已复制{resource_type == 'report' and '报告' or '文档'}并归档到“{resolved_name}”")
            return json.dumps({
                "copied": True, "resource_type": resource_type,
                "source_id": resource_id, "copied_id": copied_id,
                "folder_id": folder_id, "folder_name": resolved_name,
            }, ensure_ascii=False)

        async def move_archive_file(
            resource_type: str, resource_id: str, folder_name: str,
        ) -> str:
            """Move an owned report or document into a real database-backed folder."""
            if archive_resource(resource_type, resource_id) is None:
                return json.dumps({"moved": False, "reason": "文件不存在或当前账号无权访问"}, ensure_ascii=False)
            folder_id, resolved_name = resolve_folder(folder_name)
            self.registry.repository.move_workspace_item(
                user_id, resource_type, resource_id, folder_id
            )
            on_event("move_archive_file", f"已将文件移入“{resolved_name}”")
            return json.dumps({
                "moved": True, "resource_type": resource_type,
                "resource_id": resource_id, "folder_id": folder_id,
                "folder_name": resolved_name,
            }, ensure_ascii=False)

        async def propose_delete_archive_file(resource_type: str, resource_id: str) -> str:
            """Prepare deletion of an owned report or document; execution requires confirmation."""
            resource = archive_resource(resource_type, resource_id)
            if resource is None:
                return json.dumps({"accepted": False, "reason": "文件不存在或当前账号无权访问"}, ensure_ascii=False)
            kind = "DELETE_REPORT" if resource_type == "report" else "DELETE_DOCUMENT"
            payload = {
                f"{resource_type}_id": resource_id,
                "title": mask_archive_text(resource.get("title") or resource.get("filename")),
                "case_id": resource.get("case_id"),
            }
            if not proposals:
                proposals.append({"kind": kind, "payload": payload})
            on_event("propose_delete_archive_file", "已核对文件和影响范围，等待医生确认删除")
            return json.dumps({
                "accepted": True, "requires_confirmation": True,
                "destructive": True, "preview": payload,
            }, ensure_ascii=False)


        tools = [
            StructuredTool.from_function(coroutine=search_guideline, name="search_guideline", args_schema=QueryInput),
            StructuredTool.from_function(coroutine=web_search, name="web_search", args_schema=QueryInput),
            StructuredTool.from_function(coroutine=calculator, name="calculator", args_schema=CalculatorInput),
            StructuredTool.from_function(coroutine=create_document, name="create_document", args_schema=DocumentCreateInput),
            StructuredTool.from_function(coroutine=list_documents, name="list_documents"),
            StructuredTool.from_function(coroutine=read_document, name="read_document", args_schema=DocumentIdInput),
            StructuredTool.from_function(coroutine=rewrite_document, name="rewrite_document", args_schema=DocumentRewriteInput),
            StructuredTool.from_function(coroutine=list_workspace_folders, name="list_workspace_folders"),
            StructuredTool.from_function(coroutine=move_document_to_folder, name="move_document_to_folder", args_schema=DocumentMoveInput),
            StructuredTool.from_function(coroutine=search_clinical_archive, name="search_clinical_archive", args_schema=ArchiveSearchInput),
            StructuredTool.from_function(coroutine=read_clinical_archive_resource, name="read_clinical_archive_resource", args_schema=ArchiveReadInput),
            StructuredTool.from_function(coroutine=propose_delete_document, name="propose_delete_document", args_schema=DocumentIdInput),
            StructuredTool.from_function(coroutine=copy_archive_file, name="copy_archive_file", args_schema=ArchiveFileCopyInput),
            StructuredTool.from_function(coroutine=move_archive_file, name="move_archive_file", args_schema=ArchiveFileMoveInput),
            StructuredTool.from_function(coroutine=propose_delete_archive_file, name="propose_delete_archive_file", args_schema=ArchiveFileInput),
        ]
        if case_id:
            def make_case_tool(tool_name: str):
                async def case_tool() -> str:
                    return await invoke(tool_name, {"case_id": case_id})
                return case_tool

            for name, description in (
                ("get_visual_evidence", "读取当前病例已保存的联合分析候选证据。"),
                ("get_triage_result", "读取当前病例的确定性风险分级。"),
                ("get_care_pathway", "读取当前病例的照护与转诊路径。"),
                ("get_privacy_report", "读取当前病例的隐私与出境审计事实。"),
            ):
                tools.append(StructuredTool.from_function(
                    coroutine=make_case_tool(name), name=name, description=description,
                ))

            async def list_case_reports() -> str:
                """List reports already saved for the current case."""
                reports = [
                    {
                        "report_id": item.get("report_id"),
                        "status": item.get("status"),
                        "updated_at": item.get("updated_at"),
                        "model": item.get("model"),
                    }
                    for item in self.registry.repository.list_reports(limit=200, user_id=user_id)
                    if item.get("case_id") == case_id
                ] if self.registry.repository is not None else []
                on_event("list_case_reports", f"已读取 {len(reports)} 份病例报告")
                return safe_tool_output("list_case_reports", {"reports": reports})

            async def propose_case_update(
                chief_complaint: str = "", lesion_site: str = ""
            ) -> str:
                """Propose changing the current case narrative or lesion site; requires clinician approval."""
                patch = {
                    key: value.strip()
                    for key, value in {
                        "chief_complaint": chief_complaint,
                        "lesion_site": lesion_site,
                    }.items()
                    if value.strip()
                }
                if not patch:
                    return json.dumps({"accepted": False, "reason": "未提供可修改字段"}, ensure_ascii=False)
                if not proposals:
                    proposals.append({"kind": "UPDATE_CASE", "payload": patch})
                on_event("propose_case_update", "已生成病例修改预览，等待医生确认")
                return json.dumps({"accepted": True, "requires_confirmation": True, "preview": patch}, ensure_ascii=False)

            async def propose_generate_report() -> str:
                """Propose generating and saving a report for the current case; requires clinician approval."""
                if not proposals:
                    proposals.append({"kind": "GENERATE_REPORT", "payload": {}})
                on_event("propose_generate_report", "已生成报告生成任务，等待医生确认")
                return json.dumps({"accepted": True, "requires_confirmation": True}, ensure_ascii=False)

            async def propose_delete_report(report_id: str) -> str:
                """Propose deleting a saved report by id; requires clinician approval."""
                report = self.registry.repository.get_report(report_id) if self.registry.repository else None
                if report is None or report.get("case_id") != case_id:
                    return json.dumps({"accepted": False, "reason": "当前病例中未找到该报告"}, ensure_ascii=False)
                if not proposals:
                    proposals.append({
                        "kind": "DELETE_REPORT",
                        "payload": {"report_id": report_id, "filename": report.get("filename", "")},
                    })
                on_event("propose_delete_report", "已生成报告删除预览，等待医生确认")
                return json.dumps({
                    "accepted": True, "requires_confirmation": True,
                    "report_id": report_id,
                }, ensure_ascii=False)

            tools.extend([
                StructuredTool.from_function(
                    coroutine=list_case_reports, name="list_case_reports",
                    description="读取当前病例已保存的报告列表。",
                ),
                StructuredTool.from_function(
                    coroutine=propose_case_update, name="propose_case_update",
                    args_schema=CaseUpdateInput,
                ),
                StructuredTool.from_function(
                    coroutine=propose_generate_report, name="propose_generate_report",
                    description="为当前病例提议生成并保存报告，操作需医生确认。",
                ),
                StructuredTool.from_function(
                    coroutine=propose_delete_report, name="propose_delete_report",
                    args_schema=ReportInput,
                ),
            ])
        return tools

    async def run(
        self, *, message: str, model_id: str, conversation_id: str, user_id: int,
        case_id: str | None, history: list[dict[str, Any]],
        on_event: Callable[[str, str], None],
    ) -> dict[str, Any]:
        model = self._model(model_id)
        proposals: list[dict[str, Any]] = []
        actual_to_alias: dict[str, str] = {}
        if self.registry.repository is not None:
            for index, patient in enumerate(
                self.registry.repository.patient_identifiers_for_user(user_id), start=1
            ):
                for key, label in (("name", "患者"), ("outpatient_no", "门诊号"), ("phone", "电话")):
                    value = str(patient.get(key) or "").strip()
                    if value:
                        actual_to_alias.setdefault(value, f"[{label}_{index:02d}]")
        alias_to_identity = {alias: value for value, alias in actual_to_alias.items()}
        tools = self._tools(
            user_id, conversation_id, case_id, on_event, proposals,
            alias_to_identity=alias_to_identity,
        )
        on_event("agent_context", "正在结合近期对话与当前病例识别任务目标")
        agent = create_deep_agent(
            model=model, tools=tools, system_prompt=AGENT_SYSTEM_PROMPT,
            checkpointer=self.checkpointer, name="jiderma_deep_agent",
            middleware=[TodoListMiddleware()],
        )
        config = {"configurable": {"thread_id": conversation_id}}
        snapshot = await agent.aget_state(config)
        def mask_identity(value: str) -> str:
            result = value
            for identity, alias in sorted(
                actual_to_alias.items(), key=lambda item: len(item[0]), reverse=True
            ):
                result = result.replace(identity, alias)
            return result

        safe_history = [
            {"role": item.get("role"), "message": mask_identity(str(item.get("message", "")))}
            for item in history[-12:]
            if item.get("role") in {"user", "assistant"} and item.get("message")
        ]
        checked = self.guard.enforce(EgressRequest(
            channel="llm", destination=model_id,
            payload={"message": mask_identity(message), "history": safe_history}, case_id=case_id,
        ))
        safe_message = str(checked.sanitized_payload["message"])
        safe_history = list(checked.sanitized_payload.get("history", []))
        messages: list[Any] = []
        if not snapshot.values.get("messages"):
            messages.extend(
                {"role": item.get("role", "user"), "content": str(item.get("message", ""))}
                for item in safe_history[-12:]
                if item.get("role") in {"user", "assistant"} and item.get("message")
            )
        messages.append(HumanMessage(content=safe_message))
        latest: AIMessage | None = None
        tool_calls: list[dict[str, Any]] = []
        seen_tool_call_ids: set[str] = set()
        task_plan: list[dict[str, str]] = []
        allowed_urls: dict[str, str] = {}
        evidence_sources: dict[str, str] = {}
        async for update in agent.astream(
            {"messages": messages}, config=config, stream_mode="updates"
        ):
            for node, payload in update.items():
                node_messages = payload.get("messages", []) if isinstance(payload, dict) else []
                for item in node_messages:
                    if isinstance(item, AIMessage):
                        latest = item
                        if item.tool_calls:
                            names = [call.get("name", "tool") for call in item.tool_calls]
                            for call in item.tool_calls:
                                if call.get("name") != "write_todos":
                                    continue
                                todos = (call.get("args") or {}).get("todos", [])
                                if isinstance(todos, list):
                                    task_plan = [
                                        {
                                            "content": str(todo.get("content", "")),
                                            "status": str(todo.get("status", "pending")),
                                        }
                                        for todo in todos if isinstance(todo, dict) and todo.get("content")
                                    ]
                                    if task_plan:
                                        on_event(
                                            "task_plan",
                                            "Agent 已更新任务计划：" + "；".join(
                                                f"{index}. {todo['content']}"
                                                for index, todo in enumerate(task_plan, start=1)
                                            ),
                                        )
                            on_event(
                                "agent_planning",
                                "接下来执行：" + " → ".join(
                                    f"{index}. {TOOL_STEP_LABELS.get(name, name)}"
                                    for index, name in enumerate(names, start=1)
                                ),
                            )
                    elif isinstance(item, ToolMessage):
                        call_id = str(getattr(item, "tool_call_id", "") or f"{node}:{len(tool_calls)}")
                        if call_id in seen_tool_call_ids:
                            continue
                        seen_tool_call_ids.add(call_id)
                        tool_calls.append({"tool_name": getattr(item, "name", "tool"), "status": "success", "latency_ms": 0})
                        raw_content = str(getattr(item, "content", ""))
                        try:
                            parsed = json.loads(raw_content)
                        except (TypeError, json.JSONDecodeError):
                            parsed = {}
                        for chunk in parsed.get("chunks", []) if isinstance(parsed, dict) else []:
                            citation = chunk.get("citation") or {}
                            evidence_id = str(chunk.get("id") or chunk.get("document_id") or "未标注")
                            evidence_sources[f"knowledge:{evidence_id}"] = (
                                f"[本地知识库（已存库）] {chunk.get('title', '未命名资料')} · "
                                f"{citation.get('organization', chunk.get('source', '未知机构'))} · "
                                f"{citation.get('section', '未标注章节')} · 证据 {evidence_id}"
                            )
                        for result in parsed.get("results", []) if isinstance(parsed, dict) else []:
                            url = str(result.get("url", ""))
                            if url.startswith(("http://", "https://")):
                                allowed_urls[url] = str(result.get("title") or url)
                            if result.get("stored_in_database"):
                                resource_id = str(result.get("resource_id") or "未标注")
                                evidence_sources[f"archive:{resource_id}"] = (
                                    f"[{result.get('source_scope', '本地资料库')}（已存库）] "
                                    f"{result.get('title', '未命名资料')} · 资源 {resource_id}"
                                )
                        if isinstance(parsed, dict) and parsed.get("stored_in_database"):
                            resource_id = str(parsed.get("resource_id") or "未标注")
                            evidence_sources[f"archive:{resource_id}"] = (
                                f"[{parsed.get('source_scope', '本地资料库')}（已存库）] "
                                f"{parsed.get('title', '未命名资料')} · 资源 {resource_id}"
                            )
                        for url in re.findall(r"https?://[^\s\]\)\"']+", raw_content):
                            allowed_urls.setdefault(url.rstrip(".,，。"), url.rstrip(".,，。"))
        if latest is None or latest.tool_calls:
            state = await agent.aget_state(config)
            latest = next((item for item in reversed(state.values.get("messages", [])) if isinstance(item, AIMessage) and not item.tool_calls), None)
        if latest is None:
            raise RuntimeError("开源 Agent Runtime 未返回最终答案")
        if not tool_calls:
            on_event("agent_planning", "当前问题不需要外部工具，我正在结合上下文直接组织回答")
        on_event("agent_synthesis", "正在汇总已验证信息并生成最终结果")
        content = latest.content
        answer = content if isinstance(content, str) else "".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict)
        )
        answer = re.sub(
            r"\[([^\]]+)\]\((https?://[^)]+)\)",
            lambda match: match.group(0) if match.group(2) in allowed_urls else match.group(1),
            answer,
        )
        answer = re.sub(
            r"https?://[^\s\]\)\"']+",
            lambda match: match.group(0)
            if match.group(0).rstrip(".,，。") in allowed_urls
            else "[未验证链接已移除]",
            answer,
        )
        if allowed_urls:
            sources = "\n".join(
                f"- [{title}]({url})"
                for url, title in list(allowed_urls.items())[:6]
            )
            answer = answer.rstrip() + "\n\n**外部公开资料（未存入本地知识库）**\n" + sources
        if evidence_sources:
            sources = "\n".join(f"- {label}" for label in list(evidence_sources.values())[:8])
            answer = answer.rstrip() + "\n\n**本次引用与存库状态**\n" + sources
        return {
            "response": answer.strip(), "tool_calls": tool_calls,
            "provider": "deepagents", "model": model.model_name,
            "task_plan": task_plan,
            "artifacts": self.registry.repository.list_agent_documents(
                user_id=user_id, conversation_id=conversation_id
            ),
            "agent_trace": [{
                "step": 1, "intent": "deep_agent_runtime",
                "retrieval_query": None, "evidence_sufficient": True,
                "decision": "MODEL_TOOL_LOOP_COMPLETED",
                "observation_summary": [f"完成 {len(tool_calls)} 次工具调用"],
                "reasoning_summary": "由 Deep Agents 完成任务规划、上下文理解、工具选择与结果整合。",
                "provider_used": "deepagents", "fallback_triggered": False,
            }],
            "proposed_action": proposals[0] if proposals else None,
        }
