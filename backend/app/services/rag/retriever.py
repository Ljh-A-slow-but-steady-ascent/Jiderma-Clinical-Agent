from __future__ import annotations

import os
import re
import select
import subprocess
import threading
from pathlib import Path
from typing import Any

from backend.app.config import AppConfig, get_config


ROOT = Path(__file__).resolve().parents[4]
DOCUMENT_DIR = Path(__file__).resolve().parents[2] / "knowledge" / "rag_documents"
RAG_WORKER = Path(__file__).resolve().parent / "model_worker.py"
RAG_PYTHON = Path(os.getenv("DERMAGUARD_RAG_PYTHON", "/home/lu/miniforge3/envs/dermfm-zero/bin/python"))

QUERY_EXPANSIONS = {
    "bcc": "basal cell carcinoma 基底细胞癌", "基底细胞癌": "basal cell carcinoma bcc",
    "scc": "squamous cell carcinoma 鳞状细胞癌", "鳞状细胞癌": "squamous cell carcinoma scc",
    "黑色素瘤": "melanoma malignant melanoma", "日光性角化病": "actinic keratosis ak",
    "色素痣": "nevus melanocytic nevus mole", "脂溢性角化病": "seborrheic keratosis benign keratosis",
    "良性角化": "benign keratosis seborrheic keratosis solar lentigo lichenoid keratosis",
    "皮肤纤维瘤": "dermatofibroma benign fibrous lesion",
    "血管性病变": "vascular lesion angioma angiokeratoma pyogenic granuloma hemorrhage",
    "带状疱疹": "herpes zoster shingles", "体癣": "tinea corporis ringworm",
    "足癣": "tinea pedis athlete foot", "疥疮": "scabies",
    "特应性皮炎": "atopic dermatitis eczema",
}


def _tokens(text: str) -> list[str]:
    lowered = text.lower().replace("_", " ")
    result = re.findall(r"[a-z0-9]+", lowered)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", lowered):
        if len(sequence) <= 4:
            result.append(sequence)
        result.extend(sequence[index:index + 2] for index in range(len(sequence) - 1))
        result.extend(sequence[index:index + 3] for index in range(len(sequence) - 2))
    return result


def _expand_query(text: str) -> str:
    lowered = text.lower()
    return " ".join([text, *(value for key, value in QUERY_EXPANSIONS.items() if key in lowered)])


def build_dynamic_query(
    user_query: str = "", visual_candidates: list[str] | None = None,
    positive_symptoms: list[str] | None = None, risk_reasons: list[str] | None = None,
    duration: str | None = None, topic: str = "鉴别依据 检查依据 风险分级 转诊指南",
) -> str:
    sections = [
        user_query.strip(), "视觉候选 " + " ".join(visual_candidates or []),
        "阳性症状 " + " ".join(positive_symptoms or []),
        "风险原因 " + " ".join(risk_reasons or []),
        f"病程变化 {duration}" if duration else "", f"检索主题 {topic}" if topic else "",
    ]
    return "；".join(item for item in sections if item and item.strip())


def _parse_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text.strip()
    _, metadata_block, content = text.split("---", 2)
    metadata: dict[str, Any] = {}
    for line in metadata_block.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    for key in ("symptoms", "conditions", "aliases", "tags"):
        metadata[key] = [item.strip() for item in str(metadata.get(key, "")).split(",") if item.strip()]
    return metadata, content.strip()


def _structured_chunks(path: Path) -> list[dict[str, Any]]:
    metadata, content = _parse_frontmatter(path)
    title = str(metadata.get("title", path.stem))
    current_heading = str(metadata.get("section", title))
    blocks: list[tuple[str, str]] = []
    buffer: list[str] = []
    for line in content.splitlines():
        heading = re.match(r"^#{1,4}\s+(.+)$", line.strip())
        if heading:
            if buffer:
                blocks.append((current_heading, "\n".join(buffer).strip())); buffer = []
            current_heading = heading.group(1).strip()
        elif not line.strip():
            if buffer:
                blocks.append((current_heading, "\n".join(buffer).strip())); buffer = []
        else:
            buffer.append(line)
    if buffer:
        blocks.append((current_heading, "\n".join(buffer).strip()))
    blocks = [(heading, text) for heading, text in blocks if text] or [(current_heading, content)]
    chunks = []
    for index, (heading, text) in enumerate(blocks, start=1):
        document_id = str(metadata.get("document_id", path.stem))
        chunks.append({
            **metadata, "id": document_id if len(blocks) == 1 else f"{document_id}#C{index:02d}", "document_id": document_id,
            "title": title, "document_name": str(metadata.get("document_name", title)),
            "organization": str(metadata.get("organization", metadata.get("source", "未知"))),
            "year": str(metadata.get("year", "未标注")), "section": heading,
            "page": str(metadata.get("page", "网页资料")), "version": str(metadata.get("version", "未标注")),
            "content": text,
        })
    return chunks


def _lexical_fallback(query: str, documents: list[dict[str, Any]], corpus_tokens: list[list[str]]) -> list[float]:
    """Dependency-free fallback used only when the local embedding worker is unavailable."""
    query_tokens = set(_tokens(query))
    lowered = query.lower()
    scores: list[float] = []
    for doc, tokens in zip(documents, corpus_tokens, strict=True):
        overlap = len(query_tokens.intersection(tokens))
        phrases = [str(item).lower() for item in [*doc.get("conditions", []), *doc.get("aliases", [])]]
        exact = sum(1 for phrase in phrases if len(phrase) >= 3 and phrase in lowered)
        scores.append(float(overlap + 4 * exact))
    maximum = max(scores, default=0.0)
    return [score / maximum if maximum else 0.0 for score in scores]


def _evidence_category(text: str) -> str:
    lowered = text.lower()
    groups = [
        ("冲突证据", ("冲突", "不能替代", "不支持", "局限")),
        ("转诊依据", ("转诊", "急诊", "优先", "就医")),
        ("检查依据", ("活检", "检查", "病理", "评估")),
        ("鉴别依据", ("鉴别", "边界", "颜色", "表现")),
        ("风险依据", ("风险", "恶性", "红旗")),
    ]
    return next((name for name, terms in groups if any(term in lowered for term in terms)), "信息缺口")


class RagModelClient:
    """Persistent medical embedding worker with a local lexical availability fallback."""

    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()

    def _start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.process = subprocess.Popen(
            [str(RAG_PYTHON), str(RAG_WORKER)], cwd=str(ROOT), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        import json
        with self.lock:
            self._start()
            assert self.process and self.process.stdin and self.process.stdout
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n"); self.process.stdin.flush()
            ready, _, _ = select.select([self.process.stdout], [], [], 60.0)
            if not ready:
                self.process.kill()
                self.process = None
                raise TimeoutError("本地医学嵌入模型响应超时")
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("本地RAG模型进程未返回结果")
            result = json.loads(line)
            if result.get("error"):
                raise RuntimeError(str(result["error"]))
            return result


class LocalRagRetriever:
    """Medical dense retrieval with a deterministic lexical availability fallback."""

    def __init__(self, config: AppConfig | None = None, document_dir: Path = DOCUMENT_DIR) -> None:
        self.config = config or get_config()
        self.documents = [chunk for path in sorted(document_dir.glob("*.md")) for chunk in _structured_chunks(path)]
        self.searchable_documents = [self._searchable(doc) for doc in self.documents]
        self.corpus_tokens = [_tokens(text) for text in self.searchable_documents]
        self.model_client = RagModelClient()
        self.last_backend = "医学 Dense Retrieval"
        if self.advanced_enabled and self.searchable_documents:
            try:
                # Pay model loading and corpus encoding cost while the API starts,
                # instead of making the first user query wait for it.
                self.model_client.request({"action": "prepare", "documents": self.searchable_documents})
            except Exception:
                # Retrieval remains available through the lightweight local fallback.
                pass

    @property
    def advanced_enabled(self) -> bool:
        return os.getenv("DERMAGUARD_RAG_ADVANCED", "0").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _searchable(doc: dict[str, Any]) -> str:
        important = [str(doc.get("title", "")), str(doc.get("section", "")), *doc.get("conditions", []),
                     *doc.get("aliases", []), *doc.get("symptoms", []), *doc.get("tags", [])]
        return " ".join([*important, *important, str(doc["content"])])

    def _rank(self, query: str) -> list[tuple[float, dict[str, Any]]]:
        scores: list[float] | None = None
        if self.advanced_enabled:
            try:
                scores = [
                    max(0.0, min(1.0, float(value)))
                    for value in self.model_client.request(
                        {"action": "dense", "query": query, "documents": self.searchable_documents}
                    )["scores"]
                ]
                self.last_backend = "医学 Dense Retrieval"
            except Exception:
                scores = None
        if scores is None:
            scores = _lexical_fallback(query, self.documents, self.corpus_tokens)
            self.last_backend = "轻量本地词法 fallback"
        order = sorted(
            range(len(self.documents)),
            key=lambda index: (-scores[index], self.documents[index]["id"]),
        )
        return [(scores[index], self.documents[index]) for index in order]

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        limit = top_k or self.config.rag.top_k
        dynamic_query = _expand_query(query)
        ranked = self._rank(dynamic_query)
        # Prefer evidence from different source documents. Structured chunking
        # can otherwise fill Top-K with three adjacent paragraphs from one page.
        diverse_ranked = []
        seen_documents: set[str] = set()
        for score, doc in ranked:
            document_id = str(doc["document_id"])
            if document_id in seen_documents:
                continue
            diverse_ranked.append((score, doc))
            seen_documents.add(document_id)
            if len(diverse_ranked) == limit:
                break
        if len(diverse_ranked) < limit:
            selected_ids = {str(doc["id"]) for _, doc in diverse_ranked}
            diverse_ranked.extend(
                (score, doc) for score, doc in ranked
                if str(doc["id"]) not in selected_ids
            )
        results = []
        for score, doc in diverse_ranked[:limit]:
            results.append({
                "id": str(doc["id"]), "document_id": str(doc["document_id"]), "title": str(doc["title"]),
                "content": str(doc["content"]), "source": str(doc.get("source", "")),
                "score": max(0.0, min(1.0, score)), "evidence_type": _evidence_category(str(doc["content"])),
                "evidence_status": "充分" if score >= self.config.rag.evidence_threshold else "当前知识库证据不足",
                "citation": {"document": str(doc["document_name"]), "organization": str(doc["organization"]),
                    "year": str(doc["year"]), "section": str(doc["section"]), "page": str(doc["page"]),
                    "version": str(doc["version"])},
                "retrieval_backend": self.last_backend, "query": dynamic_query,
                "stored_in_database": True, "source_scope": "本地知识库",
            })
        return results
