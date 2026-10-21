from __future__ import annotations

from backend.app.care.schemas import GuidelineReference
from backend.app.services.rag import LocalRagRetriever
from backend.app.services.rag.retriever import build_dynamic_query


class KnowledgeService:
    def __init__(self, retriever: LocalRagRetriever | None = None) -> None:
        self.retriever = retriever or LocalRagRetriever()

    def search(self, query: str, top_k: int = 3) -> list[dict[str, object]]:
        return self.retriever.search(query, top_k=top_k)

    def search_patient_context(
        self,
        *,
        user_query: str = "",
        visual_candidates: list[str] | None = None,
        positive_symptoms: list[str] | None = None,
        risk_reasons: list[str] | None = None,
        duration: str | None = None,
        topic: str = "鉴别依据 检查依据 风险分级 转诊指南",
        top_k: int = 3,
    ) -> list[dict[str, object]]:
        query = build_dynamic_query(
            user_query=user_query,
            visual_candidates=visual_candidates,
            positive_symptoms=positive_symptoms,
            risk_reasons=risk_reasons,
            duration=duration,
            topic=topic,
        )
        return self.retriever.search(query, top_k=top_k)

    def references(self, chunks: list[dict[str, object]]) -> list[GuidelineReference]:
        return [
            GuidelineReference(id=str(item["id"]), title=str(item["title"]), source=str(item["source"]))
            for item in chunks
        ]

    @staticmethod
    def assemble_evidence(chunks: list[dict[str, object]]) -> dict[str, object]:
        categories = {
            "风险依据": [], "鉴别依据": [], "检查依据": [],
            "转诊依据": [], "冲突证据": [], "信息缺口": [],
        }
        for item in chunks:
            category = str(item.get("evidence_type", "信息缺口"))
            categories.setdefault(category, []).append(
                {
                    "evidence_id": item["id"],
                    "statement": item["content"],
                    "citation": item.get("citation", {}),
                    "source": item["source"],
                    "score": item["score"],
                }
            )
        sufficient = any(item.get("evidence_status") == "充分" for item in chunks)
        missing = [name for name, items in categories.items() if name != "信息缺口" and not items]
        return {
            "categories": categories,
            "evidence_sufficient": sufficient,
            "information_gaps": missing,
            "answer_policy": "可引用证据作有限解释" if sufficient else "当前知识库证据不足，禁止模型自行补充",
        }
