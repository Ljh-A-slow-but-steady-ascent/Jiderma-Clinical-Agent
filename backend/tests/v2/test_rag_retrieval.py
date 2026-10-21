from backend.app.services.rag.retriever import LocalRagRetriever


def test_pad_conditions_have_authoritative_sources():
    retriever = LocalRagRetriever()
    expected = {
        "actinic keratosis": "GUIDE_AAD_ACTINIC_KERATOSIS",
        "basal cell carcinoma": "GUIDE_AAD_BASAL_CELL_CARCINOMA",
        "seborrheic keratosis": "GUIDE_DERMNET_BENIGN_KERATOSIS",
        "melanoma": "GUIDE_AAD_MELANOMA",
        "nevus": "GUIDE_AAD_MELANOCYTIC_NEVUS",
        "squamous cell carcinoma": "GUIDE_NICE_NG12_SKIN",
    }
    for query, document_id in expected.items():
        ids = {item["document_id"] for item in retriever.search(query, top_k=3)}
        assert document_id in ids


def test_rag_results_are_distinct_and_traceable():
    results = LocalRagRetriever().search("黑色素瘤 变化 出血 转诊", top_k=3)
    assert len({item["document_id"] for item in results}) == len(results)
    for item in results:
        assert item["evidence_status"] in {"充分", "当前知识库证据不足"}
        assert set(item["citation"]) == {
            "document", "organization", "year", "section", "page", "version"
        }
        assert item["query"]
        assert item["stored_in_database"] is True
        assert item["source_scope"] == "本地知识库"
