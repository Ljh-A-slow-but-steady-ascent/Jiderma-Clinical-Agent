from __future__ import annotations

from typing import Any

from backend.app.chat.gemini import GeminiChatProvider
from backend.app.chat.mock import chat_model_catalog


def test_model_catalog_reports_key_availability(monkeypatch) -> None:
    monkeypatch.setenv("ZHIPUAI_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    catalog = {item["id"]: item for item in chat_model_catalog()}
    assert catalog["glm-5.3-flash"]["available"] is True
    assert catalog["gemini-flash"]["available"] is False
    assert catalog["openai/gpt-oss-120b"]["available"] is False
    assert catalog["openai/gpt-oss-20b"]["available"] is False
    assert catalog["qwen/qwen3.8-27b"]["available"] is False
    assert catalog["deepseek-flash"]["available"] is True
    assert catalog["deepseek-v4-pro"]["available"] is True


class FakeGeminiResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"candidates": [{"content": {"parts": [{"text": "请继续线下评估。"}]}}]}


def test_gemini_provider_uses_text_only_sanitized_context(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> FakeGeminiResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeGeminiResponse()

    monkeypatch.setattr("backend.app.chat.gemini.requests.post", fake_post)
    provider = GeminiChatProvider(api_key="test-key", model="gemini-test")
    answer = provider.plan({"level": "REFERRAL"}, "只返回JSON")

    serialized = str(captured["json"])
    assert captured["headers"]["x-goog-api-key"] == "test-key"
    assert captured["url"].endswith("/gemini-test:generateContent")
    assert "REFERRAL" in serialized
    assert "image_bytes" not in serialized
    assert answer
