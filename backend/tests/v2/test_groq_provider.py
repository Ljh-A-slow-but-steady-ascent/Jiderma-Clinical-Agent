from __future__ import annotations

from backend.app.chat.groq import GroqChatProvider


class FakeResponse:
    status_code = 200
    headers = {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "请按本地分级结果进行后续评估。"}}]}


def test_groq_provider_sends_only_supplied_sanitized_context(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("backend.app.chat.groq.requests.post", fake_post)
    provider = GroqChatProvider(api_key="test-key", model="test-model")
    answer = provider.plan({"level": "REFERRAL"}, "只返回JSON")

    serialized = str(captured["json"])
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "test-model"
    assert "REFERRAL" in serialized
    assert "image_bytes" not in serialized
    assert "local_image_path" not in serialized
    assert answer


def test_groq_current_models_include_supported_reasoning(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("backend.app.chat.groq.requests.post", fake_post)
    provider = GroqChatProvider(api_key="test-key", model="openai/gpt-oss-120b")
    provider.plan({"task": "结构化"}, "只返回JSON")
    assert captured["json"]["reasoning_effort"] == "low"
    assert captured["json"]["max_tokens"] == 700
