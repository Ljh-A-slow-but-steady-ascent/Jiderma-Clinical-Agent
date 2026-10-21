from __future__ import annotations

from backend.app.chat.deepseek import DeepSeekChatProvider
from backend.app.chat.mock import create_chat_provider


class FakeResponse:
    status_code = 200
    headers = {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "请结合现有证据继续评估。"}}]}


def test_deepseek_provider_uses_sanitized_text_and_thinking_mode(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("backend.app.chat.deepseek.requests.post", fake_post)
    provider = DeepSeekChatProvider(api_key="test-key", model="deepseek-flash")
    answer = provider.plan({"evidence": "已脱敏证据"}, "只返回JSON")

    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "deepseek-flash"
    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert "已脱敏证据" in str(captured["json"])
    assert "image_bytes" not in str(captured["json"])
    assert answer


def test_deepseek_factory(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    provider = create_chat_provider("deepseek-v4-pro")
    assert isinstance(provider, DeepSeekChatProvider)
    assert provider.model == "deepseek-v4-pro"
