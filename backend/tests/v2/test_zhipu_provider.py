from __future__ import annotations

import pytest

from backend.app.chat.zhipu import ZhipuChatProvider


class FakeResponse:
    status_code = 200
    headers = {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "请按本地分级结果进行后续评估。"}}]}


def test_zhipu_provider_uses_chat_completions_without_image(monkeypatch):
    captured = {}
    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("backend.app.chat.zhipu.requests.post", fake_post)
    provider = ZhipuChatProvider(api_key="test-key", model="glm-test")
    answer = provider.plan({"level": "REFERRAL"}, "只返回JSON")

    serialized = str(captured["json"])
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "glm-test"
    assert captured["json"]["temperature"] == 0.05
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert "image_bytes" not in serialized
    assert "local_image_path" not in serialized
    assert answer


def test_zhipu_provider_rejects_empty_content(monkeypatch):
    class EmptyResponse(FakeResponse):
        def json(self) -> dict:
            return {
                "choices": [
                    {"message": {"content": ""}, "finish_reason": "length"}
                ]
            }

    monkeypatch.setattr(
        "backend.app.chat.zhipu.requests.post", lambda *args, **kwargs: EmptyResponse()
    )
    provider = ZhipuChatProvider(api_key="test-key", model="glm-test")
    with pytest.raises(RuntimeError, match="空正文"):
        provider.plan({"task": "结构化"}, "只返回JSON")
