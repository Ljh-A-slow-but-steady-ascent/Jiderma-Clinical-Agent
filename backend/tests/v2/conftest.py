import pytest


@pytest.fixture(autouse=True)
def force_offline_mock_provider(monkeypatch):
    """单元测试不得消耗任何云端免费额度。"""
    monkeypatch.setenv("DERMAGUARD_CHAT_PROVIDER", "mock")
    monkeypatch.setenv("DERMAGUARD_AUTH_DISABLED", "1")
