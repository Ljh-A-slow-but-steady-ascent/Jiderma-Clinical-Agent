from __future__ import annotations

import json
import os
from typing import Any, Protocol


class ChatProvider(Protocol):
    name: str
    model: str

    def plan(self, payload: dict[str, Any], system_prompt: str) -> str: ...


class MockChatProvider:
    """Offline planner; callers retain deterministic local parsing defaults."""

    name = "mock"
    model = "local-mock"

    def plan(self, payload: dict[str, Any], system_prompt: str) -> str:
        return json.dumps({}, ensure_ascii=False)


def chat_model_catalog() -> list[dict[str, Any]]:
    """Return the controlled Deep Agent model allowlist without credentials."""
    has_zhipu = bool(os.getenv("ZHIPUAI_API_KEY"))
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    has_deepseek = bool(os.getenv("DEEPSEEK_API_KEY"))
    has_auto = has_deepseek or has_zhipu or has_groq
    return [
        {"id": "auto", "label": "自动选择", "available": has_auto,
         "status": "可用" if has_auto else "未配置Deep Agent模型密钥"},
        {"id": "deepseek-flash", "label": "DeepSeek Flash", "available": has_deepseek,
         "status": "可用" if has_deepseek else "未配置 DeepSeek 密钥"},
        {"id": "deepseek-v4-pro", "label": "DeepSeek V4 Pro", "available": has_deepseek,
         "status": "可用" if has_deepseek else "未配置 DeepSeek 密钥"},
        {"id": "glm-5.3-flash", "label": "智谱 GLM-5.3-Flash", "available": has_zhipu,
         "status": "可用" if has_zhipu else "未配置密钥"},
        {"id": "glm-4-flash", "label": "智谱 GLM-4-Flash", "available": has_zhipu,
         "status": "可用" if has_zhipu else "未配置密钥"},
        {"id": "openai/gpt-oss-120b", "label": "Groq·GPT-OSS 120B", "available": has_groq,
         "status": "可用" if has_groq else "未配置 Groq 密钥"},
        {"id": "openai/gpt-oss-20b", "label": "Groq·GPT-OSS 20B", "available": has_groq,
         "status": "可用" if has_groq else "未配置 Groq 密钥"},
        {"id": "qwen/qwen3.8-27b", "label": "Groq·Qwen3.8 27B", "available": has_groq,
         "status": "可用（预览）" if has_groq else "未配置 Groq 密钥"},
        {"id": "gemini-flash", "label": "Gemini Flash", "available": False,
         "status": "当前Deep Agent Runtime未启用" if has_gemini else "未配置密钥"},
    ]


def create_chat_provider(model_id: str | None = None) -> ChatProvider:
    """Create a JSON planner for clinical extraction and report generation."""
    if model_id and model_id != "auto":
        if model_id in {"deepseek-flash", "deepseek-v4-pro"}:
            from backend.app.chat.deepseek import DeepSeekChatProvider
            return DeepSeekChatProvider(model=model_id)
        if model_id in {"glm-5.3-flash", "glm-4-flash"}:
            from backend.app.chat.zhipu import ZhipuChatProvider
            return ZhipuChatProvider(model=model_id)
        if model_id == "gemini-flash":
            from backend.app.chat.gemini import GeminiChatProvider
            return GeminiChatProvider()
        if model_id in {"openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"}:
            from backend.app.chat.groq import GroqChatProvider
            return GroqChatProvider(model=model_id)
        raise ValueError(f"不支持的模型标识：{model_id}")

    provider = os.getenv("DERMAGUARD_CHAT_PROVIDER", "mock").strip().lower()
    if provider == "mock":
        return MockChatProvider()
    if provider == "groq":
        from backend.app.chat.groq import GroqChatProvider
        return GroqChatProvider()
    if provider == "deepseek":
        from backend.app.chat.deepseek import DeepSeekChatProvider
        return DeepSeekChatProvider()
    if provider in {"zhipu", "zhipu_glm", "glm"}:
        from backend.app.chat.zhipu import ZhipuChatProvider
        return ZhipuChatProvider()
    raise ValueError(f"不支持的对话提供方：{provider}")
