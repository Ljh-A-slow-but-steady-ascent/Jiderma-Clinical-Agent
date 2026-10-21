from __future__ import annotations

import base64
import json
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.database import CaseRepository


PROHIBITED_KEYS = frozenset(
    {
        "raw_image",
        "image_bytes",
        "image_base64",
        "local_image_path",
        "image_path",
        "exif",
        "name",
        "phone",
        "id_number",
        "address",
    }
)
PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_PATTERN = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
LOCAL_IMAGE_PATTERN = re.compile(
    r"(?:^|\s)(?:/|[A-Za-z]:\\)[^\s]+\.(?:jpe?g|png|webp|bmp|gif)(?:$|\s)", re.I
)
BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/]{512,}={0,2}$")


class EgressRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channel: Literal["llm", "mcp", "http"]
    destination: str
    payload: Any
    case_id: str | None = None
    tool_name: str | None = None


class EgressDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allowed: bool
    sanitized_payload: Any | None
    sanitized_fields: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)


class EgressBlockedError(RuntimeError):
    pass


def sanitize_text(value: str) -> tuple[str, list[str]]:
    fields: list[str] = []
    sanitized, count = PHONE_PATTERN.subn("[REDACTED_PHONE]", value)
    if count:
        fields.append("phone")
    sanitized, count = ID_PATTERN.subn("[REDACTED_ID_NUMBER]", sanitized)
    if count:
        fields.append("id_number")
    return sanitized, fields


def sanitize_for_storage(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value)[0]
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in PROHIBITED_KEYS else sanitize_for_storage(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_for_storage(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "[REDACTED_BINARY]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"[REDACTED_OBJECT:{type(value).__name__}]"


def _inspect(value: Any, path: str = "payload") -> list[str]:
    reasons: list[str] = []
    if isinstance(value, (bytes, bytearray, memoryview)):
        return [f"{path}:RAW_BINARY"]
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if (
            "把" in stripped
            and any(item in stripped for item in ["原始图片", "原始皮肤图片", "原始医疗影像"])
            and any(item in stripped for item in ["发给", "传给", "发送到", "上传到"])
        ) or ("忽略之前" in stripped and "base64" in lowered):
            reasons.append(f"{path}:RAW_IMAGE_EXFILTRATION_REQUEST")
        if "data:image/" in lowered:
            reasons.append(f"{path}:DATA_IMAGE_URI")
        compact = "".join(stripped.split())
        if BASE64_PATTERN.fullmatch(compact) or compact.startswith(("/9j/", "iVBORw0KGgo")):
            try:
                base64.b64decode(compact[: min(len(compact), 4096)], validate=False)
                reasons.append(f"{path}:POSSIBLE_IMAGE_BASE64")
            except Exception:
                pass
        if LOCAL_IMAGE_PATTERN.search(stripped) or "/data/cases/" in lowered:
            reasons.append(f"{path}:LOCAL_IMAGE_PATH")
        return reasons
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if str(key).lower() in PROHIBITED_KEYS and item not in (None, ""):
                reasons.append(f"{child}:PROHIBITED_FIELD")
            reasons.extend(_inspect(item, child))
        return reasons
    if isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            reasons.extend(_inspect(item, f"{path}[{index}]"))
        return reasons
    module = type(value).__module__.lower()
    name = type(value).__name__.lower()
    if "pil" in module or "image" in name:
        reasons.append(f"{path}:IMAGE_OBJECT")
    return reasons


class EgressGuard:
    def __init__(self, repository: CaseRepository | None = None) -> None:
        self.repository = repository

    def check(self, request: EgressRequest) -> EgressDecision:
        started = time.perf_counter()
        reasons = _inspect(request.payload)
        sanitized = sanitize_for_storage(request.payload)
        serialized_before = json.dumps(sanitized, ensure_ascii=False, default=str)
        sanitized_fields = sorted(
            field for field in {"phone", "id_number"} if f"REDACTED_{field.upper()}" in serialized_before
        )
        decision = EgressDecision(
            allowed=not reasons,
            sanitized_payload=None if reasons else sanitized,
            sanitized_fields=sanitized_fields,
            blocked_reasons=reasons,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if self.repository is not None:
            safe_keys = list(sanitized.keys()) if isinstance(sanitized, dict) else []
            self.repository.add_audit_event(
                {
                    "event_type": "egress_check",
                    "case_id": request.case_id,
                    "channel": request.channel,
                    "tool_name": request.tool_name,
                    "destination": request.destination,
                    "privacy_action": "BLOCK" if reasons else ("SANITIZE" if sanitized_fields else "ALLOW"),
                    "allowed": not reasons,
                    "sanitized_fields": safe_keys,
                    "latency_ms": decision.latency_ms,
                }
            )
        return decision

    def enforce(self, request: EgressRequest) -> EgressDecision:
        decision = self.check(request)
        if not decision.allowed:
            raise EgressBlockedError("RAW_IMAGE BLOCK policy: outbound payload rejected")
        return decision
