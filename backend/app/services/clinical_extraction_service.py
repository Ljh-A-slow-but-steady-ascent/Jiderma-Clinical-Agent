from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from backend.app.services.privacy import EgressGuard, EgressRequest, sanitize_text


REGIONS = (
    "ABDOMEN", "ARM", "BACK", "CHEST", "EAR", "FACE", "FOOT",
    "FOREARM", "HAND", "LIP", "NECK", "NOSE", "SCALP", "THIGH",
)
CLINICAL_FIELDS = (
    "age", "gender", "region", "itch", "hurt", "bleed", "grew", "changed",
    "diameter_1", "diameter_2", "skin_cancer_history", "cancer_history",
)
EXTRACTED_FIELDS = tuple(item for item in CLINICAL_FIELDS if item not in {"age", "gender"})
FIELD_LABELS = {
    "patient_name": "患者姓名",
    "outpatient_no": "患者编号或门诊号",
    "sex": "性别",
    "age": "年龄",
    "lesion_site": "病灶部位",
    "region": "解剖部位",
    "itch": "是否瘙痒",
    "hurt": "是否疼痛",
    "bleed": "是否出血",
    "grew": "是否近期增大",
    "changed": "颜色、边界或外观是否变化",
    "diameter_1": "病灶长径",
    "diameter_2": "病灶短径",
    "skin_cancer_history": "既往皮肤肿瘤史",
    "cancer_history": "其他肿瘤史",
}

EXTRACTION_SYSTEM_PROMPT = """你是临床病例字段抽取器，不是诊断模型。
只能依据医生提供的原文抽取事实，不得推测、补全或改写疾病结论。
必须只返回 JSON 对象。未明确提供的字段必须返回 null，不能把未提及理解为 false。
region 只能是给定英文枚举；布尔字段只能是 true、false 或 null；尺寸单位统一为毫米。
"""


@dataclass(frozen=True)
class ClinicalExtractionResult:
    metadata: dict[str, Any]
    missing_fields: list[str]
    uncertain_fields: list[str]
    provider: str
    model: str
    privacy_status: str
    sanitized_fields: list[str]


@dataclass(frozen=True)
class AgentIntakeExtractionResult:
    intake: dict[str, Any]
    missing_fields: list[str]
    uncertain_fields: list[str]
    provider: str
    model: str
    privacy_status: str
    sanitized_fields: list[str]


def _strip_json_fence(raw: str) -> dict[str, Any]:
    text = str(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("病例抽取模型未返回 JSON 对象")
    return parsed


def _optional_bool(value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "yes", "1", "是", "有"}:
        return True
    if lowered in {"false", "no", "0", "否", "无"}:
        return False
    return None


def _optional_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0 < number <= 200 else None


def _normalise_metadata(raw: dict[str, Any], age: int, gender: str) -> dict[str, Any]:
    source = raw.get("metadata", raw)
    if not isinstance(source, dict):
        raise ValueError("病例抽取结果中缺少 metadata 对象")
    region = str(source.get("region") or "").strip().upper() or None
    if region not in REGIONS:
        region = None
    result: dict[str, Any] = {"age": age, "gender": gender, "region": region}
    for field in ("itch", "hurt", "bleed", "grew", "changed", "skin_cancer_history", "cancer_history"):
        result[field] = _optional_bool(source.get(field))
    result["diameter_1"] = _optional_number(source.get("diameter_1"))
    result["diameter_2"] = _optional_number(source.get("diameter_2"))
    return result


def _is_negated(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        if re.search(rf"(?:无|未见|否认|没有|未出现)[^。；;]{{0,18}}{re.escape(term)}", text):
            return True
    return False


def _local_bool(text: str, terms: tuple[str, ...]) -> bool | None:
    for term in terms:
        labelled = re.search(
            rf"{re.escape(term)}\s*[：:]?\s*(是|否|有|无|存在|没有|未见)", text
        )
        if labelled:
            return labelled.group(1) in {"是", "有", "存在"}
    if _is_negated(text, terms):
        return False
    return True if any(term in text for term in terms) else None


def _local_extract(narrative: str, age: int, gender: str) -> dict[str, Any]:
    """Offline development fallback; it preserves unknowns instead of guessing."""
    region_terms = {
        "ABDOMEN": ("腹部",), "ARM": ("上臂", "上肢"), "BACK": ("背部",),
        "CHEST": ("胸部",), "EAR": ("耳部", "耳廓"), "FACE": ("面部", "脸部"),
        "FOOT": ("足部", "脚部"), "FOREARM": ("前臂",), "HAND": ("手部",),
        "LIP": ("唇部", "嘴唇"), "NECK": ("颈部",), "NOSE": ("鼻部",),
        "SCALP": ("头皮",), "THIGH": ("大腿",),
    }
    region = next((code for code, terms in region_terms.items() if any(term in narrative for term in terms)), None)
    sizes = re.search(
        r"(?:约|大约)?\s*(\d+(?:\.\d+)?)\s*(?:毫米|mm)?\s*[xX×乘*]\s*(\d+(?:\.\d+)?)\s*(?:毫米|mm)?",
        narrative,
        flags=re.I,
    )
    diameter_1 = float(sizes.group(1)) if sizes else None
    diameter_2 = float(sizes.group(2)) if sizes else None
    return {
        "age": age,
        "gender": gender,
        "region": region,
        "itch": _local_bool(narrative, ("瘦痒", "痒")),
        "hurt": _local_bool(narrative, ("疼痛", "痛")),
        "bleed": _local_bool(narrative, ("出血", "渗血")),
        "grew": _local_bool(narrative, ("增大", "长大", "变大")),
        "changed": _local_bool(narrative, ("近期变化", "颜色变化", "边界变化", "外观变化", "发生变化")),
        "diameter_1": diameter_1,
        "diameter_2": diameter_2,
        "skin_cancer_history": _local_bool(narrative, ("皮肤癌病史", "皮肤肿瘤史")),
        "cancer_history": _local_bool(narrative, ("其他癌症病史", "其他肿瘤史")),
    }


class ClinicalExtractionService:
    def __init__(self, guard: EgressGuard) -> None:
        self.guard = guard

    @staticmethod
    def _mask_identity(narrative: str, identity_values: list[str]) -> str:
        masked = sanitize_text(narrative)[0]
        for value in sorted((item.strip() for item in identity_values if item.strip()), key=len, reverse=True):
            masked = masked.replace(value, "[REDACTED_IDENTITY]")
        return masked

    def extract(
        self,
        *,
        narrative: str,
        age: int,
        gender: Literal["MALE", "FEMALE"],
        model_id: str,
        identity_values: list[str],
    ) -> ClinicalExtractionResult:
        # Import lazily: backend.app.chat initializes the Agent, which depends
        # on the service container that owns this extraction service.
        from backend.app.chat.mock import create_chat_provider

        safe_narrative = self._mask_identity(narrative, identity_values)
        provider = create_chat_provider(model_id)
        payload = {
            "task": "从医生自然语言病例描述中抽取本地多模态模型所需字段。",
            "known_demographics": {"age": age, "gender": gender},
            "clinical_narrative": safe_narrative,
            "allowed_regions": list(REGIONS),
            "required_output": {
                "metadata": {field: "value or null" for field in EXTRACTED_FIELDS},
                "uncertain_fields": ["field_name"],
            },
        }
        decision = self.guard.enforce(EgressRequest(
            channel="llm", destination=provider.name, payload=payload,
        ))
        checked = decision.sanitized_payload
        if provider.name == "mock":
            metadata = _local_extract(safe_narrative, age, gender)
            uncertain_fields: list[str] = []
        else:
            parsed = _strip_json_fence(provider.plan(checked, EXTRACTION_SYSTEM_PROMPT))
            metadata = _normalise_metadata(parsed, age, gender)
            uncertain_fields = [
                str(item) for item in parsed.get("uncertain_fields", [])
                if str(item) in EXTRACTED_FIELDS
            ][:10]
        missing = [field for field in CLINICAL_FIELDS if metadata.get(field) is None]
        return ClinicalExtractionResult(
            metadata=metadata,
            missing_fields=missing,
            uncertain_fields=uncertain_fields,
            provider=provider.name,
            model=getattr(provider, "model", model_id),
            privacy_status="SAFE",
            sanitized_fields=decision.sanitized_fields,
        )

    def extract_agent_intake(
        self, *, narrative: str, model_id: str
    ) -> AgentIntakeExtractionResult:
        """Build a complete Agent intake from one clinician-authored narrative.

        Direct identifiers are deliberately parsed locally and masked before the
        optional cloud extractor sees the clinical narrative.
        """
        text = narrative.strip()
        name_match = re.search(
            r"(?:患者姓名|姓名|患者)\s*[：:]?\s*([一-鿿·]{2,20})", text
        )
        number_match = re.search(
            r"(?:门诊号|病历号|患者编号|编号)\s*[：:]?\s*([A-Za-z0-9_-]{2,80})", text,
            flags=re.I,
        )
        phone_match = re.search(r"(?<!\d)(1[3-9]\d{9})(?!\d)", text)
        age_match = re.search(r"(?<!\d)(\d{1,3})\s*岁", text)
        patient_name = name_match.group(1) if name_match else ""
        outpatient_no = number_match.group(1) if number_match else ""
        phone = phone_match.group(1) if phone_match else ""
        age = int(age_match.group(1)) if age_match and int(age_match.group(1)) <= 130 else None
        sex = "女" if re.search(r"(?:性别\s*[：:]?\s*)?女(?:性)?", text) else (
            "男" if re.search(r"(?:性别\s*[：:]?\s*)?男(?:性)?", text) else ""
        )
        region_terms = {
            "腹部": "ABDOMEN", "上臂": "ARM", "背部": "BACK", "胸部": "CHEST",
            "耳部": "EAR", "耳廓": "EAR", "面部": "FACE", "脸部": "FACE",
            "足部": "FOOT", "脚部": "FOOT", "前臂": "FOREARM", "手部": "HAND",
            "唇部": "LIP", "颈部": "NECK", "鼻部": "NOSE", "头皮": "SCALP", "大腿": "THIGH",
        }
        lesion_site = next((label for label in region_terms if label in text), "")
        metadata: dict[str, Any] = {}
        uncertain: list[str] = []
        provider_name = "not_called"
        provider_model = model_id
        sanitized_fields: list[str] = []
        if age is not None and sex:
            clinical = self.extract(
                narrative=text,
                age=age,
                gender="MALE" if sex == "男" else "FEMALE",
                model_id=model_id,
                identity_values=[patient_name, outpatient_no, phone],
            )
            metadata = clinical.metadata
            uncertain = clinical.uncertain_fields
            provider_name = clinical.provider
            provider_model = clinical.model
            sanitized_fields = clinical.sanitized_fields
        intake = {
            "patient_name": patient_name,
            "sex": sex,
            "age": age,
            "outpatient_no": outpatient_no,
            "phone": phone,
            "medical_history": "",
            "allergy_history": "",
            "lesion_site": lesion_site,
            "chief_complaint": text,
            "clinical_metadata": metadata,
        }
        missing = [
            field for field in ("patient_name", "outpatient_no", "sex", "age", "lesion_site")
            if intake.get(field) in (None, "")
        ]
        if age is not None and sex:
            missing.extend(field for field in CLINICAL_FIELDS if metadata.get(field) is None)
        return AgentIntakeExtractionResult(
            intake=intake,
            missing_fields=list(dict.fromkeys(missing)),
            uncertain_fields=uncertain,
            provider=provider_name,
            model=provider_model,
            privacy_status="SAFE",
            sanitized_fields=sanitized_fields,
        )


def field_labels(fields: list[str]) -> list[str]:
    return [FIELD_LABELS.get(field, field) for field in fields]
