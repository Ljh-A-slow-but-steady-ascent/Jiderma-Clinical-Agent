from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from backend.app.chat.mock import create_chat_provider
from backend.app.services.dermfm_candidate_sets import display_label_zh
from backend.app.services.privacy import EgressGuard, EgressRequest, sanitize_text


REPORT_KEYS = (
    "clinical_summary",
    "visual_summary",
    "risk_interpretation",
    "evidence_summary",
    "recommendations",
    "follow_up",
)


@dataclass(frozen=True)
class GeneratedReport:
    content: str
    structured: dict[str, Any]
    provider: str
    model: str


def _fallback(payload: dict[str, Any]) -> dict[str, str]:
    top = payload.get("visual", {}).get("top_predictions", [])
    candidates = "、".join(display_label_zh(str(item.get("label", "未知候选"))) for item in top[:3]) or "暂无稳定候选"
    triage = payload.get("triage", {})
    care = payload.get("care_pathway", {})
    return {
        "clinical_summary": str(payload.get("chief_complaint") or "已完成图像与症状信息采集。"),
        "visual_summary": f"图像辅助分析候选包括：{candidates}。候选用于辅助鉴别，不代表确诊。",
        "risk_interpretation": f"当前辅助风险分级为 {triage.get('level', '待医生复核')}。",
        "evidence_summary": "系统已结合多模态候选、结构化病例与本地医学知识整理当前证据。",
        "recommendations": str(care.get("action") or "请结合面诊、皮肤镜检查及必要的进一步检查进行判断。"),
        "follow_up": "如病灶持续变化、出血、破溃或症状加重，请及时复评并完善记录。",
    }


def _parse_json(raw: str) -> dict[str, Any]:
    value = raw.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0]
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("报告模型未返回 JSON 对象")
    result: dict[str, Any] = {}
    for key in REPORT_KEYS:
        item = parsed.get(key)
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"报告字段缺失：{key}")
        result[key] = item.strip()[:3000]
    return result


def _mask_text(value: Any, replacements: list[str]) -> Any:
    if isinstance(value, str):
        masked = sanitize_text(value)[0]
        for item in replacements:
            if item:
                masked = masked.replace(item, "[REDACTED_IDENTITY]")
        return masked
    if isinstance(value, dict):
        return {key: _mask_text(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_mask_text(item, replacements) for item in value]
    return value


def render_report(patient: dict[str, Any], case_id: str, structured: dict[str, Any]) -> str:
    return "\n".join([
        "# 霁肤 · Jiderma 皮肤辅助诊断报告", "",
        "## 患者信息",
        f"- 姓名：{patient['name']}", f"- 性别：{patient['sex']}",
        f"- 年龄：{patient['age']} 岁", f"- 门诊号：{patient['outpatient_no']}",
        f"- 病灶部位：{patient['lesion_site']}", f"- 病例编号：{case_id}", "",
        "## 患者病灶图像", f"![患者病灶图片](/v1/cases/{case_id}/image)", "",
        "## 主诉与病史摘要", structured["clinical_summary"], "",
        "## 图像辅助分析", structured["visual_summary"], "",
        "## 风险与病例信息", structured["risk_interpretation"], "",
        "## 医学证据摘要", structured["evidence_summary"], "",
        "## 建议", structured["recommendations"], "",
        "## 随访提示", structured["follow_up"], "",
        "## 医生补充意见", "请在此处补充医生意见。", "", "---",
        "本报告由智能系统生成初稿，经医生审核、编辑后方可作为辅助记录；不能替代临床诊断或病理检查。",
    ])


class ReportService:
    def __init__(self, guard: EgressGuard) -> None:
        self.guard = guard

    def generate(
        self, *, case_id: str, patient: dict[str, Any],
        clinical_payload: dict[str, Any], model_id: str,
    ) -> GeneratedReport:
        replacements = [patient.get("name", ""), patient.get("outpatient_no", ""), patient.get("phone", "")]
        outbound = _mask_text({
            "case_ref": "CASE_REDACTED", "age": patient.get("age"), "sex": patient.get("sex"),
            "lesion_site": patient.get("lesion_site"), "chief_complaint": patient.get("chief_complaint"),
            "medical_history": patient.get("medical_history", ""),
            "allergy_history": patient.get("allergy_history", ""),
            **clinical_payload,
        }, replacements)
        checked = self.guard.enforce(EgressRequest(
            channel="llm", destination="structured_report_generator",
            payload=outbound, case_id=case_id,
        )).sanitized_payload
        structured = _fallback(checked)
        provider_name, actual_model = "local", "template"
        if model_id != "local-template":
            try:
                provider = create_chat_provider(model_id)
                prompt = {
                    "task": "根据结构化病例证据填写辅助诊断报告字段，只返回 JSON，不得新增诊断、药物或未提供的事实。",
                    "required_schema": {key: "string" for key in REPORT_KEYS},
                    "case": checked,
                }
                structured = _parse_json(provider.plan(
                    prompt, "你是受约束的皮肤科报告字段整理器，只能返回指定 JSON。"
                ))
                provider_name = provider.name
                actual_model = getattr(provider, "model", model_id)
            except Exception:
                provider_name, actual_model = "local-fallback", "template"
        return GeneratedReport(
            content=render_report(patient, case_id, structured), structured=structured,
            provider=provider_name, model=actual_model,
        )
