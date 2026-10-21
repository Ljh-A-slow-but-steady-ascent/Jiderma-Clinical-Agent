"""Chinese display names for the current PAD multimodal classifier."""

from __future__ import annotations


DISPLAY_LABELS_ZH = {
    "nevus": "色素痣",
    "nev": "色素痣",
    "basal cell carcinoma": "基底细胞癌",
    "bcc": "基底细胞癌",
    "actinic keratosis": "日光性角化病",
    "ack": "日光性角化病",
    "seborrheic keratosis": "脂溢性角化病",
    "sek": "脂溢性角化病",
    "squamous cell carcinoma": "鳞状细胞癌",
    "scc": "鳞状细胞癌",
    "melanoma": "黑色素瘤",
    "mel": "黑色素瘤",
}


def display_label_zh(label: str) -> str:
    """Return a stable Chinese presentation label without changing model output."""
    raw = str(label).strip()
    return DISPLAY_LABELS_ZH.get(raw.lower(), raw)
