"""Subprocess bridge for the trained PAD image-text classifier."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# All model/tokenizer artifacts are already permanent local files. Prevent
# availability checks from introducing proxy/network delays during inference.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PAD_MULTIMODAL_HEAD = Path(
    os.getenv(
        "DERMAGUARD_PAD_MULTIMODAL_CHECKPOINT",
        str(ROOT / "artifacts" / "training" / "pad_multimodal_mvp" / "image_text" / "best.pt"),
    )
)
PAD_LABELS = {
    "NEV": "nevus",
    "BCC": "basal cell carcinoma",
    "ACK": "actinic keratosis",
    "SEK": "seborrheic keratosis",
    "SCC": "squamous cell carcinoma",
    "MEL": "melanoma",
}


def predict_pad_multimodal(image_path: Path, metadata: dict, top_k: int) -> dict:
    """Run the trained official image-text fusion adapter on one clinical case."""
    from pad_multimodal_mvp.core import DermFMPADClassifier, build_pad_prompt, load_dermfm_components
    from pad_multimodal_mvp.inference import preprocess_image

    if not PAD_MULTIMODAL_HEAD.is_file():
        raise FileNotFoundError(f"PAD multimodal checkpoint not found: {PAD_MULTIMODAL_HEAD}")
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt = build_pad_prompt(metadata)
    vision, text, tokenizer = load_dermfm_components()
    saved = torch.load(PAD_MULTIMODAL_HEAD, map_location="cpu", weights_only=False)
    if saved.get("mode") != "image_text":
        raise RuntimeError("Configured PAD checkpoint is not an image_text checkpoint")
    model = DermFMPADClassifier(
        vision, text, mode="image_text", num_classes=int(saved["num_classes"])
    )
    model.load_trainable_checkpoint(PAD_MULTIMODAL_HEAD)
    model.to(device).eval()
    image = preprocess_image(image_path).to(device)
    tokens = tokenizer([prompt]).to(device)
    with torch.inference_mode():
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                probabilities = model(image, tokens).float().softmax(dim=1)[0].cpu()
        else:
            probabilities = model(image, tokens).float().softmax(dim=1)[0].cpu()
    label_names = {int(key): value for key, value in saved["label_names"].items()}
    values, indices = probabilities.topk(min(top_k, len(label_names)))
    predictions = [
        {
            "label": PAD_LABELS.get(str(label_names[int(index)]), str(label_names[int(index)])),
            "score": float(value),
        }
        for value, index in zip(values, indices)
    ]
    if not all(math.isfinite(item["score"]) for item in predictions):
        raise RuntimeError("PAD multimodal head returned a non-finite score")
    return {
        "model_name": "local_multimodal_classifier",
        "candidate_set": "pad_clinical_multimodal_v1",
        "top_predictions": predictions,
        "prediction_mode": "supervised_multimodal_pad",
        "input_modality": "clinical_image+structured_text",
        "latency_ms": (time.perf_counter() - started) * 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    metadata = json.loads(sys.stdin.read())
    result = predict_pad_multimodal(Path(args.image), metadata, args.top_k)
    print("DERMAGUARD_RESULT=" + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
