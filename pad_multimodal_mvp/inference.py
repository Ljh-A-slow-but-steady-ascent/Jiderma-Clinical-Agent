"""Single-sample PAD image + JSON metadata inference."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .core import (
    DEFAULT_CHECKPOINT,
    DermFMPADClassifier,
    build_pad_prompt,
    load_dermfm_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--dermfm-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def preprocess_image(path: Path) -> torch.Tensor:
    pipeline = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    with Image.open(path) as image:
        return pipeline(image.convert("RGB")).unsqueeze(0)


def main() -> None:
    args = parse_args()
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    prompt = build_pad_prompt(metadata)
    vision, text, tokenizer = load_dermfm_components(args.dermfm_checkpoint)
    saved = torch.load(args.model_checkpoint, map_location="cpu", weights_only=False)
    model = DermFMPADClassifier(
        vision, text, mode=saved["mode"], num_classes=int(saved["num_classes"])
    )
    model.load_trainable_checkpoint(args.model_checkpoint)
    model.to(device).eval()
    image = preprocess_image(args.image).to(device) if saved["mode"] != "text_only" else None
    tokens = tokenizer([prompt]).to(device) if saved["mode"] != "image_only" else None
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else nullcontext()
    )
    with torch.inference_mode(), context:
        probabilities = model(image, tokens).float().softmax(dim=1)[0].cpu()
    label_names = {int(key): value for key, value in saved["label_names"].items()}
    k = min(max(args.top_k, 1), len(label_names))
    values, indices = probabilities.topk(k)
    result = {
        "mode": saved["mode"],
        "image": str(args.image.resolve()),
        "prompt": prompt,
        "top_k": [
            {
                "class_id": int(index),
                "disease": label_names[int(index)],
                "probability": float(value),
            }
            for value, index in zip(values, indices)
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
