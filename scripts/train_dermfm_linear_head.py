#!/usr/bin/env python3
"""Train a small supervised head on frozen DermFM-Zero image features.

The script never updates the official checkpoint. It first caches normalized
768-dimensional image embeddings, then selects a linear classifier on the
official validation split and evaluates the selected head once on the test
split.
"""

from __future__ import annotations

import argparse
import csv
import logging
import json
import math
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LOGGER = logging.getLogger("dermfm_linear_head")


def configure_logging(output_dir: Path) -> Path:
    log_path = output_dir / "training.log"
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
    return log_path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CsvImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], repo_dir: Path, preprocess: Any) -> None:
        self.rows = rows
        self.repo_dir = repo_dir
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows[index]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = self.repo_dir / image_path
        with Image.open(image_path) as image:
            tensor = self.preprocess(image.convert("RGB"))
        return tensor, int(row["label"]), str(image_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT
        / "third_party/DermFM-Zero/data/linear_probe/HAM-official-7-lp/meta.csv",
    )
    parser.add_argument(
        "--dermfm-repo",
        type=Path,
        default=ROOT / "third_party/DermFM-Zero",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/training/ham7_linear_head",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 1000])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--refresh-features", action="store_true")
    return parser.parse_args()


def load_rows(csv_path: Path) -> tuple[list[dict[str, str]], dict[int, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_path", "label", "split", "diag"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"CSV must contain {sorted(required)}: {csv_path}")
    splits = {row["split"] for row in rows}
    if splits != {"train", "val", "test"}:
        raise ValueError(f"Expected official train/val/test splits, got: {sorted(splits)}")
    label_names: dict[int, str] = {}
    for row in rows:
        label = int(row["label"])
        diagnosis = row["diag"].strip()
        previous = label_names.setdefault(label, diagnosis)
        if previous != diagnosis:
            raise ValueError(f"Label {label} maps to both {previous!r} and {diagnosis!r}")
    expected = list(range(len(label_names)))
    if sorted(label_names) != expected:
        raise ValueError(f"Labels must be contiguous from zero, got {sorted(label_names)}")
    return rows, label_names


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def feature_cache_path(output_dir: Path, split: str) -> Path:
    return output_dir / f"{split}_features.pt"


def build_linear_probe_preprocess() -> transforms.Compose:
    """Match the author's DermFM linear-probe preprocessing."""
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def extract_features(
    rows: list[dict[str, str]],
    split: str,
    provider: Any,
    preprocess: Any,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    dataset = CsvImageDataset(selected, provider.repo_dir, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    paths: list[str] = []
    started = time.perf_counter()
    for batch_index, (images, batch_labels, batch_paths) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    batch_features = provider.model.encode_image(images)
            else:
                batch_features = provider.model.encode_image(images)
            batch_features = torch.nn.functional.normalize(batch_features.float(), dim=-1)
        features.append(batch_features.cpu())
        labels.append(batch_labels.long().cpu())
        paths.extend(batch_paths)
        if batch_index % 50 == 0 or batch_index == len(loader):
            LOGGER.info("feature extraction split=%s progress=%d/%d", split, len(paths), len(dataset))
    payload = {
        "features": torch.cat(features),
        "labels": torch.cat(labels),
        "paths": paths,
        "split": split,
        "elapsed_seconds": time.perf_counter() - started,
    }
    torch.save(payload, feature_cache_path(output_dir, split))
    return payload


def confusion_matrix(target: torch.Tensor, prediction: torch.Tensor, classes: int) -> torch.Tensor:
    flat = target.to(torch.int64) * classes + prediction.to(torch.int64)
    return torch.bincount(flat, minlength=classes * classes).reshape(classes, classes)


def metrics(logits: torch.Tensor, target: torch.Tensor, classes: int) -> dict[str, Any]:
    prediction = logits.argmax(dim=1)
    matrix = confusion_matrix(target.cpu(), prediction.cpu(), classes)
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    true_positive = matrix.diag()
    recall = true_positive.float() / support.clamp_min(1).float()
    precision = true_positive.float() / predicted.clamp_min(1).float()
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "accuracy": float(true_positive.sum() / matrix.sum().clamp_min(1)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def class_weights(labels: torch.Tensor, classes: int, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = torch.bincount(labels, minlength=classes).float()
    inverse = counts.sum() / (classes * counts.clamp_min(1))
    if mode == "sqrt":
        inverse = inverse.sqrt()
    return inverse / inverse.mean()


def train_candidate(
    train: dict[str, Any],
    validation: dict[str, Any],
    classes: int,
    seed: int,
    weight_mode: str,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[dict[str, Any]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    feature_dim = int(train["features"].shape[1])
    head = nn.Linear(feature_dim, classes).to(device)
    weight = class_weights(train["labels"], classes, weight_mode)
    criterion = nn.CrossEntropyLoss(weight=None if weight is None else weight.to(device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(train["features"], train["labels"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, generator=generator)
    val_x = validation["features"].to(device)
    val_y = validation["labels"].to(device)
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        total_examples = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_y)
            total_examples += len(batch_y)
        head.eval()
        with torch.inference_mode():
            val_logits = head(val_x)
        report = metrics(val_logits, val_y, classes)
        val_loss = float(torch.nn.functional.cross_entropy(val_logits, val_y))
        score = report["macro_f1"]
        improved = score > best_score + 1e-7
        history.append(
            {
                "seed": seed,
                "weight_mode": weight_mode,
                "epoch": epoch,
                "train_loss": total_loss / max(total_examples, 1),
                "val_loss": val_loss,
                "val_accuracy": report["accuracy"],
                "val_balanced_accuracy": report["balanced_accuracy"],
                "val_macro_f1": report["macro_f1"],
                "is_best": int(improved),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best_state is not None
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        val_logits = head(val_x).cpu()
    report = metrics(val_logits, validation["labels"], classes)
    report.update({"seed": seed, "weight_mode": weight_mode, "best_epoch": best_epoch})
    return best_state, report, history


def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    temperatures = torch.logspace(math.log10(0.2), math.log10(5.0), 300)
    losses = torch.stack(
        [torch.nn.functional.cross_entropy(logits / value, labels) for value in temperatures]
    )
    index = int(losses.argmin())
    return float(temperatures[index]), float(losses[index])


def main() -> None:
    args = parse_args()
    args.csv = args.csv.resolve()
    args.dermfm_repo = args.dermfm_repo.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = configure_logging(args.output_dir)
    started_at = datetime.now().astimezone()
    rows, label_names = load_rows(args.csv)
    classes = len(label_names)
    split_counts = Counter(row["split"] for row in rows)
    LOGGER.info("run started at %s", started_at.isoformat())
    LOGGER.info("official split counts=%s", dict(split_counts))
    LOGGER.info("labels=%s", label_names)
    device = resolve_device(args.device)
    run_config = {
        "started_at": started_at.isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "split_counts": dict(split_counts),
        "class_names": [label_names[index] for index in range(classes)],
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    caches = {split: feature_cache_path(args.output_dir, split) for split in ("train", "val", "test")}
    if args.refresh_features or not all(path.is_file() for path in caches.values()):
        from src.dermfm_provider import DermFMZeroProvider

        provider = DermFMZeroProvider(repo_dir=args.dermfm_repo, device=str(device))
        provider.load()
        preprocess = build_linear_probe_preprocess()
        for split in ("train", "val", "test"):
            if args.refresh_features or not caches[split].is_file():
                extract_features(
                    rows,
                    split,
                    provider,
                    preprocess,
                    device,
                    args.output_dir,
                    args.batch_size,
                    args.workers,
                )
        del provider
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payloads = {
        split: torch.load(path, map_location="cpu", weights_only=False)
        for split, path in caches.items()
    }
    candidates: list[tuple[dict[str, torch.Tensor], dict[str, Any]]] = []
    all_history: list[dict[str, Any]] = []
    for weight_mode in ("none", "sqrt", "balanced"):
        for seed in args.seeds:
            LOGGER.info("candidate started seed=%d weight_mode=%s", seed, weight_mode)
            state, report, history = train_candidate(
                payloads["train"],
                payloads["val"],
                classes,
                seed,
                weight_mode,
                device,
                args.epochs,
                args.patience,
                args.lr,
                args.weight_decay,
            )
            candidates.append((state, report))
            all_history.extend(history)
            LOGGER.info(
                "candidate completed seed=%d weight_mode=%s best_epoch=%d "
                "val_macro_f1=%.4f val_accuracy=%.4f",
                seed,
                weight_mode,
                report["best_epoch"],
                report["macro_f1"],
                report["accuracy"],
            )
    write_csv(args.output_dir / "epoch_history.csv", all_history)
    candidate_rows = [
        {
            "seed": report["seed"],
            "weight_mode": report["weight_mode"],
            "best_epoch": report["best_epoch"],
            "val_accuracy": report["accuracy"],
            "val_balanced_accuracy": report["balanced_accuracy"],
            "val_macro_f1": report["macro_f1"],
        }
        for _, report in candidates
    ]
    write_csv(args.output_dir / "candidate_summary.csv", candidate_rows)
    best_state, validation_report = max(candidates, key=lambda item: item[1]["macro_f1"])
    head = nn.Linear(int(payloads["train"]["features"].shape[1]), classes)
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        validation_logits = head(payloads["val"]["features"])
        test_logits = head(payloads["test"]["features"])
    temperature, validation_nll = calibrate_temperature(
        validation_logits, payloads["val"]["labels"]
    )
    test_report = metrics(test_logits / temperature, payloads["test"]["labels"], classes)
    calibrated_probabilities = torch.softmax(test_logits / temperature, dim=1)
    confidence = torch.softmax(validation_logits / temperature, dim=1).max(dim=1).values
    correct = validation_logits.argmax(dim=1).eq(payloads["val"]["labels"])
    abstention_threshold = float(torch.quantile(confidence[correct], 0.1)) if correct.any() else 0.5
    class_names = [label_names[index] for index in range(classes)]
    test_predictions: list[dict[str, Any]] = []
    for index, (path, target) in enumerate(
        zip(payloads["test"]["paths"], payloads["test"]["labels"].tolist())
    ):
        probability = calibrated_probabilities[index]
        predicted = int(probability.argmax())
        row: dict[str, Any] = {
            "image_path": path,
            "target_index": target,
            "target_name": class_names[target],
            "predicted_index": predicted,
            "predicted_name": class_names[predicted],
            "confidence": float(probability[predicted]),
            "correct": int(predicted == target),
        }
        row.update(
            {f"probability_{name}": float(probability[class_index]) for class_index, name in enumerate(class_names)}
        )
        test_predictions.append(row)
    write_csv(args.output_dir / "test_predictions.csv", test_predictions)
    confusion_rows = []
    for target_index, matrix_row in enumerate(test_report["confusion_matrix"]):
        confusion_rows.append(
            {"target": class_names[target_index], **dict(zip(class_names, matrix_row))}
        )
    write_csv(args.output_dir / "confusion_matrix.csv", confusion_rows)
    per_class_rows = [
        {
            "class_index": index,
            "class_name": name,
            "precision": test_report["per_class_precision"][index],
            "recall": test_report["per_class_recall"][index],
            "f1": test_report["per_class_f1"][index],
            "support": test_report["support"][index],
        }
        for index, name in enumerate(class_names)
    ]
    write_csv(args.output_dir / "per_class_metrics.csv", per_class_rows)
    checkpoint = {
        "format_version": 1,
        "head_type": "linear_probe",
        "dataset": "HAM-official-7-lp",
        "modality": "dermoscopic",
        "feature_dim": int(payloads["train"]["features"].shape[1]),
        "class_names": class_names,
        "state_dict": best_state,
        "temperature": temperature,
        "abstention_threshold": abstention_threshold,
        "validation_selection": validation_report,
        "test_metrics": test_report,
        "split_counts": dict(split_counts),
        "base_checkpoint": str(ROOT / "weights/DermFM-Zero-Open/open_clip_pytorch_model.bin"),
        "base_checkpoint_frozen": True,
        "preprocess": {
            "resize": 256,
            "center_crop": 224,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "source": "DermFM-Zero linear_probe/models/builder.py",
        },
    }
    checkpoint_path = args.output_dir / "ham7_linear_head.pt"
    torch.save(checkpoint, checkpoint_path)
    summary = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"state_dict"}
    }
    summary["checkpoint_path"] = str(checkpoint_path)
    summary["validation_nll_after_temperature_scaling"] = validation_nll
    summary["all_validation_candidates"] = [report for _, report in candidates]
    summary_path = args.output_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info(
        "selected seed=%d weight_mode=%s epoch=%d",
        validation_report["seed"],
        validation_report["weight_mode"],
        validation_report["best_epoch"],
    )
    LOGGER.info(
        "test accuracy=%.4f balanced_accuracy=%.4f macro_f1=%.4f",
        test_report["accuracy"],
        test_report["balanced_accuracy"],
        test_report["macro_f1"],
    )
    LOGGER.info("checkpoint=%s", checkpoint_path)
    LOGGER.info("metrics=%s", summary_path)
    LOGGER.info("training log=%s", log_path)


if __name__ == "__main__":
    main()
