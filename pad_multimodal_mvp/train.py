"""Training/evaluation loop for the three PAD multimodal MVP modes."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader

from .core import (
    DEFAULT_CHECKPOINT,
    VALID_MODES,
    DermFMPADClassifier,
    load_dermfm_components,
    write_json,
)
from .data import PADView, infer_label_names, patient_overlap_summary, validate_split_csv


ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("pad_multimodal_mvp")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_meta = ROOT / "data" / "pad_ufes20" / "meta"
    parser.add_argument("--train-csv", type=Path, default=default_meta / "train.csv")
    parser.add_argument("--val-csv", type=Path, default=default_meta / "val.csv")
    parser.add_argument("--test-csv", type=Path, default=default_meta / "test.csv")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "training" / "pad_multimodal_mvp",
    )
    parser.add_argument(
        "--mode", choices=["all", *VALID_MODES], default="all",
        help="Run one ablation or all three with the identical official split.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--fusion-depth", type=int, default=2)
    parser.add_argument("--fusion-heads", type=int, default=8)
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable. Check nvidia-smi/driver before starting training."
        )
    return torch.device(requested)


def resolve_precision(requested: str, device: torch.device) -> str:
    if device.type != "cuda":
        return "fp32"
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        LOGGER.warning("GPU does not support bf16; falling back to fp16")
        return "fp16"
    return requested


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision in {"bf16", "fp16"}:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def configure_logger(output_dir: Path | None = None) -> None:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)
    if output_dir is not None:
        file_handler = logging.FileHandler(
            output_dir / "training.log", mode="w", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
    LOGGER.addHandler(file_handler)


def make_loaders(args: argparse.Namespace, tokenizer: Any, device: torch.device):
    frames = {
        "train": validate_split_csv(args.train_csv, "train"),
        "val": validate_split_csv(args.val_csv, "val"),
        "test": validate_split_csv(args.test_csv, "test"),
    }
    overlap = patient_overlap_summary(frames["train"], frames["val"], frames["test"])
    if any(overlap.values()):
        LOGGER.warning(
            "The author-provided split contains repeated patient IDs across sets: %s. "
            "It is preserved unchanged as required.",
            overlap,
        )
    label_names = infer_label_names(frames["train"], frames["val"], frames["test"])
    datasets = {
        split: PADView(getattr(args, f"{split}_csv"), split, tokenizer)
        for split in ("train", "val", "test")
    }
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, drop_last=True, **common),
        "val": DataLoader(datasets["val"], shuffle=False, drop_last=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, drop_last=False, **common),
    }
    return frames, datasets, loaders, label_names


def _move_batch(batch: dict[str, Any], device: torch.device, mode: str):
    image = batch["image"].to(device, non_blocking=True) if mode != "text_only" else None
    tokens = batch["tokens"].to(device, non_blocking=True) if mode != "image_only" else None
    target = batch["target"].to(device, non_blocking=True)
    return image, tokens, target


def calculate_metrics(targets: list[int], predictions: list[int], label_names: dict[int, str]):
    labels = list(range(len(label_names)))
    per_class = f1_score(targets, predictions, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {
            label_names[index]: float(per_class[index]) for index in labels
        },
    }


@torch.no_grad()
def evaluate(
    model: DermFMPADClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    precision: str,
    label_names: dict[int, str],
    include_rows: bool = False,
):
    model.eval()
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    rows: list[dict[str, Any]] = []
    for batch in loader:
        image, tokens, target = _move_batch(batch, device, model.mode)
        with autocast_context(device, precision):
            logits = model(image, tokens)
            loss = criterion(logits, target)
        probabilities = logits.float().softmax(dim=1).cpu()
        prediction = probabilities.argmax(dim=1)
        target_cpu = target.cpu()
        total_loss += float(loss.item()) * len(target_cpu)
        targets.extend(target_cpu.tolist())
        predictions.extend(prediction.tolist())
        if include_rows:
            for item_index in range(len(target_cpu)):
                row = {
                    "image_path": batch["path"][item_index],
                    "prompt": batch["prompt"][item_index],
                    "target": int(target_cpu[item_index]),
                    "target_name": label_names[int(target_cpu[item_index])],
                    "prediction": int(prediction[item_index]),
                    "prediction_name": label_names[int(prediction[item_index])],
                }
                for class_index, class_name in label_names.items():
                    row[f"prob_{class_name}"] = float(probabilities[item_index, class_index])
                rows.append(row)
    result = calculate_metrics(targets, predictions, label_names)
    result["loss"] = total_loss / max(len(targets), 1)
    result["samples"] = len(targets)
    result["confusion_matrix"] = confusion_matrix(
        targets, predictions, labels=list(range(len(label_names)))
    ).tolist()
    return result, rows


def save_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to save empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_plot(matrix: list[list[int]], label_names: dict[int, str], path: Path) -> None:
    plt.figure(figsize=(7.5, 6.5))
    sns.heatmap(
        np.asarray(matrix),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[label_names[i] for i in range(len(label_names))],
        yticklabels=[label_names[i] for i in range(len(label_names))],
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("PAD-UFES-20 Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_comparison_table(summary: dict[str, dict[str, Any]], output_dir: Path) -> None:
    """Save the final test metrics of all requested modes."""
    rows = []
    for mode, result in summary.items():
        test = result["test"]
        rows.append(
            {
                "mode": mode,
                "best_epoch": result["best_epoch"],
                "accuracy": test["accuracy"],
                "balanced_accuracy": test["balanced_accuracy"],
                "macro_f1": test["macro_f1"],
            }
        )
    save_csv(output_dir / "comparison.csv", rows)


def run_mode(
    mode: str,
    args: argparse.Namespace,
    encoders: tuple[nn.Module, nn.Module],
    loaders: dict[str, DataLoader],
    label_names: dict[int, str],
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    mode_dir = args.output_dir / mode
    configure_logger(mode_dir)
    set_seed(args.seed)
    model = DermFMPADClassifier(
        vision_encoder=encoders[0],
        text_encoder=encoders[1],
        mode=mode,
        num_classes=len(label_names),
        fusion_depth=args.fusion_depth,
        fusion_heads=args.fusion_heads,
    ).to(device)
    model.assert_only_adapter_is_trainable()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info(
        "mode=%s device=%s precision=%s trainable=%d total=%d effective_batch=%d",
        mode,
        device,
        precision,
        trainable_count,
        total_count,
        args.batch_size * args.accumulation_steps,
    )
    LOGGER.info("trainable modules: %s", model.trainable_parameter_names())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == "fp16"
    )
    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    config.update(
        {
            "resolved_mode": mode,
            "resolved_device": str(device),
            "resolved_precision": precision,
            "trainable_parameters": trainable_count,
            "total_parameters": total_count,
        }
    )
    write_json(mode_dir / "config.json", config)
    write_json(mode_dir / "label_mapping.json", label_names)

    history: list[dict[str, Any]] = []
    best_macro_f1 = -math.inf
    best_epoch = 0
    checkpoint_path = mode_dir / "best.pt"
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen = 0
        for step, batch in enumerate(loaders["train"], start=1):
            image, tokens, target = _move_batch(batch, device, mode)
            with autocast_context(device, precision):
                logits = model(image, tokens)
                raw_loss = criterion(logits, target)
                loss = raw_loss / args.accumulation_steps
            scaler.scale(loss).backward()
            if step % args.accumulation_steps == 0 or step == len(loaders["train"]):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(raw_loss.item()) * len(target)
            seen += len(target)

        validation, _ = evaluate(
            model, loaders["val"], criterion, device, precision, label_names
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "val_balanced_accuracy": validation["balanced_accuracy"],
            "val_macro_f1": validation["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(epoch_row)
        LOGGER.info(
            "epoch=%d/%d train_loss=%.5f val_loss=%.5f val_acc=%.4f val_bal_acc=%.4f val_macro_f1=%.4f",
            epoch,
            args.epochs,
            epoch_row["train_loss"],
            validation["loss"],
            validation["accuracy"],
            validation["balanced_accuracy"],
            validation["macro_f1"],
        )
        if validation["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation["macro_f1"]
            best_epoch = epoch
            payload = model.checkpoint_payload(label_names, config)
            payload["best_epoch"] = best_epoch
            payload["best_val_metrics"] = validation
            torch.save(payload, checkpoint_path)
        scheduler.step()

    save_csv(mode_dir / "training_log.csv", history)
    payload = model.load_trainable_checkpoint(checkpoint_path)
    model.to(device)
    test_metrics, prediction_rows = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        precision,
        label_names,
        include_rows=True,
    )
    final_metrics = {
        "mode": mode,
        "best_epoch": payload["best_epoch"],
        "validation": payload["best_val_metrics"],
        "test": test_metrics,
    }
    write_json(mode_dir / "metrics.json", final_metrics)
    save_csv(
        mode_dir / "per_class_f1.csv",
        [
            {"class_id": index, "class_name": name, "f1": test_metrics["per_class_f1"][name]}
            for index, name in label_names.items()
        ],
    )
    save_csv(mode_dir / "predictions.csv", prediction_rows)
    save_confusion_plot(
        test_metrics["confusion_matrix"], label_names, mode_dir / "confusion_matrix.png"
    )
    LOGGER.info(
        "test mode=%s accuracy=%.4f balanced_accuracy=%.4f macro_f1=%.4f best_epoch=%d",
        mode,
        test_metrics["accuracy"],
        test_metrics["balanced_accuracy"],
        test_metrics["macro_f1"],
        best_epoch,
    )
    return final_metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.epochs < 1 or args.batch_size < 2 or args.accumulation_steps < 1:
        raise ValueError("epochs>=1, batch_size>=2 and accumulation_steps>=1 are required")
    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    set_seed(args.seed)
    configure_logger()
    vision, text, tokenizer = load_dermfm_components(args.checkpoint)
    frames, _, loaders, label_names = make_loaders(args, tokenizer, device)
    overlap = patient_overlap_summary(frames["train"], frames["val"], frames["test"])
    write_json(
        args.output_dir / "dataset_audit.json",
        {
            "split_source": "DermFM-Zero author-provided train.csv/val.csv/test.csv",
            "split_sizes": {name: len(frame) for name, frame in frames.items()},
            "class_counts": {
                name: {
                    str(key): int(value)
                    for key, value in frame["target"].value_counts().sort_index().items()
                }
                for name, frame in frames.items()
            },
            "label_mapping": label_names,
            "cross_split_patient_id_overlap": overlap,
            "split_membership_modified": False,
        },
    )
    del frames
    modes = list(VALID_MODES) if args.mode == "all" else [args.mode]
    summary = {}
    for mode in modes:
        summary[mode] = run_mode(
            mode, args, (vision, text), loaders, label_names, device, precision
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "comparison.json", summary)
    save_comparison_table(summary, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
