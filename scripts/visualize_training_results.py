#!/usr/bin/env python3
"""Render reproducible training and evaluation figures from saved CSV logs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dermfm-matplotlib")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES_EN = {
    "actinic keratosis": "Actinic keratosis",
    "basal cell carcinoma": "Basal cell carcinoma",
    "benign keratosis": "Benign keratosis",
    "dermatofibroma": "Dermatofibroma",
    "melanoma": "Melanoma",
    "nevus": "Nevus",
    "vascular": "Vascular lesion",
}
WEIGHT_NAMES_EN = {
    "none": "No weighting",
    "sqrt": "Square-root weighting",
    "balanced": "Balanced weighting",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "artifacts/training/ham7_linear_head",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output = args.output or input_dir / "training_evaluation.png"
    history = read_csv(input_dir / "epoch_history.csv")
    candidates = read_csv(input_dir / "candidate_summary.csv")
    class_metrics = read_csv(input_dir / "per_class_metrics.csv")
    confusion_rows = read_csv(input_dir / "confusion_matrix.csv")
    summary = json.loads((input_dir / "metrics.json").read_text(encoding="utf-8"))

    selected = summary["validation_selection"]
    selected_history = [
        row
        for row in history
        if int(row["seed"]) == int(selected["seed"])
        and row["weight_mode"] == selected["weight_mode"]
    ]
    epochs = [int(row["epoch"]) for row in selected_history]
    train_loss = [float(row["train_loss"]) for row in selected_history]
    val_loss = [float(row["val_loss"]) for row in selected_history]

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    loss_axis = axes[0, 0]
    loss_axis.plot(epochs, train_loss, label="Training loss", linewidth=2)
    loss_axis.plot(epochs, val_loss, label="Validation loss", linewidth=2)
    loss_axis.axvline(selected["best_epoch"], color="#d95f02", linestyle="--", label="Selected epoch")
    loss_axis.set(title="Loss Curves of the Selected Model", xlabel="Epoch", ylabel="Cross-entropy loss")
    loss_axis.legend()

    f1_axis = axes[0, 1]
    labels = [f"{WEIGHT_NAMES_EN[row['weight_mode']]}\nSeed {row['seed']}" for row in candidates]
    values = [float(row["val_macro_f1"]) for row in candidates]
    colors = ["#4c78a8" if value < max(values) else "#f58518" for value in values]
    f1_axis.bar(range(len(values)), values, color=colors)
    f1_axis.set_xticks(range(len(values)), labels, rotation=35, ha="right")
    f1_axis.set_ylim(max(0.0, min(values) - 0.05), min(1.0, max(values) + 0.04))
    f1_axis.set(title="Validation Candidate Comparison", ylabel="Macro F1")

    raw_names = [row["class_name"] for row in class_metrics]
    names = [CLASS_NAMES_EN.get(name, name.title()) for name in raw_names]
    class_f1 = [float(row["f1"]) for row in class_metrics]
    recall = [float(row["recall"]) for row in class_metrics]
    metric_axis = axes[1, 0]
    positions = np.arange(len(names))
    metric_axis.bar(positions - 0.2, class_f1, width=0.4, label="F1", color="#54a24b")
    metric_axis.bar(positions + 0.2, recall, width=0.4, label="Recall", color="#e45756")
    metric_axis.set_xticks(positions, names, rotation=35, ha="right")
    metric_axis.set_ylim(0, 1)
    metric_axis.set(title="Per-class Test Performance", ylabel="Score")
    metric_axis.legend()

    matrix = np.asarray(
        [[int(row[name]) for name in raw_names] for row in confusion_rows], dtype=np.int64
    )
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    confusion_axis = axes[1, 1]
    image = confusion_axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(names)):
        for column in range(len(names)):
            confusion_axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if normalized[row, column] > 0.55 else "black",
            )
    confusion_axis.set_xticks(range(len(names)), names, rotation=40, ha="right")
    confusion_axis.set_yticks(range(len(names)), names)
    confusion_axis.set(title="Test Confusion Matrix", xlabel="Predicted class", ylabel="True class")
    figure.colorbar(image, ax=confusion_axis, fraction=0.046, label="Row-normalized proportion")

    test = summary["test_metrics"]
    figure.suptitle(
        "DermFM-Zero HAM7 Frozen-Backbone Linear Probe\n"
        f"Test Accuracy {test['accuracy']:.3f} | Balanced Accuracy {test['balanced_accuracy']:.3f} | "
        f"Macro F1 {test['macro_f1']:.3f}",
        fontsize=16,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
