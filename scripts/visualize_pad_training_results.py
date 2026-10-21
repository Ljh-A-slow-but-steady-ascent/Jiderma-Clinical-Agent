"""Create clean English figures from the completed PAD multimodal run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "training" / "pad_multimodal_mvp"
MODES = {
    "image_only": "Image Only",
    "text_only": "Clinical Text Only",
    "image_text": "Image + Clinical Text",
}
COLORS = {
    "image_only": "#3973B7",
    "text_only": "#D18A32",
    "image_text": "#16836F",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_results():
    history = {
        mode: pd.read_csv(RESULTS / mode / "training_log.csv") for mode in MODES
    }
    metrics = json.loads((RESULTS / "comparison.json").read_text(encoding="utf-8"))
    return history, metrics


def plot_training_curves(history: dict[str, pd.DataFrame], metrics: dict) -> Path:
    panels = (
        ("train_loss", "Training Loss", None),
        ("val_loss", "Validation Loss", None),
        ("val_macro_f1", "Validation Macro-F1", (0, 1)),
        ("val_balanced_accuracy", "Validation Balanced Accuracy", (0, 1)),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
    for axis, (column, title, limits) in zip(axes.flat, panels):
        for mode, label in MODES.items():
            frame = history[mode]
            axis.plot(
                frame["epoch"], frame[column], label=label,
                color=COLORS[mode], linewidth=2.15,
            )
            if column in {"val_macro_f1", "val_balanced_accuracy"}:
                best_epoch = int(metrics[mode]["best_epoch"])
                best_row = frame.loc[frame["epoch"] == best_epoch].iloc[0]
                axis.scatter(
                    [best_epoch], [best_row[column]], s=42,
                    color=COLORS[mode], edgecolor="white", linewidth=1.1, zorder=4,
                )
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.set_xlabel("Epoch")
        if limits:
            axis.set_ylim(*limits)
        axis.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.76, 1.035))
    fig.suptitle(
        "PAD-UFES-20 Training Dynamics",
        fontsize=17, fontweight="bold", x=0.02, ha="left", y=1.035,
    )
    output = RESULTS / "training_curves.png"
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_comparison(metrics: dict) -> Path:
    metric_keys = ("accuracy", "balanced_accuracy", "macro_f1")
    metric_labels = ("Accuracy", "Balanced Accuracy", "Macro-F1")
    class_names = ("NEV", "BCC", "ACK", "SEK", "SCC", "MEL")
    modes = list(MODES)

    fig, axes = plt.subplots(
        1, 2, figsize=(14.5, 5.8), gridspec_kw={"width_ratios": [1, 1.35]},
        constrained_layout=True,
    )

    x = np.arange(len(metric_keys))
    width = 0.24
    for index, mode in enumerate(modes):
        values = [metrics[mode]["test"][key] for key in metric_keys]
        bars = axes[0].bar(
            x + (index - 1) * width, values, width,
            color=COLORS[mode], label=MODES[mode],
        )
        axes[0].bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
    axes[0].set_xticks(x, metric_labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Test Score")
    axes[0].set_title("Overall Performance", loc="left", fontweight="semibold")
    axes[0].spines[["top", "right"]].set_visible(False)

    x = np.arange(len(class_names))
    for index, mode in enumerate(modes):
        values = [metrics[mode]["test"]["per_class_f1"][name] for name in class_names]
        axes[1].bar(
            x + (index - 1) * width, values, width,
            color=COLORS[mode], label=MODES[mode],
        )
    axes[1].set_xticks(x, class_names)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Per-Class F1", loc="left", fontweight="semibold")
    axes[1].spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.76, 1.04))
    fig.suptitle(
        "PAD-UFES-20 Modality Comparison",
        fontsize=17, fontweight="bold", x=0.015, ha="left", y=1.04,
    )
    output = RESULTS / "comparison_detailed.png"
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    configure_style()
    history, metrics = load_results()
    for output in (plot_training_curves(history, metrics), plot_comparison(metrics)):
        print(output)


if __name__ == "__main__":
    main()
