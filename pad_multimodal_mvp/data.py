"""PAD-UFES-20 data validation around the official DermFM PAD Dataset."""

from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import Dataset

from .core import DERMFM_MM, normalise_prompt_frame


if str(DERMFM_MM) not in sys.path:
    sys.path.insert(0, str(DERMFM_MM))

# DermFM-Zero-Open was trained with symmetric CLIP normalisation. The official
# dataloader exposes these environment switches but otherwise falls back to its
# legacy ImageNet values.
os.environ.setdefault("PANDERM_NORM_MEAN", "0.5,0.5,0.5")
os.environ.setdefault("PANDERM_NORM_STD", "0.5,0.5,0.5")

from dataset.prompt import PADPromptBuilder  # noqa: E402

# The upstream file lives in a directory named ``src`` without an
# ``__init__.py``.  This project already has its own ``src`` package, so
# ``from src.dataloader`` is order-dependent and fails from a clean process.
# Load the unchanged vendor file by its absolute path instead of shadowing the
# application's package namespace.
_DATALOADER_PATH = DERMFM_MM / "src" / "dataloader.py"
_DATALOADER_SPEC = importlib.util.spec_from_file_location(
    "_dermfm_zero_pad_dataloader", _DATALOADER_PATH
)
if _DATALOADER_SPEC is None or _DATALOADER_SPEC.loader is None:
    raise ImportError(f"Cannot load official DermFM dataloader: {_DATALOADER_PATH}")
_DATALOADER_MODULE = importlib.util.module_from_spec(_DATALOADER_SPEC)
_DATALOADER_SPEC.loader.exec_module(_DATALOADER_MODULE)
PAD = _DATALOADER_MODULE.PAD


REQUIRED_COLUMNS = {
    "clinical_path",
    "target",
    "age",
    "gender",
    "region",
    "itch",
    "hurt",
    "bleed",
    "grew",
    "changed",
    "diameter_1",
    "diameter_2",
    "skin_cancer_history",
    "cancer_history",
}


def validate_split_csv(path: Path, split: str) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Official PAD {split} split is missing: {path}")
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required PAD columns: {missing}")
    if frame.empty:
        raise ValueError(f"PAD {split} split is empty: {path}")
    labels = sorted(frame["target"].dropna().astype(int).unique().tolist())
    if not set(labels).issubset(set(range(6))):
        raise ValueError(f"PAD {split} contains invalid class ids: {labels}")
    missing_images = [p for p in frame["clinical_path"].astype(str) if not Path(p).is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"PAD {split} references {len(missing_images)} missing images; first: {missing_images[0]}"
        )
    return frame


def infer_label_names(*frames: pd.DataFrame) -> dict[int, str]:
    merged = pd.concat(frames, ignore_index=True)
    name_column = next(
        (column for column in ("diagnostic", "diag", "class_name") if column in merged),
        None,
    )
    names: dict[int, str] = {}
    if name_column:
        for label, group in merged.groupby("target"):
            values = sorted({str(value).strip() for value in group[name_column] if pd.notna(value)})
            if len(values) != 1:
                raise ValueError(f"target={label} maps to inconsistent diagnoses: {values}")
            names[int(label)] = values[0]
    else:
        from .core import PAD_CLASS_NAMES

        names = dict(PAD_CLASS_NAMES)
    if sorted(names) != list(range(6)):
        raise ValueError(f"Expected six contiguous PAD labels, got {names}")
    return names


def patient_overlap_summary(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> dict[str, int]:
    if "patient_id" not in train or "patient_id" not in validation or "patient_id" not in test:
        return {}
    sets = {
        "train": set(train["patient_id"].dropna().astype(str)),
        "val": set(validation["patient_id"].dropna().astype(str)),
        "test": set(test["patient_id"].dropna().astype(str)),
    }
    return {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }


class PADView(Dataset):
    """A narrow view of the official PAD dataset with paths/prompts retained."""

    def __init__(self, csv_path: Path, split: str, tokenizer: Any) -> None:
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.frame = validate_split_csv(self.csv_path, split)
        # Keep official augmentation, image loading, tokenisation and labels.
        self.tokenizer = tokenizer
        self.official = PAD(
            df_path=str(self.csv_path),
            shape=(224, 224),
            mode="train" if split == "train" else "test",
            tokenizer=None,
        )
        prompt_frame = normalise_prompt_frame(self.frame)
        builder = PADPromptBuilder(prompt_frame)
        self.prompts = [builder.build_patient_description(i) for i in range(len(prompt_frame))]

    def __len__(self) -> int:
        return len(self.official)

    def __getitem__(self, index: int):
        path, _, _, image, _, target = self.official[index]
        # Upstream loads with cv2 (BGR). Since all three normalisation channels
        # use identical 0.5 statistics, swapping after normalisation is exactly
        # equivalent to converting the source image to RGB before normalising.
        image = image[[2, 1, 0], ...]
        tokens = self.tokenizer([self.prompts[index]])[0]
        return {
            "index": index,
            "path": str(path),
            "image": image,
            "tokens": tokens.long(),
            "target": target.squeeze(0).long(),
            "prompt": self.prompts[index],
        }
