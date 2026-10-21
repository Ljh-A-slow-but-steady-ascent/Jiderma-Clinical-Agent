"""Official-component adapter for the PAD-UFES-20 multimodal MVP.

The vendor tree under ``third_party/DermFM-Zero`` is intentionally not edited.
This module only composes its released image tower, PubMedBERT text tower,
PAD prompt builder and ``CrossTransformer_meta`` fusion block.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DERMFM_REPO = ROOT / "third_party" / "DermFM-Zero"
DERMFM_SRC = DERMFM_REPO / "src"
DERMFM_MM = DERMFM_REPO / "multimodal_finetune"
DEFAULT_CHECKPOINT = ROOT / "weights" / "DermFM-Zero-Open" / "open_clip_pytorch_model.bin"

for path in (DERMFM_SRC, DERMFM_MM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import open_clip  # noqa: E402
from dataset.prompt import PADPromptBuilder  # noqa: E402
from models.PanDermFormer import _DermFMZeroMeanPool  # noqa: E402
from models.mca import CrossTransformer_meta  # noqa: E402
from open_clip.utils import load_dermfm_checkpoint  # noqa: E402


PAD_CLASS_NAMES = {
    0: "NEV",
    1: "BCC",
    2: "ACK",
    3: "SEK",
    4: "SCC",
    5: "MEL",
}
VALID_MODES = ("image_only", "text_only", "image_text")
BOOL_FIELDS = (
    "smoke",
    "drink",
    "pesticide",
    "skin_cancer_history",
    "cancer_history",
    "itch",
    "grew",
    "hurt",
    "changed",
    "bleed",
    "elevation",
)


def _normalise_bool(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    if text in {"unk", "unknown", "nan", "none", "null"}:
        return None
    return value


def normalise_prompt_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise JSON/CSV booleans before calling the unmodified official builder."""
    result = frame.copy()
    for field in BOOL_FIELDS:
        if field in result:
            result[field] = result[field].map(_normalise_bool)
    return result


def build_pad_prompt(metadata: Mapping[str, Any]) -> str:
    frame = normalise_prompt_frame(pd.DataFrame([dict(metadata)]))
    return PADPromptBuilder(frame).build_patient_description(0)


def load_dermfm_components(
    checkpoint: Path = DEFAULT_CHECKPOINT,
) -> tuple[nn.Module, nn.Module, Any]:
    """Load one official DermFM instance and return its two frozen towers.

    The upstream ``PanDermTFormer`` constructs two complete DermFM models for
    image+text. Reusing the two towers from one checkpoint is mathematically
    equivalent while avoiding duplicated parameters on a 16 GB GPU.
    """
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DermFM-Zero checkpoint not found: {checkpoint}")

    full_model, _, _ = open_clip.create_model_and_transforms(
        "PanDerm-large-v2-w-PubMed-256",
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    load_dermfm_checkpoint(full_model, str(checkpoint))
    vision = _DermFMZeroMeanPool(full_model.visual.trunk)
    text = full_model.text
    tokenizer = open_clip.get_tokenizer("PanDerm-large-v2-w-PubMed-256")
    del full_model
    return vision, text, tokenizer


class _OfficialMLPHead(nn.Sequential):
    """The MLP head used by the official multimodal fine-tuning entrypoint."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )


class DermFMPADClassifier(nn.Module):
    """Frozen DermFM image/text encoders plus the official PAD fusion block."""

    image_dim = 1024
    text_dim = 768

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        mode: str,
        num_classes: int = 6,
        fusion_depth: int = 2,
        fusion_heads: int = 8,
    ) -> None:
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.num_classes = num_classes
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False

        self.text_projection = (
            nn.Linear(self.text_dim, self.image_dim)
            if mode in {"text_only", "image_text"}
            else None
        )
        self.fusion = (
            CrossTransformer_meta(
                x_dim=self.image_dim,
                c_dim=self.image_dim,
                depth=fusion_depth,
                num_heads=fusion_heads,
            )
            if mode == "image_text"
            else None
        )
        head_dim = self.image_dim * 2 if mode == "image_text" else self.image_dim
        self.head = _OfficialMLPHead(head_dim, num_classes)
        self._shape_checked = False

    def train(self, mode: bool = True) -> "DermFMPADClassifier":
        super().train(mode)
        # Frozen towers must stay deterministic even while fusion/head train.
        self.vision_encoder.eval()
        self.text_encoder.eval()
        return self

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def assert_only_adapter_is_trainable(self) -> None:
        bad = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
            and not name.startswith(("text_projection.", "fusion.", "head."))
        ]
        if bad:
            raise RuntimeError(f"Frozen encoder parameters unexpectedly trainable: {bad[:5]}")

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feature = self.vision_encoder(image)
        return feature.float()

    def _encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feature = self.text_encoder(tokens)
        return feature.float()

    def forward(
        self,
        image: torch.Tensor | None,
        tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        image_feature = None
        text_feature = None

        if self.mode in {"image_only", "image_text"}:
            if image is None:
                raise ValueError(f"image is required for mode={self.mode}")
            image_feature = self._encode_image(image)
        if self.mode in {"text_only", "image_text"}:
            if tokens is None:
                raise ValueError(f"tokens are required for mode={self.mode}")
            text_feature = self._encode_text(tokens)

        if not self._shape_checked:
            if image_feature is not None and image_feature.shape[-1] != self.image_dim:
                raise RuntimeError(
                    f"Official vision feature is {image_feature.shape[-1]}D; expected {self.image_dim}D"
                )
            if text_feature is not None and text_feature.shape[-1] != self.text_dim:
                raise RuntimeError(
                    f"Official text feature is {text_feature.shape[-1]}D; expected {self.text_dim}D"
                )
            self._shape_checked = True

        if self.mode == "image_only":
            combined = image_feature
        elif self.mode == "text_only":
            combined = self.text_projection(text_feature)
        else:
            projected_text = self.text_projection(text_feature)
            fused_text = self.fusion(projected_text, image_feature)
            combined = torch.cat([image_feature, fused_text], dim=-1)
        return self.head(combined)

    def checkpoint_payload(
        self,
        label_names: Mapping[int, str],
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        trainable = {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if name.startswith(("text_projection.", "fusion.", "head."))
        }
        return {
            "format_version": 1,
            "mode": self.mode,
            "num_classes": self.num_classes,
            "label_names": {str(k): v for k, v in label_names.items()},
            "config": dict(config),
            "trainable_state_dict": trainable,
        }

    def load_trainable_checkpoint(self, checkpoint: Path) -> dict[str, Any]:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("mode") != self.mode:
            raise ValueError(
                f"Checkpoint mode {payload.get('mode')!r} does not match model mode {self.mode!r}"
            )
        missing, unexpected = self.load_state_dict(
            payload["trainable_state_dict"], strict=False
        )
        encoder_missing = [
            name
            for name in missing
            if name.startswith(("vision_encoder.", "text_encoder."))
        ]
        non_encoder_missing = [name for name in missing if name not in encoder_missing]
        if non_encoder_missing or unexpected:
            raise RuntimeError(
                f"Invalid adapter checkpoint: missing={non_encoder_missing}, unexpected={unexpected}"
            )
        return payload


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
