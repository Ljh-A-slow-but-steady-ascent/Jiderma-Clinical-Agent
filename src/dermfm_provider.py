"""Load the frozen vision encoder used by the archived HAM7 linear-head task."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class DermFMZeroProvider:
    """Minimal checkpoint loader; classification stays in the training script."""

    ARCHITECTURE_NAME = "PanDerm-large-v2-w-PubMed-256"

    def __init__(
        self,
        repo_dir: Path | None = None,
        checkpoint_path: Path | None = None,
        device: str = "auto",
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        self.repo_dir = Path(repo_dir or root / "third_party" / "DermFM-Zero").resolve()
        self.checkpoint_path = Path(
            checkpoint_path
            or root / "weights" / "DermFM-Zero-Open" / "open_clip_pytorch_model.bin"
        ).resolve()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        self.device = torch.device(device)
        self.model = None

    def _bundled_open_clip(self):
        src_dir = (self.repo_dir / "src").resolve()
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Bundled source directory not found: {src_dir}")
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        import open_clip  # pylint: disable=import-outside-toplevel

        module_path = Path(open_clip.__file__).resolve()
        if src_dir not in module_path.parents:
            raise RuntimeError(
                f"Refusing non-bundled open_clip import: {module_path}; expected under {src_dir}"
            )
        return open_clip

    def load(self) -> "DermFMZeroProvider":
        if self.model is not None:
            return self
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Permanent checkpoint not found: {self.checkpoint_path}")

        open_clip = self._bundled_open_clip()
        model, _, _ = open_clip.create_model_and_transforms(
            self.ARCHITECTURE_NAME,
            device=self.device,
            pretrained_hf=False,
        )
        open_clip.utils.load_dermfm_checkpoint(model, str(self.checkpoint_path))
        model.eval().requires_grad_(False)
        self.model = model
        return self
