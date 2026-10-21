"""Minimal PAD-UFES-20 multimodal classifier built from DermFM-Zero components."""

from .core import (
    PAD_CLASS_NAMES,
    DermFMPADClassifier,
    build_pad_prompt,
    load_dermfm_components,
)

__all__ = [
    "PAD_CLASS_NAMES",
    "DermFMPADClassifier",
    "build_pad_prompt",
    "load_dermfm_components",
]
