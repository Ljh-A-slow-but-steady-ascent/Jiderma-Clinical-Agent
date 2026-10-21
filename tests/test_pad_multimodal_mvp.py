from __future__ import annotations

import torch
from torch import nn

from pad_multimodal_mvp.core import DermFMPADClassifier, build_pad_prompt


class DummyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.ones(image.shape[0], 1024, device=image.device)


class DummyText(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.ones(tokens.shape[0], 768, device=tokens.device)


def test_official_pad_prompt_contains_required_positive_findings() -> None:
    prompt = build_pad_prompt(
        {
            "age": 54,
            "gender": "FEMALE",
            "region": "FOREARM",
            "itch": False,
            "hurt": True,
            "bleed": True,
            "grew": True,
            "changed": True,
            "diameter_1": 8,
            "diameter_2": 6,
            "skin_cancer_history": False,
            "cancer_history": True,
        }
    )
    assert "54-year-old" in prompt
    assert "forearm" in prompt
    assert "8mm" in prompt
    assert "painful" in prompt
    assert "bleeding" in prompt
    assert "growing" in prompt
    assert "changing" in prompt
    assert "family history of cancer" in prompt
    assert "skin cancer" not in prompt


def test_all_modes_keep_encoders_frozen_and_return_six_logits() -> None:
    image = torch.randn(2, 3, 224, 224)
    tokens = torch.ones(2, 16, dtype=torch.long)
    for mode in ("image_only", "text_only", "image_text"):
        model = DermFMPADClassifier(DummyVision(), DummyText(), mode=mode)
        model.assert_only_adapter_is_trainable()
        assert not any(parameter.requires_grad for parameter in model.vision_encoder.parameters())
        assert not any(parameter.requires_grad for parameter in model.text_encoder.parameters())
        output = model(
            image if mode != "text_only" else None,
            tokens if mode != "image_only" else None,
        )
        assert output.shape == (2, 6)
