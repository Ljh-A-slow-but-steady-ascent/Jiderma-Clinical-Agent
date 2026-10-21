"""Persistent JSON-lines worker for the permanent local medical embedding checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedTokenizerFast


ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = Path(os.getenv("DERMAGUARD_EMBEDDING_PATH", ROOT / "weights/rag/IlhaEmbed"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = _tokenizer = None
_document_signature: str | None = None
_document_vectors: torch.Tensor | None = None


def _load() -> None:
    global _model, _tokenizer
    if _model is None:
        # The checkpoint declares the newer TokenizersBackend wrapper. Build the
        # equivalent fast tokenizer explicitly for compatibility with the
        # project's pinned Transformers runtime.
        _tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(MODEL_PATH / "tokenizer.json"),
            bos_token="<|startoftext|>", cls_token="<|startoftext|>",
            eos_token="<|return|>", sep_token="<|return|>",
            pad_token="<|endoftext|>", mask_token="[MASK]",
        )
        _model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE).eval()


def _pool(output: object, attention_mask: torch.Tensor) -> torch.Tensor:
    hidden = output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)


def encode(texts: list[str]) -> torch.Tensor:
    _load()
    batches = []
    for start in range(0, len(texts), 12):
        encoded = _tokenizer(
            texts[start:start + 12], padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        ).to(DEVICE)
        encoded.pop("token_type_ids", None)
        with torch.inference_mode():
            output = _model(**encoded)
            pooled = F.normalize(_pool(output, encoded["attention_mask"]), p=2, dim=1)
        batches.append(pooled.cpu())
    return torch.cat(batches)


def _signature(documents: list[str]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def prepare_documents(documents: list[str]) -> torch.Tensor:
    """Encode the stable guideline corpus once per worker process."""
    global _document_signature, _document_vectors
    signature = _signature(documents)
    if _document_vectors is None or signature != _document_signature:
        _document_vectors = encode(documents)
        _document_signature = signature
    return _document_vectors


def similarity(query: str, documents: list[str]) -> list[float]:
    document_vectors = prepare_documents(documents)
    query_vector = encode([query])[0]
    return (document_vectors @ query_vector).tolist()


def main() -> None:
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            if payload["action"] not in {"prepare", "dense"}:
                raise ValueError(f"unknown action: {payload['action']}")
            documents = list(payload["documents"])
            if payload["action"] == "prepare":
                prepare_documents(documents)
                result = {"prepared": len(documents), "device": str(DEVICE)}
            else:
                result = {"scores": similarity(str(payload["query"]), documents)}
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
