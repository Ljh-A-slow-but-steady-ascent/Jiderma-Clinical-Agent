from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from backend.app.services.case_service import ALLOWED_IMAGE_TYPES


@dataclass(frozen=True)
class StagedAsset:
    asset_id: str
    path: Path
    content_type: str
    sha256: str
    size: int


class LocalAssetStore:
    """Short-lived local staging for ChatBox uploads; never exposed to an LLM."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[3] / "data" / "agent_uploads"

    def stage(self, content: bytes, content_type: str) -> StagedAsset:
        if content_type not in ALLOWED_IMAGE_TYPES:
            raise ValueError("仅支持 JPG、PNG 和 WebP 图片")
        if not content:
            raise ValueError("上传图片为空")
        if len(content) > 10 * 1024 * 1024:
            raise ValueError("图片不能超过10MB")
        self.root.mkdir(parents=True, exist_ok=True)
        asset_id = f"ASSET_{uuid4().hex[:12]}"
        path = self.root / f"{asset_id}{ALLOWED_IMAGE_TYPES[content_type]}"
        path.write_bytes(content)
        return StagedAsset(
            asset_id=asset_id,
            path=path,
            content_type=content_type,
            sha256=hashlib.sha256(content).hexdigest(),
            size=len(content),
        )

    @staticmethod
    def discard(path: str | Path | None) -> None:
        if not path:
            return
        candidate = Path(path)
        if candidate.is_file() and candidate.parent.name == "agent_uploads":
            candidate.unlink()

