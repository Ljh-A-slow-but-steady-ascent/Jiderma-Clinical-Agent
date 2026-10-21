from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from uuid import uuid4

from backend.app.database import CaseRepository


ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


class CaseService:
    def __init__(self, repository: CaseRepository | None = None, data_root: Path | None = None) -> None:
        self.repository = repository or CaseRepository()
        self.data_root = data_root or Path(__file__).resolve().parents[3] / "data" / "cases"

    def create(
        self,
        image_bytes: bytes,
        content_type: str,
        mode: str = "patient",
        retention_mode: str = "session",
    ) -> str:
        if content_type not in ALLOWED_IMAGE_TYPES:
            raise ValueError("only JPEG, PNG, and WebP images are supported")
        if not image_bytes:
            raise ValueError("image is empty")
        if len(image_bytes) > 10 * 1024 * 1024:
            raise ValueError("image exceeds 10 MB")
        if mode not in {"patient", "clinician"}:
            raise ValueError("mode must be patient or clinician")
        if retention_mode not in {"session", "local"}:
            raise ValueError("retention_mode must be session or local")
        case_id = f"CASE_{uuid4().hex[:12]}"
        case_dir = self.data_root / case_id
        case_dir.mkdir(parents=True, exist_ok=False)
        image_path = case_dir / f"lesion{ALLOWED_IMAGE_TYPES[content_type]}"
        image_path.write_bytes(image_bytes)
        self.repository.create_case(
            case_id=case_id,
            local_image_path=str(image_path.resolve()),
            image_sha256=hashlib.sha256(image_bytes).hexdigest(),
            retention_mode=retention_mode,
            mode=mode,
        )
        return case_id

    def get_local_image_path(self, case_id: str) -> Path:
        record = self.repository.get_case(case_id)
        if record is None:
            raise KeyError(f"case not found: {case_id}")
        return Path(record.local_image_path)

    def delete(self, case_id: str, user_id: int) -> None:
        """Delete an owned case and remove only its validated local asset directory."""
        image_path = self.repository.delete_case_record(case_id, user_id=user_id)
        case_dir = Path(image_path).resolve().parent
        root = self.data_root.resolve()
        if case_dir != root and root in case_dir.parents:
            shutil.rmtree(case_dir, ignore_errors=True)
