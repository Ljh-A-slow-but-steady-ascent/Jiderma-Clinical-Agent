#!/usr/bin/env python3
"""Extract only PAD-UFES-20 from the official DermFM multimodal ZIP.

The author-provided CSV files are retained byte-for-byte under ``official/meta``.
Working copies only replace ``clinical_path`` with local absolute paths; row
order, labels and train/val/test membership are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREFIX = Path("multimodal_finetune/PAD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "pad_ufes20")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    archive = args.archive.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    images_dir = output / "images"
    official_meta = output / "official" / "meta"
    working_meta = output / "meta"
    images_dir.mkdir(parents=True, exist_ok=True)
    official_meta.mkdir(parents=True, exist_ok=True)
    working_meta.mkdir(parents=True, exist_ok=True)

    extracted = 0
    with zipfile.ZipFile(archive) as bundle:
        members = [
            member
            for member in bundle.infolist()
            if Path(member.filename).is_relative_to(PREFIX) and not member.is_dir()
        ]
        for member in members:
            relative = Path(member.filename).relative_to(PREFIX)
            if relative.parts[0] not in {"images", "meta"}:
                continue
            destination = images_dir / relative.name if relative.parts[0] == "images" else official_meta / relative.name
            with bundle.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            extracted += 1

    audit = {"archive": str(archive), "extracted_files": extracted, "splits": {}}
    for split in ("train", "val", "test"):
        source = official_meta / f"{split}.csv"
        frame = pd.read_csv(source)
        frame["clinical_path"] = frame["img_id"].map(lambda name: str((images_dir / name).resolve()))
        target = working_meta / f"{split}.csv"
        frame.to_csv(target, index=False)
        missing = int((~frame["clinical_path"].map(lambda value: Path(value).is_file())).sum())
        if missing:
            raise FileNotFoundError(f"{split} has {missing} missing images after extraction")
        audit["splits"][split] = {
            "rows": len(frame),
            "official_csv_sha256": sha256(source),
            "working_csv_sha256": sha256(target),
            "target_counts": {
                str(key): int(value)
                for key, value in frame["target"].value_counts().sort_index().items()
            },
        }
    (output / "split_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
