#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DERMAGUARD_DERMFM_PYTHON:-/home/lu/miniforge3/envs/dermfm-zero/bin/python}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -m pad_multimodal_mvp.inference "$@"
