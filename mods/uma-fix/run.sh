#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$MOD_DIR/uma_fix.patch"
UVA_FALLBACK_PATCH_FILE="$MOD_DIR/wsl-uva-fallback.patch"

apply_patch_once() {
  local name="$1"
  local patch_file="$2"
  shift 2
  local git_apply_args=("$@")

  if [ ! -f "$patch_file" ]; then
    echo "[uma-fix] $name patch not found: $patch_file" >&2
    exit 1
  elif git apply "${git_apply_args[@]}" --reverse --check "$patch_file" 2>/dev/null; then
    echo "[uma-fix] $name patch is already applied; skipping."
  elif git apply "${git_apply_args[@]}" --check "$patch_file"; then
    git apply "${git_apply_args[@]}" "$patch_file"
    echo "[uma-fix] Applied $name patch."
  else
    echo "[uma-fix] $name patch could not be applied to installed vLLM." >&2
    exit 1
  fi
}


patch_memory_call_sites() {
  python3 - "$PYTHON_ROOT" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
import_line = "from vllm.utils.mem_utils import cuda_mem_get_info\n"


def ensure_import(text: str, path: Path) -> tuple[str, bool]:
    if import_line in text:
        return text, False

    lines = text.splitlines(keepends=True)
    insert_at = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            insert_at = i + 1
        elif insert_at is not None and not stripped:
            break

    if insert_at is None:
        print(f"[uma-fix] Could not find import block in {path}.", file=sys.stderr)
        raise SystemExit(1)

    lines.insert(insert_at, import_line)
    return "".join(lines), True


def patch_file(relative_path: str, replacements: tuple[tuple[str, str], ...]) -> None:
    path = root / relative_path
    if not path.exists():
        print(f"[uma-fix] {relative_path} not found; skipping.")
        return

    text = path.read_text()
    original = text

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    if "cuda_mem_get_info" in text:
        text, _ = ensure_import(text, path)

    if text == original:
        if "cuda_mem_get_info" in text:
            print(f"[uma-fix] {relative_path} is already patched; skipping.")
        else:
            print(f"[uma-fix] {relative_path} has no legacy memory calls; skipping.")
        return

    path.write_text(text)
    print(f"[uma-fix] Patched {relative_path}.")


patch_file(
    "vllm/model_executor/models/gemma4_mm.py",
    (
        (r"current_platform\.mem_get_info\(\)", "cuda_mem_get_info()"),
    ),
)
patch_file(
    "vllm/v1/worker/gpu/model_runner.py",
    (
        (
            r"torch\.cuda\.mem_get_info\(\s*\)\[0\]",
            "cuda_mem_get_info(self.device)[0]",
        ),
    ),
)
patch_file(
    "vllm/v1/worker/gpu/spec_decode/eagle/utils.py",
    (
        (
            r"torch\.cuda\.mem_get_info\(\s*w\.device\s*\)\[0\]",
            "cuda_mem_get_info(w.device)[0]",
        ),
        (
            r"torch\.cuda\.mem_get_info\(\s*device\s*=\s*w\.device\s*\)\[0\]",
            "cuda_mem_get_info(w.device)[0]",
        ),
    ),
)
patch_file(
    "vllm/v1/worker/gpu_model_runner.py",
    (
        (
            r"torch\.cuda\.mem_get_info\(\s*\)\[0\]",
            "cuda_mem_get_info(self.device)[0]",
        ),
    ),
)
patch_file(
    "vllm/v1/worker/gpu_worker.py",
    (
        (
            r"torch\.cuda\.mem_get_info\(\s*\)\[0\]",
            "cuda_mem_get_info(self.device)[0]",
        ),
        (
            r"torch\.cuda\.mem_get_info\(\s*\)",
            "cuda_mem_get_info(self.device)",
        ),
    ),
)
PY
}

if ! command -v git >/dev/null 2>&1; then
  echo "[uma-fix] git is required to apply this mod." >&2
  echo "[uma-fix] Apply mods/use-official-vllm first if this container does not include git." >&2
  exit 1
fi

if [ ! -d "$PYTHON_ROOT/vllm" ]; then
  echo "[uma-fix] vLLM package not found at $PYTHON_ROOT/vllm" >&2
  exit 1
fi

cd "$PYTHON_ROOT"

apply_patch_once \
  "UMA memory accounting fix" \
  "$PATCH_FILE" \
  --exclude=vllm/model_executor/models/gemma4_mm.py \
  --exclude=vllm/v1/worker/gpu/model_runner.py \
  --exclude=vllm/v1/worker/gpu/spec_decode/eagle/utils.py \
  --exclude=vllm/v1/worker/gpu_model_runner.py \
  --exclude=vllm/v1/worker/gpu_worker.py
patch_memory_call_sites
apply_patch_once "WSL UVA fallback" "$UVA_FALLBACK_PATCH_FILE"
