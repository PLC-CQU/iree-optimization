#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
IREE_BUILD="${IREE_BUILD:-/home/zhongjialin/projects/iree-build}"
GPU="${GPU:-2}"
B="${B:-1}"
S="${S:-32}"
ATOL="${ATOL:-0.06}"
RTOL="${RTOL:-0.01}"
OUT_DIR="${OUT_DIR:-/tmp/iree_safe_dynamic_m_correctness_b${B}_s${S}}"

OLD_DYNAMIC_VMFB="${OLD_DYNAMIC_VMFB:-$ROOT/deepseek_dynamic_b_s_last_flatten_matmul_input_i32_with_demote.vmfb}"
SAFE_DYNAMIC_M_VMFB="${SAFE_DYNAMIC_M_VMFB:-$ROOT/deepseek_dynamic_b_s_last_safe_dynamic_m_input_i32_with_demote.vmfb}"
CANDIDATE_DYNAMIC_VMFB="${CANDIDATE_DYNAMIC_VMFB:-}"
DYNAMIC_PARAMS="${DYNAMIC_PARAMS:-$ROOT/dynamic_shape_b_s/build_flatten_matmul_input_i32_with_demote/deepseek_r1_8b_params.irpa}"
STATIC_VMFB="${STATIC_VMFB:-$ROOT/deepseek_r1_8b_onnx_iree_cuda_b${B}_s${S}_last_nonpad_index_flatten_matmul_input_i32_with_demote.vmfb}"
STATIC_PARAMS="${STATIC_PARAMS:-$ROOT/flatten_shape_b${B}_s${S}/build_flatten_matmul_last_nonpad_index_input_i32_with_demote/deepseek_r1_8b_params.irpa}"

mkdir -p "$OUT_DIR"

for required_file in "$OLD_DYNAMIC_VMFB" "$SAFE_DYNAMIC_M_VMFB" "$DYNAMIC_PARAMS" "$STATIC_VMFB" "$STATIC_PARAMS"; do
  if [[ ! -f "$required_file" ]]; then
    echo "missing required file: $required_file" >&2
    exit 1
  fi
done
if [[ -n "$CANDIDATE_DYNAMIC_VMFB" && ! -f "$CANDIDATE_DYNAMIC_VMFB" ]]; then
  echo "missing required file: $CANDIDATE_DYNAMIC_VMFB" >&2
  exit 1
fi

run_dynamic() {
  local name="$1"
  local module="$2"
  local output="$OUT_DIR/${name}.npy"

  echo
  echo "## $name"
  CUDA_VISIBLE_DEVICES="$GPU" "$IREE_BUILD/tools/iree-run-module" \
    --module="$module" \
    --parameters="model=$DYNAMIC_PARAMS" \
    --parameter_mode=file \
    --device=cuda \
    --function=main_graph \
    --input="${B}x${S}xi32=0" \
    --input="${B}x${S}xi32=1" \
    --output="@$output"
}

run_static() {
  local output="$OUT_DIR/static_exact.npy"

  echo
  echo "## static_exact"
  CUDA_VISIBLE_DEVICES="$GPU" "$IREE_BUILD/tools/iree-run-module" \
    --module="$STATIC_VMFB" \
    --parameters="model=$STATIC_PARAMS" \
    --parameter_mode=file \
    --device=cuda \
    --function=main_graph \
    --input="${B}x${S}xi32=0" \
    --input="${B}x${S}xi32=1" \
    --input="${B}xi32=$((S - 1))" \
    --output="@$output"
}

run_dynamic old_dynamic "$OLD_DYNAMIC_VMFB"
run_dynamic safe_dynamic_m "$SAFE_DYNAMIC_M_VMFB"
if [[ -n "$CANDIDATE_DYNAMIC_VMFB" ]]; then
  run_dynamic candidate_dynamic "$CANDIDATE_DYNAMIC_VMFB"
fi
run_static

python3 - "$OUT_DIR" "$ATOL" "$RTOL" "$CANDIDATE_DYNAMIC_VMFB" <<'PY'
import pathlib
import sys

import numpy as np

out_dir = pathlib.Path(sys.argv[1])
atol = float(sys.argv[2])
rtol = float(sys.argv[3])
has_candidate = bool(sys.argv[4])

old = np.load(out_dir / "old_dynamic.npy")
safe = np.load(out_dir / "safe_dynamic_m.npy")
static = np.load(out_dir / "static_exact.npy")
candidate = np.load(out_dir / "candidate_dynamic.npy") if has_candidate else None

def report(name, a, b):
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    diff = np.abs(a32 - b32)
    close = np.allclose(a32, b32, atol=atol, rtol=rtol)
    print(f"{name}")
    print(f"  shape_a: {a.shape}")
    print(f"  shape_b: {b.shape}")
    print(f"  max_abs: {float(diff.max()):.8f}")
    print(f"  mean_abs: {float(diff.mean()):.8f}")
    print(f"  allclose_atol_{atol}_rtol_{rtol}: {close}")
    if not close:
        raise SystemExit(1)

print()
print("## correctness summary")
report("old_dynamic vs safe_dynamic_m", old, safe)
if has_candidate:
    report("safe_dynamic_m vs candidate_dynamic", safe, candidate)
    report("candidate_dynamic vs static_exact", candidate, static)
report("safe_dynamic_m vs static_exact", safe, static)
print(f"output_dir: {out_dir}")
PY
