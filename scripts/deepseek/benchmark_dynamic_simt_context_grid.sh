#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd)}"
GPU="${GPU:-2}"
REPETITIONS="${REPETITIONS:-3}"
MIN_TIME="${MIN_TIME:-1x}"
WARMUP_TIME="${WARMUP_TIME:-0}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-120s}"
BOUNDED_DYNAMIC_VMFB="${BOUNDED_DYNAMIC_VMFB:-/tmp/deepseek_dynamic_bs_bounded_b8_s128_simt_context.vmfb}"
BOUNDED_DYNAMIC_NAME="${BOUNDED_DYNAMIC_NAME:-bounded_dynamic_b8_s128}"
OUT_ROOT="${OUT_ROOT:-/tmp/iree_dynamic_simt_context_grid}"

B_VALUES=(${B_VALUES:-1 4 8})
S_VALUES=(${S_VALUES:-32 64 128})

mkdir -p "${OUT_ROOT}"

for b in "${B_VALUES[@]}"; do
  for s in "${S_VALUES[@]}"; do
    echo
    echo "## grid b${b}_s${s}"
    out_dir="${OUT_ROOT}/b${b}_s${s}"
    if [[ ! -f "${ROOT}/deepseek_r1_8b_onnx_iree_cuda_b${b}_s${s}_last_nonpad_index_flatten_matmul_input_i32_with_demote.vmfb" ]]; then
      echo "skip b${b}_s${s}: missing static exact vmfb"
      continue
    fi
    if [[ ! -d "${ROOT}/flatten_shape_b${b}_s${s}/build_flatten_matmul_last_nonpad_index_input_i32_with_demote" ]]; then
      echo "skip b${b}_s${s}: missing static params dir"
      continue
    fi
    GPU="${GPU}" \
    B="${b}" \
    S="${s}" \
    REPETITIONS="${REPETITIONS}" \
    MIN_TIME="${MIN_TIME}" \
    WARMUP_TIME="${WARMUP_TIME}" \
    BENCH_TIMEOUT="${BENCH_TIMEOUT}" \
    BOUNDED_DYNAMIC_VMFB="${BOUNDED_DYNAMIC_VMFB}" \
    BOUNDED_DYNAMIC_NAME="${BOUNDED_DYNAMIC_NAME}" \
    OUT_DIR="${out_dir}" \
      "${ROOT}/benchmark_bounded_dynamic_shape.sh" || {
        status="$?"
        echo "failed b${b}_s${s}: status=${status}"
        continue
      }
  done
done

python3 - "${OUT_ROOT}" "${BOUNDED_DYNAMIC_NAME}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidate_name = sys.argv[2]

def mean_ms(path):
    with path.open() as f:
        data = json.load(f)
    for item in data.get("benchmarks", []):
        if item.get("run_name", "").endswith("_mean"):
            return float(item["real_time"])
    vals = [
        float(item["real_time"])
        for item in data.get("benchmarks", [])
        if item.get("run_type") == "iteration"
    ]
    if not vals:
        raise RuntimeError(f"missing benchmark time in {path}")
    return sum(vals) / len(vals)

rows = []
for case_dir in sorted(root.glob("b*_s*")):
    safe = case_dir / "safe_dynamic_m.googlebench.json"
    bounded = case_dir / f"{candidate_name}.googlebench.json"
    static = case_dir / "static_exact.googlebench.json"
    if not (safe.exists() and bounded.exists() and static.exists()):
        continue
    safe_ms = mean_ms(safe)
    bounded_ms = mean_ms(bounded)
    static_ms = mean_ms(static)
    rows.append((case_dir.name, safe_ms, bounded_ms, static_ms))

print()
print("## grid summary")
print(f"case safe_dynamic_m_ms {candidate_name}_ms static_exact_ms {candidate_name}_vs_safe speedup {candidate_name}_vs_static")
for name, safe_ms, bounded_ms, static_ms in rows:
    print(
        f"{name} {safe_ms:.3f} {bounded_ms:.3f} {static_ms:.3f} "
        f"{bounded_ms / safe_ms:.3f}x {safe_ms / bounded_ms:.3f}x "
        f"{bounded_ms / static_ms:.3f}x"
    )
PY
