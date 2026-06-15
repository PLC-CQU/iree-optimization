#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd)}"
BUILD="${BUILD:-${IREE_BUILD:-/home/zhongjialin/projects/iree-build}}"
GPU="${GPU:-2}"
B="${B:-1}"
S="${S:-32}"
REPETITIONS="${REPETITIONS:-3}"
MIN_TIME="${MIN_TIME:-1x}"
WARMUP_TIME="${WARMUP_TIME:-0}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-120s}"

SAFE_DYNAMIC_VMFB="${SAFE_DYNAMIC_VMFB:-${ROOT}/deepseek_dynamic_b_s_last_safe_dynamic_m_input_i32_with_demote.vmfb}"
BOUNDED_DYNAMIC_VMFB="${BOUNDED_DYNAMIC_VMFB:-/tmp/deepseek_dynamic_bs_bounded_b8_s128_simt_context.vmfb}"
BOUNDED_DYNAMIC_NAME="${BOUNDED_DYNAMIC_NAME:-bounded_dynamic_b8_s128}"
STATIC_VMFB="${STATIC_VMFB:-${ROOT}/deepseek_r1_8b_onnx_iree_cuda_b${B}_s${S}_last_nonpad_index_flatten_matmul_input_i32_with_demote.vmfb}"
DYNAMIC_PARAMS="${DYNAMIC_PARAMS:-${ROOT}/dynamic_shape_b_s/build_flatten_matmul_input_i32_with_demote/deepseek_r1_8b_params.irpa}"
STATIC_PARAMS="${STATIC_PARAMS:-${ROOT}/flatten_shape_b${B}_s${S}/build_flatten_matmul_last_nonpad_index_input_i32_with_demote/deepseek_r1_8b_params.irpa}"
OUT_DIR="${OUT_DIR:-/tmp/iree_bounded_dynamic_benchmark_b${B}_s${S}}"

mkdir -p "${OUT_DIR}"

run_benchmark() {
  local name="$1"
  local module="$2"
  local params="$3"
  local last_token_indices="${4:-}"
  local json="${OUT_DIR}/${name}.googlebench.json"

  echo
  echo "## ${name}"
  echo "module: ${module}"
  echo "params: ${params}"
  echo "timeout: ${BENCH_TIMEOUT}"
  local cmd=(
    "${BUILD}/tools/iree-benchmark-module"
    --module="${module}" \
    --parameters="model=${params}" \
    --parameter_mode=file \
    --device=cuda \
    --function=main_graph \
    --input="${B}x${S}xi32=0" \
    --input="${B}x${S}xi32=1" \
    --benchmark_repetitions="${REPETITIONS}" \
    --benchmark_min_time="${MIN_TIME}" \
    --benchmark_min_warmup_time="${WARMUP_TIME}" \
    --benchmark_time_unit=ms \
    --benchmark_out="${json}" \
    --benchmark_out_format=json
  )
  if [[ -n "${last_token_indices}" ]]; then
    cmd+=(--input="${last_token_indices}")
  fi
  timeout "${BENCH_TIMEOUT}" env CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}" || {
      local status="$?"
      echo "benchmark failed or timed out: ${name}, status=${status}"
      return "${status}"
  }
}

run_benchmark "safe_dynamic_m" "${SAFE_DYNAMIC_VMFB}" "${DYNAMIC_PARAMS}"
run_benchmark "${BOUNDED_DYNAMIC_NAME}" "${BOUNDED_DYNAMIC_VMFB}" "${DYNAMIC_PARAMS}"
run_benchmark "static_exact" "${STATIC_VMFB}" "${STATIC_PARAMS}" "${B}xi32=$((S - 1))"

python3 - "${OUT_DIR}" "${BOUNDED_DYNAMIC_NAME}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
candidate_name = sys.argv[2]

def mean_ms(name):
    path = out_dir / f"{name}.googlebench.json"
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

safe = mean_ms("safe_dynamic_m")
bounded = mean_ms(candidate_name)
static = mean_ms("static_exact")
print()
print("## summary")
print(f"safe_dynamic_m_ms: {safe:.3f}")
print(f"{candidate_name}_ms: {bounded:.3f}")
print(f"static_exact_ms: {static:.3f}")
print(f"{candidate_name}_vs_safe_speedup: {safe / bounded:.3f}x")
print(f"safe_vs_static_slowdown: {safe / static:.3f}x")
print(f"safe_vs_static_gap: {(safe / static - 1.0) * 100.0:.2f}%")
print(f"{candidate_name}_vs_static_slowdown: {bounded / static:.3f}x")
print(f"{candidate_name}_vs_static_gap: {(bounded / static - 1.0) * 100.0:.2f}%")
print(f"json_dir: {out_dir}")
PY
