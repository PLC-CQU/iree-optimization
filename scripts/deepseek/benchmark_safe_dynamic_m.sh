#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
IREE_BUILD="${IREE_BUILD:-/home/zhongjialin/projects/iree-build}"
GPU="${GPU:-2}"
B="${B:-4}"
S="${S:-64}"
REPETITIONS="${REPETITIONS:-3}"
MIN_TIME="${MIN_TIME:-1x}"
WARMUP_TIME="${WARMUP_TIME:-0.5}"
OUT_DIR="${OUT_DIR:-/tmp/iree_safe_dynamic_m_benchmark_b${B}_s${S}}"

OLD_DYNAMIC_VMFB="${OLD_DYNAMIC_VMFB:-$ROOT/deepseek_dynamic_b_s_last_flatten_matmul_input_i32_with_demote.vmfb}"
SAFE_DYNAMIC_M_VMFB="${SAFE_DYNAMIC_M_VMFB:-$ROOT/deepseek_dynamic_b_s_last_safe_dynamic_m_input_i32_with_demote.vmfb}"
PARAMS="${PARAMS:-$ROOT/dynamic_shape_b_s/build_flatten_matmul_input_i32_with_demote/deepseek_r1_8b_params.irpa}"
STATIC_VMFB="${STATIC_VMFB:-$ROOT/deepseek_r1_8b_onnx_iree_cuda_b${B}_s${S}_last_nonpad_index_flatten_matmul_input_i32_with_demote.vmfb}"
STATIC_PARAMS="${STATIC_PARAMS:-$ROOT/flatten_shape_b${B}_s${S}/build_flatten_matmul_last_nonpad_index_input_i32_with_demote/deepseek_r1_8b_params.irpa}"

mkdir -p "$OUT_DIR"

for required_file in "$OLD_DYNAMIC_VMFB" "$SAFE_DYNAMIC_M_VMFB" "$PARAMS"; do
  if [[ ! -f "$required_file" ]]; then
    echo "missing required file: $required_file" >&2
    exit 1
  fi
done

run_case() {
  local name="$1"
  local module="$2"
  local params="$3"
  local last_token_indices="${4:-}"
  local out_json="$OUT_DIR/${name}.googlebench.json"

  echo
  echo "## $name"
  echo "module: $module"
  local cmd=(
    "$IREE_BUILD/tools/iree-benchmark-module"
    --module="$module" \
    --parameters="model=$params" \
    --parameter_mode=file \
    --device=cuda \
    --function=main_graph \
    --input="${B}x${S}xi32=0" \
    --input="${B}x${S}xi32=1" \
    --benchmark_repetitions="$REPETITIONS" \
    --benchmark_min_time="$MIN_TIME" \
    --benchmark_min_warmup_time="$WARMUP_TIME" \
    --benchmark_time_unit=ms \
    --benchmark_out="$out_json" \
    --benchmark_out_format=json
  )
  if [[ -n "$last_token_indices" ]]; then
    cmd+=(--input="$last_token_indices")
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}"
}

run_case old_dynamic "$OLD_DYNAMIC_VMFB" "$PARAMS"
run_case safe_dynamic_m "$SAFE_DYNAMIC_M_VMFB" "$PARAMS"

if [[ -f "$STATIC_VMFB" && -f "$STATIC_PARAMS" ]]; then
  run_case static_exact "$STATIC_VMFB" "$STATIC_PARAMS" "${B}xi32=$((S - 1))"
else
  echo
  echo "## static_exact"
  echo "skip: no exact static model for b${B}_s${S}"
  echo "expected module: $STATIC_VMFB"
  echo "expected params: $STATIC_PARAMS"
fi

python3 - "$OUT_DIR" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])

def mean_ms(name):
    path = out_dir / f"{name}.googlebench.json"
    if not path.exists():
        return None
    data = json.load(open(path))
    single_run_times = []
    for bench in data["benchmarks"]:
        if bench["name"].endswith("_mean"):
            return float(bench["real_time"])
        if bench["name"].endswith("/real_time"):
            single_run_times.append(float(bench["real_time"]))
    if single_run_times:
        return sum(single_run_times) / len(single_run_times)
    raise RuntimeError(f"missing real_time benchmark in {path}")

old = mean_ms("old_dynamic")
safe = mean_ms("safe_dynamic_m")
static = mean_ms("static_exact")
print()
print("## summary")
print(f"old_dynamic_ms: {old:.3f}")
print(f"safe_dynamic_m_ms: {safe:.3f}")
print(f"speedup: {old / safe:.3f}x")
print(f"improvement: {(old - safe) / old * 100.0:.2f}%")
if static is not None:
    print(f"static_exact_ms: {static:.3f}")
    print(f"safe_vs_static_slowdown: {safe / static:.3f}x")
    print(f"safe_vs_static_gap: {(safe - static) / static * 100.0:.2f}%")
print(f"json_dir: {out_dir}")
PY
