#!/usr/bin/env bash
set -euo pipefail

IREE_BUILD="${IREE_BUILD:-/home/zhongjialin/projects/iree-build}"
CUDA_ARCH="${CUDA_ARCH:-sm_86}"
GPU="${GPU:-0}"
BENCH_REPETITIONS="${BENCH_REPETITIONS:-5}"
BENCH_MIN_TIME="${BENCH_MIN_TIME:-1s}"
BENCH_WARMUP_TIME="${BENCH_WARMUP_TIME:-}"
BENCH_OUT_FORMAT="${BENCH_OUT_FORMAT:-}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/iree_dynamic_shape_extended_cuda}"
mkdir -p "$OUT_DIR"

compile_cuda() {
  local input="$1"
  local output="$2"
  "$IREE_BUILD/tools/iree-compile" \
    --iree-hal-target-backends=cuda \
    --iree-cuda-target="$CUDA_ARCH" \
    --iree-gpu-test-target="$CUDA_ARCH" \
    "$input" \
    -o "$output"
}

bench_cuda() {
  local module="$1"
  local bench_name="$2"
  shift 2
  local extra_args=()
  if [[ -n "$BENCH_WARMUP_TIME" ]]; then
    extra_args+=(--benchmark_min_warmup_time="$BENCH_WARMUP_TIME")
  fi
  if [[ -n "$BENCH_OUT_FORMAT" ]]; then
    extra_args+=(--benchmark_out="$OUT_DIR/${bench_name}.googlebench.json")
    extra_args+=(--benchmark_out_format="$BENCH_OUT_FORMAT")
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "$IREE_BUILD/tools/iree-benchmark-module" \
    --module="$module" \
    --device=cuda \
    --function=main \
    "$@" \
    --benchmark_repetitions="$BENCH_REPETITIONS" \
    --benchmark_min_time="$BENCH_MIN_TIME" \
    --benchmark_time_unit=us \
    "${extra_args[@]}"
}

run_case() {
  local name="$1"
  local static_input="$2"
  local dynamic_input="$3"
  shift 3
  local static_vmfb="$OUT_DIR/${name}_static.vmfb"
  local dynamic_vmfb="$OUT_DIR/${name}_dynamic.vmfb"

  echo
  echo "## $name"
  compile_cuda "$ROOT/$static_input" "$static_vmfb"
  compile_cuda "$ROOT/$dynamic_input" "$dynamic_vmfb"

  echo "-- static --"
  bench_cuda "$static_vmfb" "${name}_static" "$@"
  echo "-- dynamic --"
  bench_cuda "$dynamic_vmfb" "${name}_dynamic" "$@"
}

run_case "01_single_matmul" \
  "static_rank3_matmul_256.mlir" \
  "dynamic_rank3_matmul_256.mlir" \
  --input=4x128x256xf32=1 \
  --input=256x256xf32=1

run_case "02_qkv_shared_lhs" \
  "static_qkv_256.mlir" \
  "dynamic_qkv_256.mlir" \
  --input=4x128x256xf32=1 \
  --input=256x256xf32=1 \
  --input=256x256xf32=1 \
  --input=256x256xf32=1

run_case "03_projection_reshape_transpose" \
  "static_project_transpose_256.mlir" \
  "dynamic_project_transpose_256.mlir" \
  --input=4x128x256xf32=1 \
  --input=256x256xf32=1

run_case "04_attention_scores" \
  "static_attention_scores_256.mlir" \
  "dynamic_attention_scores_256.mlir" \
  --input=4x32x128x8xf32=1 \
  --input=4x32x8x128xf32=1

run_case "05_mlp_gate_up_down" \
  "static_mlp_256.mlir" \
  "dynamic_mlp_256.mlir" \
  --input=4x128x256xf32=1 \
  --input=256x512xf32=1 \
  --input=256x512xf32=1 \
  --input=512x256xf32=1

echo
echo "CUDA vmfb written to: $OUT_DIR"
