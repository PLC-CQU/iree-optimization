#!/usr/bin/env bash
set -euo pipefail

IREE_BUILD="${IREE_BUILD:-/home/zhongjialin/projects/iree-build}"
CUDA_ARCH="${CUDA_ARCH:-sm_86}"
GPU="${GPU:-0}"
BENCH_REPETITIONS="${BENCH_REPETITIONS:-5}"
BENCH_MIN_TIME="${BENCH_MIN_TIME:-1s}"
BENCH_WARMUP_TIME="${BENCH_WARMUP_TIME:-1.0}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/iree_dynamic_shape_specialization_cuda_benchmark}"
mkdir -p "$OUT_DIR"

compile_cuda() {
  local input="$1"
  local output="$2"
  "$IREE_BUILD/tools/iree-compile" \
    --iree-hal-target-backends=cuda \
    --iree-cuda-target="$CUDA_ARCH" \
    --iree-gpu-test-target="$CUDA_ARCH" \
    "$ROOT/$input" \
    -o "$output"
}

bench_cuda() {
  local label="$1"
  local module="$2"
  local safe_label="$3"
  echo
  echo "## $label"
  CUDA_VISIBLE_DEVICES="$GPU" "$IREE_BUILD/tools/iree-benchmark-module" \
    --module="$module" \
    --device=cuda \
    --function=main \
    --input=4x128x256xf32=1 \
    --input=256x256xf32=1 \
    --benchmark_repetitions="$BENCH_REPETITIONS" \
    --benchmark_min_time="$BENCH_MIN_TIME" \
    --benchmark_min_warmup_time="$BENCH_WARMUP_TIME" \
    --benchmark_time_unit=us \
    --benchmark_out="$OUT_DIR/${safe_label}.googlebench.json" \
    --benchmark_out_format=json
}

compile_cuda "static_rank3_matmul_256.mlir" "$OUT_DIR/static.vmfb"
compile_cuda "dynamic_rank3_matmul_256.mlir" "$OUT_DIR/dynamic.vmfb"
compile_cuda "dynamic_specialized_rank3_matmul_256.mlir" "$OUT_DIR/dynamic_internal_static.vmfb"
compile_cuda "dynamic_guarded_specialized_rank3_matmul_256.mlir" "$OUT_DIR/dynamic_guarded.vmfb"

bench_cuda "static" "$OUT_DIR/static.vmfb" "static"
bench_cuda "dynamic" "$OUT_DIR/dynamic.vmfb" "dynamic"
bench_cuda "dynamic ABI + internal static cast" "$OUT_DIR/dynamic_internal_static.vmfb" "dynamic_internal_static"
bench_cuda "dynamic ABI + guarded static fast path" "$OUT_DIR/dynamic_guarded.vmfb" "dynamic_guarded"

echo
echo "VMFB files written to: $OUT_DIR"
echo "Benchmark JSON files written to: $OUT_DIR/*.googlebench.json"
