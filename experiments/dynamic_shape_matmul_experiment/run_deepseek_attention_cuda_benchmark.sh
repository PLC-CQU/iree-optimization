#!/usr/bin/env bash
set -euo pipefail

IREE_BUILD="${IREE_BUILD:-/home/zhongjialin/projects/iree-build}"
CUDA_ARCH="${CUDA_ARCH:-sm_86}"
GPU="${GPU:-0}"
BENCH_REPETITIONS="${BENCH_REPETITIONS:-10}"
BENCH_MIN_TIME="${BENCH_MIN_TIME:-10s}"
BENCH_WARMUP_TIME="${BENCH_WARMUP_TIME:-2.0}"
OUT_DIR="${OUT_DIR:-/tmp/iree_deepseek_attention_cuda}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
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
  local name="$1"
  local module="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$GPU" "$IREE_BUILD/tools/iree-benchmark-module" \
    --module="$module" \
    --device=cuda \
    --function=main \
    "$@" \
    --benchmark_repetitions="$BENCH_REPETITIONS" \
    --benchmark_min_time="$BENCH_MIN_TIME" \
    --benchmark_min_warmup_time="$BENCH_WARMUP_TIME" \
    --benchmark_time_unit=us \
    --benchmark_out="$OUT_DIR/${name}.googlebench.json" \
    --benchmark_out_format=json
}

run_case() {
  local name="$1"
  local input="$2"
  shift 2
  local vmfb="$OUT_DIR/${name}.vmfb"
  echo
  echo "## $name"
  compile_cuda "$ROOT/$input" "$vmfb"
  bench_cuda "$name" "$vmfb" "$@"
}

run_case static_scores \
  static_deepseek_attn_scores_104.mlir \
  --input=128x104x128xf16=1 \
  --input=128x128x104xf16=1

run_case dynamic_scores \
  dynamic_deepseek_attn_scores.mlir \
  --input=128x104x128xf16=1 \
  --input=128x128x104xf16=1

run_case static_context \
  static_deepseek_attn_context_104.mlir \
  --input=128x104x104xf16=1 \
  --input=128x104x128xf16=1

run_case dynamic_context \
  dynamic_deepseek_attn_context.mlir \
  --input=128x104x104xf16=1 \
  --input=128x104x128xf16=1

echo
echo "VMFB and JSON written to: $OUT_DIR"
