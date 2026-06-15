#!/usr/bin/env bash
set -euo pipefail

IREE_BUILD="${IREE_BUILD:-/home/zhongjialin/projects/iree-build}"
CUDA_ARCH="${CUDA_ARCH:-sm_86}"
GPU="${GPU:-2}"
BH="${BH:-128}"
S="${S:-104}"
REPETITIONS="${REPETITIONS:-3}"
MIN_TIME="${MIN_TIME:-1x}"
WARMUP_TIME="${WARMUP_TIME:-0.5}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/iree_bounded_attention_context_benchmark}"
mkdir -p "$OUT_DIR"

compile_vmfb() {
  local input="$1"
  local output="$2"
  "$IREE_BUILD/tools/iree-compile" \
    --iree-hal-target-backends=cuda \
    --iree-cuda-target="$CUDA_ARCH" \
    --iree-gpu-test-target="$CUDA_ARCH" \
    "$ROOT/$input" \
    -o "$output"
}

compile_config() {
  local input="$1"
  local output="$2"
  "$IREE_BUILD/tools/iree-compile" \
    --iree-hal-target-backends=cuda \
    --iree-cuda-target="$CUDA_ARCH" \
    --iree-gpu-test-target="$CUDA_ARCH" \
    --compile-to=executable-configurations \
    "$ROOT/$input" \
    -o "$output"
}

bench() {
  local label="$1"
  local module="$2"
  local out_json="$OUT_DIR/${label}.googlebench.json"

  echo
  echo "## $label"
  CUDA_VISIBLE_DEVICES="$GPU" "$IREE_BUILD/tools/iree-benchmark-module" \
    --module="$module" \
    --device=cuda \
    --function=main \
    --input="${BH}x${S}x${S}xf16=1" \
    --input="${BH}x${S}x128xf16=1" \
    --benchmark_repetitions="$REPETITIONS" \
    --benchmark_min_time="$MIN_TIME" \
    --benchmark_min_warmup_time="$WARMUP_TIME" \
    --benchmark_time_unit=ms \
    --benchmark_out="$out_json" \
    --benchmark_out_format=json
}

compile_config dynamic_deepseek_attn_context.mlir "$OUT_DIR/dynamic_unbounded_config.mlir"
compile_config dynamic_bounded_deepseek_attn_context.mlir "$OUT_DIR/dynamic_bounded_config.mlir"
compile_config static_deepseek_attn_context_104.mlir "$OUT_DIR/static_config.mlir"

echo "## configs"
grep -n "translation_info\\|lowering_config" "$OUT_DIR/dynamic_unbounded_config.mlir" || true
grep -n "translation_info\\|lowering_config" "$OUT_DIR/dynamic_bounded_config.mlir" || true
grep -n "translation_info\\|lowering_config" "$OUT_DIR/static_config.mlir" || true

compile_vmfb dynamic_deepseek_attn_context.mlir "$OUT_DIR/dynamic_unbounded.vmfb"
compile_vmfb dynamic_bounded_deepseek_attn_context.mlir "$OUT_DIR/dynamic_bounded.vmfb"
compile_vmfb static_deepseek_attn_context_104.mlir "$OUT_DIR/static.vmfb"

bench dynamic_unbounded "$OUT_DIR/dynamic_unbounded.vmfb"
bench dynamic_bounded "$OUT_DIR/dynamic_bounded.vmfb"
bench static "$OUT_DIR/static.vmfb"

python3 "$ROOT/summarize_googlebench.py" "$OUT_DIR"/*.googlebench.json
