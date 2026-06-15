#!/usr/bin/env bash
set -euo pipefail

IREE_BUILD="${IREE_BUILD:-/home/zhongjialin/projects/iree-build}"
CUDA_ARCH="${CUDA_ARCH:-sm_86}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/iree_dynamic_shape_specialization_probe}"
mkdir -p "$OUT_DIR"

compile_config() {
  local input="$1"
  local output="$2"
  "$IREE_BUILD/tools/iree-compile" \
    --iree-hal-target-backends=cuda \
    --iree-cuda-target="$CUDA_ARCH" \
    --iree-gpu-test-target="$CUDA_ARCH" \
    --compile-to=executable-configurations \
    "$input" \
    -o "$output"
}

compile_hal() {
  local input="$1"
  local output="$2"
  "$IREE_BUILD/tools/iree-compile" \
    --iree-hal-target-backends=cuda \
    --iree-cuda-target="$CUDA_ARCH" \
    --iree-gpu-test-target="$CUDA_ARCH" \
    --compile-to=hal \
    "$input" \
    -o "$output"
}

show_case() {
  local label="$1"
  local input="$2"
  local config="$OUT_DIR/${label}_exec_config.mlir"
  local hal="$OUT_DIR/${label}_hal.mlir"

  compile_config "$ROOT/$input" "$config"
  compile_hal "$ROOT/$input" "$hal"

  echo
  echo "## $label"
  grep -n "hal.executable.export public" "$config" || true
  grep -n "translation_info" "$config" || true
  grep -n "lowering_config" "$config" || true
  grep -n "hal.command_buffer.dispatch" "$hal" || true
  grep -n "hal.device.queue.execute.indirect" "$hal" || true
  grep -n "hal.device.queue.execute<" "$hal" || true
  echo "config=$config"
  echo "hal=$hal"
}

show_case "static" "static_rank3_matmul_256.mlir"
show_case "dynamic" "dynamic_rank3_matmul_256.mlir"
show_case "dynamic_specialized_internal_static" "dynamic_specialized_rank3_matmul_256.mlir"
