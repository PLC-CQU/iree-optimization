#!/usr/bin/env bash
set -euo pipefail

IREE_BUILD="${IREE_BUILD:-/home/zhongjialin/projects/iree-build}"
CUDA_ARCH="${CUDA_ARCH:-sm_86}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/iree_dynamic_shape_extended_cuda_ir}"
mkdir -p "$OUT_DIR"

compile_stage() {
  local stage="$1"
  local input="$2"
  local output="$3"
  "$IREE_BUILD/tools/iree-compile" \
    --iree-hal-target-backends=cuda \
    --iree-cuda-target="$CUDA_ARCH" \
    --iree-gpu-test-target="$CUDA_ARCH" \
    --compile-to="$stage" \
    "$input" \
    -o "$output"
}

count_pattern() {
  local pattern="$1"
  local file="$2"
  grep -c "$pattern" "$file" || true
}

show_codegen_lines() {
  local file="$1"
  grep -n "hal.executable.export public" "$file" || true
  grep -n "translation_info" "$file" || true
  grep -n "lowering_config" "$file" || true
  grep -n "stream.cmd.dispatch" "$file" || true
}

show_hal_lines() {
  local file="$1"
  grep -n "hal.command_buffer.dispatch" "$file" || true
  grep -n "hal.device.queue.execute.indirect" "$file" || true
  grep -n "hal.device.queue.execute<" "$file" || true
}

compare_case() {
  local name="$1"
  local static_input="$2"
  local dynamic_input="$3"
  local static_config="$OUT_DIR/${name}_static_exec_config.mlir"
  local dynamic_config="$OUT_DIR/${name}_dynamic_exec_config.mlir"
  local static_hal="$OUT_DIR/${name}_static_hal.mlir"
  local dynamic_hal="$OUT_DIR/${name}_dynamic_hal.mlir"

  echo
  echo "## $name"
  compile_stage executable-configurations "$ROOT/$static_input" "$static_config"
  compile_stage executable-configurations "$ROOT/$dynamic_input" "$dynamic_config"
  compile_stage hal "$ROOT/$static_input" "$static_hal"
  compile_stage hal "$ROOT/$dynamic_input" "$dynamic_hal"

  printf "static : TileAndFuse=%s VectorDistribute=%s constants_layout=%s indirect_execute=%s file=%s\n" \
    "$(count_pattern "TileAndFuse" "$static_config")" \
    "$(count_pattern "VectorDistribute" "$static_config")" \
    "$(count_pattern "layout(#hal.pipeline.layout<constants" "$static_config")" \
    "$(count_pattern "hal.device.queue.execute.indirect" "$static_hal")" \
    "$static_config"
  printf "dynamic: TileAndFuse=%s VectorDistribute=%s constants_layout=%s indirect_execute=%s file=%s\n" \
    "$(count_pattern "TileAndFuse" "$dynamic_config")" \
    "$(count_pattern "VectorDistribute" "$dynamic_config")" \
    "$(count_pattern "layout(#hal.pipeline.layout<constants" "$dynamic_config")" \
    "$(count_pattern "hal.device.queue.execute.indirect" "$dynamic_hal")" \
    "$dynamic_config"

  echo "-- static codegen lines --"
  show_codegen_lines "$static_config"
  echo "-- dynamic codegen lines --"
  show_codegen_lines "$dynamic_config"
  echo "-- static hal lines --"
  show_hal_lines "$static_hal"
  echo "-- dynamic hal lines --"
  show_hal_lines "$dynamic_hal"
}

compare_case "01_single_matmul" \
  "static_rank3_matmul_256.mlir" \
  "dynamic_rank3_matmul_256.mlir"

compare_case "02_qkv_shared_lhs" \
  "static_qkv_256.mlir" \
  "dynamic_qkv_256.mlir"

compare_case "03_projection_reshape_transpose" \
  "static_project_transpose_256.mlir" \
  "dynamic_project_transpose_256.mlir"

compare_case "04_attention_scores" \
  "static_attention_scores_256.mlir" \
  "dynamic_attention_scores_256.mlir"

compare_case "05_mlp_gate_up_down" \
  "static_mlp_256.mlir" \
  "dynamic_mlp_256.mlir"

echo
echo "CUDA IR written to: $OUT_DIR"
