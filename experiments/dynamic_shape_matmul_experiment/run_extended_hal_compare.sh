#!/usr/bin/env bash
set -euo pipefail

IREE_BUILD="${IREE_BUILD:-/home/zhongjialin/projects/iree-build}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/iree_dynamic_shape_extended}"
mkdir -p "$OUT_DIR"

compile_hal() {
  local input="$1"
  local output="$2"
  "$IREE_BUILD/tools/iree-compile" \
    --iree-hal-target-backends=llvm-cpu \
    --iree-hal-target-device=local \
    --compile-to=hal \
    "$input" \
    -o "$output"
}

count_pattern() {
  local pattern="$1"
  local file="$2"
  grep -c "$pattern" "$file" || true
}

shape_arith_count() {
  local file="$1"
  grep -E -c "hal\.buffer_view\.dim|arith\.(muli|divsi|select|index_castui|shrui|trunci)" "$file" || true
}

show_dispatch_lines() {
  local file="$1"
  grep -n "hal.command_buffer.dispatch" "$file" || true
  grep -n "hal.device.queue.execute.indirect" "$file" || true
  grep -n "hal.device.queue.execute<" "$file" || true
}

compare_case() {
  local name="$1"
  local static_input="$2"
  local dynamic_input="$3"
  local static_hal="$OUT_DIR/${name}_static_hal.mlir"
  local dynamic_hal="$OUT_DIR/${name}_dynamic_hal.mlir"

  echo
  echo "## $name"
  compile_hal "$ROOT/$static_input" "$static_hal"
  compile_hal "$ROOT/$dynamic_input" "$dynamic_hal"

  printf "static : cmd_dispatch=%s indirect_execute=%s direct_execute=%s dims=%s shape_arith=%s file=%s\n" \
    "$(count_pattern "hal.command_buffer.dispatch" "$static_hal")" \
    "$(count_pattern "hal.device.queue.execute.indirect" "$static_hal")" \
    "$(count_pattern "hal.device.queue.execute<" "$static_hal")" \
    "$(count_pattern "hal.buffer_view.dim" "$static_hal")" \
    "$(shape_arith_count "$static_hal")" \
    "$static_hal"
  printf "dynamic: cmd_dispatch=%s indirect_execute=%s direct_execute=%s dims=%s shape_arith=%s file=%s\n" \
    "$(count_pattern "hal.command_buffer.dispatch" "$dynamic_hal")" \
    "$(count_pattern "hal.device.queue.execute.indirect" "$dynamic_hal")" \
    "$(count_pattern "hal.device.queue.execute<" "$dynamic_hal")" \
    "$(count_pattern "hal.buffer_view.dim" "$dynamic_hal")" \
    "$(shape_arith_count "$dynamic_hal")" \
    "$dynamic_hal"

  echo "-- static dispatch lines --"
  show_dispatch_lines "$static_hal"
  echo "-- dynamic dispatch lines --"
  show_dispatch_lines "$dynamic_hal"
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
echo "HAL IR written to: $OUT_DIR"
