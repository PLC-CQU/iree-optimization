#!/usr/bin/env bash
set -euo pipefail

IREE_BUILD="${IREE_BUILD:-/home/zhongjialin/projects/iree-build}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/iree_dynamic_shape_extended}"
mkdir -p "$OUT_DIR"

compile_flow() {
  local input="$1"
  local output="$2"
  "$IREE_BUILD/tools/iree-compile" \
    --iree-hal-target-backends=llvm-cpu \
    --iree-hal-target-device=local \
    --compile-to=flow \
    "$input" \
    -o "$output"
}

count_pattern() {
  local pattern="$1"
  local file="$2"
  grep -c "$pattern" "$file" || true
}

show_key_lines() {
  local file="$1"
  grep -n "flow.executable.export public .*matmul" "$file" || true
  grep -n " = flow.dispatch .*matmul" "$file" || true
  grep -n "flow.dispatch.tie_shape" "$file" || true
  grep -n "workgroups(" "$file" || true
}

compare_case() {
  local name="$1"
  local static_input="$2"
  local dynamic_input="$3"
  local static_flow="$OUT_DIR/${name}_static_flow.mlir"
  local dynamic_flow="$OUT_DIR/${name}_dynamic_flow.mlir"

  echo
  echo "## $name"
  compile_flow "$ROOT/$static_input" "$static_flow"
  compile_flow "$ROOT/$dynamic_input" "$dynamic_flow"

  local static_dispatches
  local dynamic_dispatches
  local static_ties
  local dynamic_ties
  local static_exports
  local dynamic_exports
  static_dispatches="$(count_pattern " = flow.dispatch " "$static_flow")"
  dynamic_dispatches="$(count_pattern " = flow.dispatch " "$dynamic_flow")"
  static_ties="$(count_pattern "flow.dispatch.tie_shape" "$static_flow")"
  dynamic_ties="$(count_pattern "flow.dispatch.tie_shape" "$dynamic_flow")"
  static_exports="$(count_pattern "flow.executable.export public" "$static_flow")"
  dynamic_exports="$(count_pattern "flow.executable.export public" "$dynamic_flow")"

  printf "static : real_dispatch=%s tie_shape=%s executable_exports=%s file=%s\n" \
    "$static_dispatches" "$static_ties" "$static_exports" "$static_flow"
  printf "dynamic: real_dispatch=%s tie_shape=%s executable_exports=%s file=%s\n" \
    "$dynamic_dispatches" "$dynamic_ties" "$dynamic_exports" "$dynamic_flow"

  echo "-- static key lines --"
  show_key_lines "$static_flow"
  echo "-- dynamic key lines --"
  show_key_lines "$dynamic_flow"
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
echo "Flow IR written to: $OUT_DIR"
