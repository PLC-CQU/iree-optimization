#!/usr/bin/env bash
set -euo pipefail

IREE_BUILD="${IREE_BUILD:-/home/zhongjialin/projects/iree-build}"
ROOT="$(cd "$(dirname "$0")" && pwd)"

STATIC_4096="$ROOT/static_rank3_matmul.mlir"
DYNAMIC_4096="$ROOT/dynamic_rank3_matmul.mlir"
STATIC_256="$ROOT/static_rank3_matmul_256.mlir"
DYNAMIC_256="$ROOT/dynamic_rank3_matmul_256.mlir"

echo "== Flatten pass only =="
"$IREE_BUILD/tools/iree-opt" \
  --pass-pipeline='builtin.module(func.func(iree-global-opt-flatten-rank3-matmul))' \
  "$STATIC_4096" -o /tmp/static_rank3_flattened.mlir
"$IREE_BUILD/tools/iree-opt" \
  --pass-pipeline='builtin.module(func.func(iree-global-opt-flatten-rank3-matmul))' \
  "$DYNAMIC_4096" -o /tmp/dynamic_rank3_flattened.mlir

grep -n "collapse_shape\\|linalg.matmul\\|expand_shape" /tmp/static_rank3_flattened.mlir
grep -n "collapse_shape\\|linalg.matmul\\|expand_shape" /tmp/dynamic_rank3_flattened.mlir

echo
echo "== IREE global-optimization =="
"$IREE_BUILD/tools/iree-compile" \
  --iree-hal-target-backends=llvm-cpu \
  --iree-hal-target-device=local \
  --compile-to=global-optimization \
  "$STATIC_4096" -o /tmp/static_rank3_global.mlir
"$IREE_BUILD/tools/iree-compile" \
  --iree-hal-target-backends=llvm-cpu \
  --iree-hal-target-device=local \
  --compile-to=global-optimization \
  "$DYNAMIC_4096" -o /tmp/dynamic_rank3_global.mlir

printf 'static ? count: '
grep -o '?' /tmp/static_rank3_global.mlir | wc -l
printf 'dynamic ? count: '
grep -o '?' /tmp/dynamic_rank3_global.mlir | wc -l
grep -n "linalg.matmul" /tmp/static_rank3_global.mlir
grep -n "linalg.matmul" /tmp/dynamic_rank3_global.mlir

echo
echo "== IREE flow =="
"$IREE_BUILD/tools/iree-compile" \
  --iree-hal-target-backends=llvm-cpu \
  --iree-hal-target-device=local \
  --compile-to=flow \
  "$STATIC_4096" -o /tmp/static_rank3_flow.mlir
"$IREE_BUILD/tools/iree-compile" \
  --iree-hal-target-backends=llvm-cpu \
  --iree-hal-target-device=local \
  --compile-to=flow \
  "$DYNAMIC_4096" -o /tmp/dynamic_rank3_flow.mlir

grep -n "flow.executable.export\\|flow.dispatch" /tmp/static_rank3_flow.mlir
grep -n "flow.executable.export\\|flow.dispatch" /tmp/dynamic_rank3_flow.mlir

echo
echo "== Small 256 benchmark with local-task =="
"$IREE_BUILD/tools/iree-compile" \
  --iree-hal-target-backends=llvm-cpu \
  --iree-hal-target-device=local \
  "$STATIC_256" -o /tmp/static_rank3_256.vmfb
"$IREE_BUILD/tools/iree-compile" \
  --iree-hal-target-backends=llvm-cpu \
  --iree-hal-target-device=local \
  "$DYNAMIC_256" -o /tmp/dynamic_rank3_256.vmfb

"$IREE_BUILD/tools/iree-benchmark-module" \
  --module=/tmp/static_rank3_256.vmfb \
  --device=local-task \
  --function=main \
  --input=4x128x256xf32=1 \
  --input=256x256xf32=1 \
  --benchmark_repetitions=5 \
  --benchmark_min_time=0.5s \
  --benchmark_time_unit=ms

"$IREE_BUILD/tools/iree-benchmark-module" \
  --module=/tmp/dynamic_rank3_256.vmfb \
  --device=local-task \
  --function=main \
  --input=4x128x256xf32=1 \
  --input=256x256xf32=1 \
  --benchmark_repetitions=5 \
  --benchmark_min_time=0.5s \
  --benchmark_time_unit=ms
