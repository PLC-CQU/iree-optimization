#!/usr/bin/env bash
set -euo pipefail

IREE_BUILD="${IREE_BUILD:-/home/zhongjialin/projects/iree-build}"
CUDA_ARCH="${CUDA_ARCH:-sm_86}"
GPU="${GPU:-0}"
ROOT="$(cd "$(dirname "$0")" && pwd)"

STATIC_256="$ROOT/static_rank3_matmul_256.mlir"
DYNAMIC_256="$ROOT/dynamic_rank3_matmul_256.mlir"
STATIC_4096="$ROOT/static_rank3_matmul.mlir"
DYNAMIC_4096="$ROOT/dynamic_rank3_matmul.mlir"

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

flow_cuda() {
  local input="$1"
  local output="$2"
  "$IREE_BUILD/tools/iree-compile" \
    --iree-hal-target-backends=cuda \
    --iree-cuda-target="$CUDA_ARCH" \
    --iree-gpu-test-target="$CUDA_ARCH" \
    --compile-to=flow \
    "$input" \
    -o "$output"
}

bench_cuda() {
  local module="$1"
  local lhs_shape="$2"
  local rhs_shape="$3"
  CUDA_VISIBLE_DEVICES="$GPU" "$IREE_BUILD/tools/iree-benchmark-module" \
    --module="$module" \
    --device=cuda \
    --function=main \
    --input="${lhs_shape}xf32=1" \
    --input="${rhs_shape}xf32=1" \
    --benchmark_repetitions=5 \
    --benchmark_min_time=1s \
    --benchmark_time_unit=ms
}

echo "== CUDA flow IR, 256 =="
flow_cuda "$STATIC_256" /tmp/static_rank3_256_cuda_flow.mlir
flow_cuda "$DYNAMIC_256" /tmp/dynamic_rank3_256_cuda_flow.mlir
grep -n "flow.dispatch .*matmul" /tmp/static_rank3_256_cuda_flow.mlir
grep -n "flow.dispatch .*matmul" /tmp/dynamic_rank3_256_cuda_flow.mlir
grep -n "flow.executable.export public .*matmul" /tmp/static_rank3_256_cuda_flow.mlir
grep -n "flow.executable.export public .*matmul" /tmp/dynamic_rank3_256_cuda_flow.mlir

echo
echo "== CUDA benchmark, 256 =="
compile_cuda "$STATIC_256" /tmp/static_rank3_256_cuda.vmfb
compile_cuda "$DYNAMIC_256" /tmp/dynamic_rank3_256_cuda.vmfb
bench_cuda /tmp/static_rank3_256_cuda.vmfb 4x128x256 256x256
bench_cuda /tmp/dynamic_rank3_256_cuda.vmfb 4x128x256 256x256

echo
echo "== Optional CUDA benchmark, 4096 =="
echo "Set RUN_4096=1 to run the larger case."
if [[ "${RUN_4096:-0}" == "1" ]]; then
  flow_cuda "$STATIC_4096" /tmp/static_rank3_4096_cuda_flow.mlir
  flow_cuda "$DYNAMIC_4096" /tmp/dynamic_rank3_4096_cuda_flow.mlir
  grep -n "flow.dispatch .*matmul" /tmp/static_rank3_4096_cuda_flow.mlir
  grep -n "flow.dispatch .*matmul" /tmp/dynamic_rank3_4096_cuda_flow.mlir
  compile_cuda "$STATIC_4096" /tmp/static_rank3_4096_cuda.vmfb
  compile_cuda "$DYNAMIC_4096" /tmp/dynamic_rank3_4096_cuda.vmfb
  bench_cuda /tmp/static_rank3_4096_cuda.vmfb 4x128x4096 4096x4096
  bench_cuda /tmp/dynamic_rank3_4096_cuda.vmfb 4x128x4096 4096x4096
fi
