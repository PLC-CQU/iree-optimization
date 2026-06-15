#!/usr/bin/env bash
set -euo pipefail

IREE_BUILD="${IREE_BUILD:-/home/zhongjialin/projects/iree-build}"
CUDA_ARCH="${CUDA_ARCH:-sm_86}"
GPU="${GPU:-0}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/iree_dynamic_shape_guarded_specialization_probe}"
mkdir -p "$OUT_DIR"

compile_config() {
  "$IREE_BUILD/tools/iree-compile" \
    --iree-hal-target-backends=cuda \
    --iree-cuda-target="$CUDA_ARCH" \
    --iree-gpu-test-target="$CUDA_ARCH" \
    --compile-to=executable-configurations \
    "$ROOT/dynamic_guarded_specialized_rank3_matmul_256.mlir" \
    -o "$OUT_DIR/guarded_exec_config.mlir"
}

compile_hal() {
  "$IREE_BUILD/tools/iree-compile" \
    --iree-hal-target-backends=cuda \
    --iree-cuda-target="$CUDA_ARCH" \
    --iree-gpu-test-target="$CUDA_ARCH" \
    --compile-to=hal \
    "$ROOT/dynamic_guarded_specialized_rank3_matmul_256.mlir" \
    -o "$OUT_DIR/guarded_hal.mlir"
}

compile_vmfb() {
  "$IREE_BUILD/tools/iree-compile" \
    --iree-hal-target-backends=cuda \
    --iree-cuda-target="$CUDA_ARCH" \
    --iree-gpu-test-target="$CUDA_ARCH" \
    "$ROOT/dynamic_guarded_specialized_rank3_matmul_256.mlir" \
    -o "$OUT_DIR/guarded.vmfb"
}

bench_vmfb() {
  CUDA_VISIBLE_DEVICES="$GPU" "$IREE_BUILD/tools/iree-benchmark-module" \
    --module="$OUT_DIR/guarded.vmfb" \
    --device=cuda \
    --function=main \
    --input=4x128x256xf32=1 \
    --input=256x256xf32=1 \
    --benchmark_repetitions=5 \
    --benchmark_min_time=1s \
    --benchmark_time_unit=us
}

compile_config
compile_hal

echo "== guarded executable configuration =="
grep -n "hal.executable.export public" "$OUT_DIR/guarded_exec_config.mlir" || true
grep -n "translation_info" "$OUT_DIR/guarded_exec_config.mlir" || true
grep -n "lowering_config" "$OUT_DIR/guarded_exec_config.mlir" || true

echo
echo "== guarded HAL =="
grep -n "hal.command_buffer.dispatch" "$OUT_DIR/guarded_hal.mlir" || true
grep -n "hal.device.queue.execute.indirect" "$OUT_DIR/guarded_hal.mlir" || true
grep -n "hal.device.queue.execute<" "$OUT_DIR/guarded_hal.mlir" || true

echo
echo "Files:"
echo "  $OUT_DIR/guarded_exec_config.mlir"
echo "  $OUT_DIR/guarded_hal.mlir"

if [[ "${RUN_BENCH:-0}" == "1" ]]; then
  compile_vmfb
  echo
  echo "== guarded CUDA benchmark, input 4x128x256 =="
  bench_vmfb
fi
