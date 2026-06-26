# Troubleshooting: Standard Compile / Run Issues

This note records issues actually encountered during the project and the
workarounds used.

## 1. Pass Not Registered

Observed error:

```text
'iree-global-opt-flatten-rank3-matmul' does not refer to a registered pass or pass pipeline
```

Where it appeared:

```bash
iree-opt --split-input-file \
  --pass-pipeline='builtin.module(func.func(iree-global-opt-flatten-rank3-matmul))' \
  compiler/src/iree/compiler/GlobalOptimization/test/flatten_rank3_matmul.mlir
```

Fix:

```text
1. Define the pass in Passes.td.
2. Implement createFlattenRank3MatmulPass.
3. Include the .cpp in CMakeLists.txt / BUILD.bazel.
4. Register it through Passes.cpp / generated pass registration.
5. Rebuild iree-opt / iree-compile.
```

Relevant files:

```text
compiler/src/iree/compiler/GlobalOptimization/FlattenRank3Matmul.cpp
compiler/src/iree/compiler/GlobalOptimization/Passes.td
compiler/src/iree/compiler/GlobalOptimization/Passes.cpp
compiler/src/iree/compiler/GlobalOptimization/CMakeLists.txt
compiler/src/iree/compiler/GlobalOptimization/BUILD.bazel
```

## 2. `rg` Not Installed

Observed error:

```text
Command 'rg' not found
```

Workaround:

```bash
grep -E "linalg\\.(batch_)?matmul" /tmp/model_global.mlir
```

or:

```bash
python3 - <<'PY'
from pathlib import Path
import re
text = Path("/tmp/model_global.mlir").read_text(errors="replace")
print("linalg.matmul", len(re.findall(r"\\blinalg\\.matmul\\b", text)))
print("linalg.batch_matmul", len(re.findall(r"\\blinalg\\.batch_matmul\\b", text)))
PY
```

## 3. CUDA Target Problem: sm_89 vs sm_86

Observed during earlier tests:

```text
sm89 target caused runtime/architecture mismatch style failures.
sm86 could run correctly on the server's RTX 4090 setup.
```

Current workaround:

```bash
--iree-cuda-target=sm_86
```

Use this consistently unless the runtime/toolchain is updated and retested.

## 4. CUDA Invalid Context

Observed error:

```text
CUDA_ERROR_INVALID_CONTEXT (201): invalid device context
mismatched target chip? missing/wrong bitcode directory?
```

Practical fixes used:

```text
1. Recompile with --iree-cuda-target=sm_86.
2. Make CUDA_VISIBLE_DEVICES consistent between compile/run/serving tests.
3. Avoid mixing multiple long-running IREE CUDA processes on the same GPU.
4. Restart the serving process after changing GPU/device settings.
```

## 5. CUDA No Device In Sandbox

Observed error:

```text
CUDA_ERROR_NO_DEVICE (100): no CUDA-capable device is detected
```

This happened when running `iree-benchmark-module` inside a sandboxed tool
environment. It is not a model problem.

Fix:

```text
Run the same command directly in the server shell, or run with non-sandbox GPU
access when using automation.
```

## 6. Benchmark Appears Stuck

Observed symptom:

```text
iree-benchmark-module prints benchmark header, then no result for a long time.
nvidia-smi shows 100% GPU utilization.
```

Useful checks:

```bash
nvidia-smi
ps -ef | grep iree-benchmark-module
```

Use timeout for risky dynamic attention experiments:

```bash
timeout 60s \
CUDA_VISIBLE_DEVICES=2 \
/home/zhongjialin/projects/iree-build/tools/iree-run-module \
  --module=/tmp/some_dynamic_attention.vmfb \
  --device=cuda \
  --function=main \
  --input=...
```

Actual conclusion from the project:

```text
Projection / MLP dynamic-M optimization is stable.
Some dynamic attention-context MMA experiments can timeout/hang and should be
treated separately.
```

## 7. Parameter Key Not Found

Observed error:

```text
NOT_FOUND; no parameter found in index with key 'onnx__MatMul_9019'
```

Cause:

```text
The VMFB and .irpa parameter archive came from different import/compile
variants. This can happen when using a newly compiled VMFB with an older params
archive.
```

Fix:

```text
Always pair VMFB with the .irpa generated from the same import path.
For dynamic DeepSeek safe_dynamic_m, use the params from:
dynamic_shape_b_s/build_flatten_matmul_input_i32_with_demote/
```

## 8. CUDA OOM When Loading Many Bucket VMFBs

Observed in C++ serving:

```text
failed to load b4_s64: CUDA_ERROR_OUT_OF_MEMORY
```

Cause:

```text
Loading multiple large bucket models and parameter archives into one GPU
context can exceed memory.
```

Workarounds used:

```text
1. Split bucket models across multiple GPUs.
2. Load fewer buckets per process.
3. Prefer dynamic single-VMFB route for the research path.
```
