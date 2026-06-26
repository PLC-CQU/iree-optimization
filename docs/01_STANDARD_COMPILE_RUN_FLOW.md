# Standard IREE Compile / Run / Benchmark Flow

This note records the standard compile and run workflow used in the project.
It is meant to be followed before investigating pass-level optimizations.

## 1. Tool Paths

Source-built IREE tools:

```text
/home/zhongjialin/projects/iree-build/tools/iree-compile
/home/zhongjialin/projects/iree-build/tools/iree-opt
/home/zhongjialin/projects/iree-build/tools/iree-run-module
/home/zhongjialin/projects/iree-build/tools/iree-benchmark-module
```

Python / pip IREE tools:

```text
/home/zhongjialin/projects/.venv/bin/iree-import-onnx
/home/zhongjialin/projects/.venv/bin/iree-compile
```

The source-built compiler includes the project pass. The `.venv` compiler has
been used as an older/no-pass baseline in Gemma dynamic experiments.

## 2. ONNX -> MLIR

```bash
/home/zhongjialin/projects/.venv/bin/iree-import-onnx \
  /path/to/model.onnx \
  --large-model \
  --externalize-params \
  --num-elements-threshold 2 \
  --param-gb-threshold 2 \
  --save-params-to /path/to/model_params.irpa \
  -o /path/to/model_external.mlir
```

For large models, weights are externalized into `.irpa`. The compiled VMFB is
then run with:

```text
--parameters=model=/path/to/model_params.irpa
```

## 3. Inline Dense Resources

```bash
python3 /home/zhongjialin/projects/iree-optimization/scripts/deepseek/inline_onnx_dense_resources.py \
  /path/to/model_external.mlir \
  -o /path/to/model_external_inlined.mlir
```

Equivalent project-local script:

```text
/home/zhongjialin/projects/iree/deepseek-R1-Llama-8b/inline_onnx_dense_resources.py
```

## 4. MLIR -> CUDA VMFB

```bash
/home/zhongjialin/projects/iree-build/tools/iree-compile \
  --iree-input-type=onnx \
  --iree-input-demote-i64-to-i32 \
  --iree-opt-strip-assertions \
  --iree-hal-target-backends=cuda \
  --iree-cuda-target=sm_86 \
  --iree-codegen-llvmgpu-use-reduction-vector-distribution=false \
  /path/to/model_external_inlined.mlir \
  -o /path/to/model.vmfb
```

Notes from the project:

```text
RTX 4090 runs reliably with --iree-cuda-target=sm_86 in this setup.
sm_89 caused target/runtime issues in earlier tests.
```

## 5. Run

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/zhongjialin/projects/iree-build/tools/iree-run-module \
  --module=/path/to/model.vmfb \
  --parameters=model=/path/to/model_params.irpa \
  --parameter_mode=file \
  --device=cuda \
  --function=main_graph \
  --input=1x32xi64=0 \
  --input=1x32xi64=1
```

If the model has an additional `last_token_indices` input, pass it as:

```bash
--input=1xi64=31
```

or as a `.npy` file, depending on the exported model wrapper.

## 6. Benchmark

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/zhongjialin/projects/iree-build/tools/iree-benchmark-module \
  --module=/path/to/model.vmfb \
  --parameters=model=/path/to/model_params.irpa \
  --parameter_mode=file \
  --device=cuda \
  --function=main_graph \
  --input=1x32xi64=0 \
  --input=1x32xi64=1 \
  --benchmark_repetitions=5 \
  --benchmark_min_time=1x \
  --benchmark_min_warmup_time=0 \
  --benchmark_time_unit=ms \
  --benchmark_out=/tmp/model.googlebench.json \
  --benchmark_out_format=json
```

For quick debugging, use:

```text
--benchmark_repetitions=1
--benchmark_min_time=1x
--benchmark_min_warmup_time=0
```

For more stable numbers, increase repetitions and warmup.

## 7. Compile To Global Optimization IR

This is the standard way to check whether the pass changed the IR:

```bash
/home/zhongjialin/projects/iree-build/tools/iree-compile \
  --iree-input-type=onnx \
  --iree-input-demote-i64-to-i32 \
  --iree-opt-strip-assertions \
  --compile-to=global-optimization \
  /path/to/model_external_inlined.mlir \
  -o /tmp/model_global_opt.mlir
```

Count important ops:

```bash
python3 - <<'PY'
from pathlib import Path
import re

text = Path("/tmp/model_global_opt.mlir").read_text(errors="replace")
for name in [
    "linalg.matmul",
    "linalg.batch_matmul",
    "linalg.generic",
    "tensor.collapse_shape",
    "tensor.expand_shape",
]:
    print(name, len(re.findall(r"\b" + re.escape(name) + r"\b", text)))
PY
```

Expected optimized behavior:

```text
more linalg.matmul
fewer linalg.batch_matmul / linalg.generic matmul-like regions
more collapse_shape / expand_shape around matmul
```

## 8. Minimal First Reproduction

Pass test:

```bash
/home/zhongjialin/projects/iree-build/tools/iree-opt \
  --split-input-file \
  --pass-pipeline='builtin.module(func.func(iree-global-opt-flatten-rank3-matmul))' \
  /home/zhongjialin/projects/iree/compiler/src/iree/compiler/GlobalOptimization/test/flatten_rank3_matmul.mlir
```

Gemma dynamic model test:

```bash
cd /home/zhongjialin/projects/iree/Gemma

/home/zhongjialin/projects/iree/Gemma/run_gemma_dynamic_shape_experiment.py \
  --model-path /home/zhongjialin/projects/iree/Gemma/googlegemma-4-E2B-it \
  --experiment-name gemma_e2b_dynamic_bs_fp16_cuda \
  --export-device cpu \
  --gpu 2 \
  --shapes b1_s32 \
  --repetitions 3 \
  --min-time 1x \
  --warmup-time 0 \
  --benchmark-timeout-seconds 180 \
  --action all
```
