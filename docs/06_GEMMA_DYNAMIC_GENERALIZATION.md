# Gemma Dynamic Shape Flatten MatMul Experiment

Date: 2026-06-18

## Goal

Test whether the dynamic-shape flatten MatMul optimization is specific to
DeepSeek, or whether it also works on another Transformer model.

This experiment uses Gemma E2B with dynamic batch and sequence dimensions:

```text
input_ids:      tensor<?x?xi64>
attention_mask: tensor<?x?xi64>
output logits:  tensor<?x262144xf32>
```

The test compares:

- `baseline_dynamic`: pip/venv IREE compiler without the local
  `iree-global-opt-flatten-rank3-matmul` pass.
- `optimized_dynamic`: current source-built IREE compiler with the pass in the
  global optimization pipeline.

## Setup

Dynamic ONNX export:

```bash
cd /home/zhongjialin/projects/iree/Gemma

/home/zhongjialin/projects/iree/Gemma/run_gemma_dynamic_shape_experiment.py \
  --model-path /home/zhongjialin/projects/iree/Gemma/googlegemma-4-E2B-it \
  --experiment-name gemma_e2b_dynamic_bs_fp16_cuda \
  --export-device cpu \
  --gpu 2 \
  --shapes b1_s32 \
  --action compile
```

The export uses CPU because the current Python environment can load the newer
Gemma4 transformers code through `/home/zhongjialin/projects/iree/Qwen/pydeps`,
but PyTorch CUDA is not visible from that Python environment. This does not
affect IREE CUDA benchmarking; benchmarking is run with `iree-benchmark-module`
on CUDA.

Generated artifacts:

```text
/home/zhongjialin/projects/iree/Gemma/dynamic_experiments/gemma_e2b_dynamic_bs_fp16_cuda/gemma_dynamic_last_token.onnx
/home/zhongjialin/projects/iree/Gemma/dynamic_experiments/gemma_e2b_dynamic_bs_fp16_cuda/imported/gemma_dynamic_external_inlined.mlir
/home/zhongjialin/projects/iree/Gemma/dynamic_experiments/gemma_e2b_dynamic_bs_fp16_cuda/imported/gemma_dynamic_params.irpa
/home/zhongjialin/projects/iree/Gemma/dynamic_experiments/gemma_e2b_dynamic_bs_fp16_cuda/baseline_dynamic/gemma_baseline_dynamic.vmfb
/home/zhongjialin/projects/iree/Gemma/dynamic_experiments/gemma_e2b_dynamic_bs_fp16_cuda/optimized_dynamic/gemma_optimized_dynamic.vmfb
```

## Reproduction Commands

Compile both dynamic variants:

```bash
cd /home/zhongjialin/projects/iree/Gemma

/home/zhongjialin/projects/iree/Gemma/run_gemma_dynamic_shape_experiment.py \
  --model-path /home/zhongjialin/projects/iree/Gemma/googlegemma-4-E2B-it \
  --experiment-name gemma_e2b_dynamic_bs_fp16_cuda \
  --export-device cpu \
  --gpu 2 \
  --shapes b1_s32 \
  --repetitions 1 \
  --min-time 1x \
  --warmup-time 0 \
  --benchmark-timeout-seconds 120 \
  --action compile
```

Run `b1_s32` benchmark and correctness:

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

Run additional dynamic shapes:

```bash
cd /home/zhongjialin/projects/iree/Gemma

/home/zhongjialin/projects/iree/Gemma/run_gemma_dynamic_shape_experiment.py \
  --model-path /home/zhongjialin/projects/iree/Gemma/googlegemma-4-E2B-it \
  --experiment-name gemma_e2b_dynamic_bs_fp16_cuda \
  --export-device cpu \
  --gpu 2 \
  --shapes b1_s64 b4_s32 \
  --repetitions 3 \
  --min-time 1x \
  --warmup-time 0 \
  --benchmark-timeout-seconds 180 \
  --action all
```

Generate global optimization IR for structural comparison:

```bash
/home/zhongjialin/projects/.venv/bin/iree-compile \
  --iree-input-type=onnx \
  --iree-input-demote-i64-to-i32 \
  --iree-opt-strip-assertions \
  --compile-to=global-optimization \
  /home/zhongjialin/projects/iree/Gemma/dynamic_experiments/gemma_e2b_dynamic_bs_fp16_cuda/imported/gemma_dynamic_external_inlined.mlir \
  -o /tmp/gemma_e2b_dynamic_baseline_global.mlir

/home/zhongjialin/projects/iree-build/tools/iree-compile \
  --iree-input-type=onnx \
  --iree-input-demote-i64-to-i32 \
  --iree-opt-strip-assertions \
  --compile-to=global-optimization \
  /home/zhongjialin/projects/iree/Gemma/dynamic_experiments/gemma_e2b_dynamic_bs_fp16_cuda/imported/gemma_dynamic_external_inlined.mlir \
  -o /tmp/gemma_e2b_dynamic_optimized_global.mlir
```

Count key ops:

```bash
python3 - <<'PY'
from pathlib import Path
import re

for label, path in [
    ("baseline", "/tmp/gemma_e2b_dynamic_baseline_global.mlir"),
    ("optimized", "/tmp/gemma_e2b_dynamic_optimized_global.mlir"),
]:
    text = Path(path).read_text(errors="replace")
    print(label)
    print("  linalg.matmul:", len(re.findall(r"\\blinalg\\.matmul\\b", text)))
    print("  linalg.batch_matmul:", len(re.findall(r"\\blinalg\\.batch_matmul\\b", text)))
    print("  linalg.generic:", len(re.findall(r"\\blinalg\\.generic\\b", text)))
    print("  tensor.collapse_shape:", len(re.findall(r"\\btensor\\.collapse_shape\\b", text)))
    print("  tensor.expand_shape:", len(re.findall(r"\\btensor\\.expand_shape\\b", text)))
PY
```

## Performance Results

All benchmark results are from CUDA on GPU 2, with:

```text
--benchmark_repetitions=3
--benchmark_min_time=1x
--benchmark_min_warmup_time=0
```

| Shape | Baseline Dynamic | Optimized Dynamic | Speedup |
| --- | ---: | ---: | ---: |
| b1_s32 | 136.603 ms | 25.102 ms | 5.442x |
| b1_s64 | 198.063 ms | 35.870 ms | 5.522x |
| b4_s32 | 396.723 ms | 40.125 ms | 9.887x |

## Correctness

Correctness compares `baseline_dynamic` and `optimized_dynamic` outputs with:

```text
atol=0.06
rtol=0.01
```

| Shape | Output Shape | Max Abs | Mean Abs | Allclose |
| --- | ---: | ---: | ---: | --- |
| b1_s32 | 1x262144 | 0.046875 | 0.00561959 | true |
| b1_s64 | 1x262144 | 0.0546875 | 0.00931777 | true |
| b4_s32 | 4x262144 | 0.046875 | 0.00561959 | true |

## IR Evidence

The global optimization IR shows the optimized compiler is not merely faster by
accident. It structurally rewrites Gemma's dynamic rank-3/batched matmul pattern
into ordinary matmul form.

| IR | `linalg.matmul` | `linalg.batch_matmul` | `linalg.generic` | `collapse_shape` | `expand_shape` |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline global opt | 0 | 279 | 4231 | 59 | 54 |
| optimized global opt | 277 | 2 | 3954 | 271 | 331 |

This is the important proof point:

```text
baseline dynamic:
  most projection/MLP rank-3 matmuls remain as dynamic batch_matmul-like forms

optimized dynamic:
  277 of those sites become linalg.matmul, surrounded by collapse/expand shape
```

That matches the intended optimization:

```text
[B, S, K] x [K, N] -> [B, S, N]

becomes:

[B*S, K] x [K, N] -> [B*S, N] -> [B, S, N]
```

The transformation works even when `B` and `S` are dynamic because the flattened
`B*S` dimension is materialized as SSA index arithmetic and passed into
`tensor.collapse_shape` / `tensor.expand_shape`.

## Conclusion

This experiment shows the dynamic flatten MatMul optimization is not limited to
DeepSeek.

For Gemma E2B dynamic B/S:

- The pass compiles successfully.
- The same VMFB runs multiple dynamic shapes: `b1_s32`, `b1_s64`, `b4_s32`.
- Output correctness passes against the no-pass dynamic baseline.
- Performance improves by `5.44x` to `9.89x`.
- IR confirms the expected rewrite: `279` dynamic batch matmul-like sites are
  reduced to only `2`, while `277` ordinary `linalg.matmul` ops appear.

So the core optimization should be treated as a general Transformer projection
and MLP optimization, not as a DeepSeek-only workaround.

The more restricted part is still the separate dynamic attention-context path.
That should remain a separate topic and should not be mixed into the conclusion
for this flatten-rank3-matmul optimization.
