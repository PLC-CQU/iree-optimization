# IREE LLM MatMul Flatten Handoff Package

This folder is the handoff package for the IREE LLM MatMul flatten optimization
project.

Start here:

```text
docs/00_PROJECT_HANDOFF_GUIDE.md
```

That document explains the whole research path:

```text
standard model compile/run
-> static shape flatten optimization
-> IREE pass implementation
-> dynamic shape optimization
-> DeepSeek / Gemma / Qwen validation
-> known limitations and next steps
```

## Directory Layout

```text
docs/
  Main handoff guide and topic-specific technical notes.

docs/troubleshooting/
  Independent issue notes. These record real problems encountered during the
  project and the actual workarounds/fixes used.

src/iree_pass/
  Main IREE GlobalOptimization pass source and minimal MLIR test.

src/codegen_dynamic/
  Dynamic shape / CUDA codegen files touched during the dynamic-M exploration.

src/gemma/
  Gemma compile/benchmark scripts, including the dynamic B/S experiment.

scripts/deepseek/
  DeepSeek scripts used for rewrite, benchmark, correctness, bucket routing,
  and external parameter inlining.

scripts/dynamic_shape_experiments/
  Single-op and staged dynamic shape benchmark helpers.

results/
  Small JSON summaries copied from completed experiments. Large VMFB/ONNX/IRPA
  artifacts are intentionally not copied into this handoff folder.
```

## What Is Included

Included:

```text
core pass source
pass registration-related source snapshots
minimal pass test
Gemma dynamic experiment script
DeepSeek benchmark/correctness scripts
dynamic shape experiment scripts
technical reports
small JSON summaries
troubleshooting notes
```

Not included:

```text
large .onnx models
large .vmfb files
large .irpa parameter archives
large generated MLIR dumps
```

Those large artifacts remain in their original experiment directories and are
referenced from the documents.

## Recommended First Run

1. Read the main guide:

```bash
less /home/zhongjialin/projects/iree_llm_matmul_flatten_handoff/docs/00_PROJECT_HANDOFF_GUIDE.md
```

2. Verify the pass on the minimal MLIR test:

```bash
/home/zhongjialin/projects/iree-build/tools/iree-opt \
  --split-input-file \
  --pass-pipeline='builtin.module(func.func(iree-global-opt-flatten-rank3-matmul))' \
  /home/zhongjialin/projects/iree/compiler/src/iree/compiler/GlobalOptimization/test/flatten_rank3_matmul.mlir
```

3. Run the Gemma dynamic B/S benchmark:

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

If any step fails, check:

```text
docs/troubleshooting/
```

## Current Headline Results

DeepSeek dynamic shape:

```text
b1_s32:
  old_dynamic:      3364.666 ms
  safe_dynamic_m:     82.213 ms
  static_exact:        37.309 ms
  speedup:             40.926x
```

Gemma E2B dynamic B/S:

```text
b1_s32: 136.603 ms -> 25.102 ms, 5.442x
b1_s64: 198.063 ms -> 35.870 ms, 5.522x
b4_s32: 396.723 ms -> 40.125 ms, 9.887x
```

Gemma IR evidence:

```text
baseline global opt:
  linalg.matmul:       0
  linalg.batch_matmul: 279

optimized global opt:
  linalg.matmul:       277
  linalg.batch_matmul: 2
```

## Most Important Limitation

The mature optimization is:

```text
[B,S,K] x [K,N] -> [B,S,N]
flattened as
[B*S,K] x [K,N] -> [B*S,N]
```

This covers projection / MLP / lm_head style matmuls.

Dynamic attention context is not solved yet. It often has dynamic reduction K,
which makes MMA lowering harder and previously caused timeout/hang in some
experiments.
