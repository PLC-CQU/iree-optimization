# Troubleshooting: Correctness Issues

This note records correctness problems encountered during the project and the
actual fixes used.

## 1. Static Bucket Position IDs Bug

Observed problem:

```text
Dynamic IREE path and static bucket IREE path produced mismatched outputs for
some shapes.
```

Root cause found:

```text
In fixed static ONNX, the no-input position_ids path used:

Constant [1,S] -> Expand [B,S] -> Cast

This triggered an IREE CUDA correctness/layout issue.
```

Fix:

```text
Do not expand RoPE position_ids to batch during static bucket export.
Keep position_ids as [1,S] / [1,S,D].
Let q/k elementwise operations naturally broadcast to batch.
```

Validation:

```text
full grid:
  total: 11
  passed: 11
  failed: 0
  max_abs: 0.04296875
  atol=0.06, rtol=0.01
```

Related report:

```text
docs/05_STATIC_BUCKET_DEPLOYMENT_AND_CORRECTNESS.md
```

## 2. Output Comparison Tolerance

The project usually uses:

```text
atol=0.06
rtol=0.01
```

Examples:

```text
DeepSeek safe_dynamic_m vs static_exact:
  max_abs: 0.0078125
  allclose: True

Gemma dynamic baseline vs optimized:
  b1_s32 max_abs: 0.046875
  b1_s64 max_abs: 0.0546875
  b4_s32 max_abs: 0.046875
  allclose: True
```

## 3. Correctness Commands

DeepSeek safe dynamic-M correctness:

```bash
cd /home/zhongjialin/projects/iree/deepseek-R1-Llama-8b

/home/zhongjialin/projects/iree_llm_matmul_flatten_handoff/scripts/deepseek/check_safe_dynamic_m_correctness.py \
  --gpu 2 --batch 1 --seq 64

/home/zhongjialin/projects/iree_llm_matmul_flatten_handoff/scripts/deepseek/check_safe_dynamic_m_correctness.py \
  --gpu 2 --batch 4 --seq 64

/home/zhongjialin/projects/iree_llm_matmul_flatten_handoff/scripts/deepseek/check_safe_dynamic_m_correctness.py \
  --gpu 2 --batch 8 --seq 128
```

Gemma dynamic correctness is part of:

```bash
cd /home/zhongjialin/projects/iree/Gemma

/home/zhongjialin/projects/iree/Gemma/run_gemma_dynamic_shape_experiment.py \
  --model-path /home/zhongjialin/projects/iree/Gemma/googlegemma-4-E2B-it \
  --experiment-name gemma_e2b_dynamic_bs_fp16_cuda \
  --export-device cpu \
  --gpu 2 \
  --shapes b1_s32 b1_s64 b4_s32 \
  --repetitions 3 \
  --min-time 1x \
  --warmup-time 0 \
  --benchmark-timeout-seconds 180 \
  --action all
```

## 4. When Correctness Fails

Checklist:

```text
1. Confirm VMFB and .irpa are paired from the same build.
2. Confirm input dtype and shape match exported ABI.
3. Confirm position_ids path did not introduce batch expand in static bucket export.
4. Compare against both old_dynamic and static_exact when available.
5. Save outputs to .npy and inspect max_abs / mean_abs, not just exact equality.
```
