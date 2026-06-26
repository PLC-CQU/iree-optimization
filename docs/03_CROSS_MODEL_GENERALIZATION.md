# Other Model Generalization Report

## 1. Purpose

This note pauses the DeepSeek dynamic-shape attention-context optimization and checks whether the core flatten MatMul idea is specific to DeepSeek or also applies to other transformer models.

The tested evidence here is for the static/fixed-shape flatten MatMul path, because the current Qwen/Gemma experiment directories contain fixed-shape ONNX/VMFB artifacts rather than dynamic-shape B/S artifacts.

## 2. What Is General And What Is Model-Specific

General part:

```text
MatMul([B,S,K], [K,N]) -> [B,S,N]

rewritten as

Reshape([B,S,K] -> [B*S,K])
MatMul([B*S,K], [K,N])
Reshape([B*S,N] -> [B,S,N])
```

This pattern appears in transformer projection and MLP weights:

```text
q_proj / k_proj / v_proj / o_proj
gate_proj / up_proj / down_proj
lm_head or model projection variants
```

This part is not DeepSeek-specific.

Model/layout-specific part:

```text
dynamic attention context MMA:
(..., M, K) x (..., K, 128) -> (..., M, 128)
```

This part currently assumes a specific attention-context shape and static N=128. It is much more model/layout dependent, and the latest `mma_flat_context` candidate compiled but timed out at runtime even on b1_s32. Therefore it should not be treated as a portable optimization yet.

## 3. Existing Cross-Model Evidence

The repository already contains Qwen and Gemma fixed-shape experiments:

```text
/home/zhongjialin/projects/iree/Qwen/experiments
/home/zhongjialin/projects/iree/Gemma/experiments
```

Each experiment contains:

```text
standard_cuda/*.vmfb
standard_cuda_flatten_matmul/*.vmfb
flatten_matmul_rewrite_report.json
benchmark_repeated_single_compare.json
```

## 4. Rewrite Coverage

### Qwen2.5-3B

Example:

```text
qwen25_3b_b4_s32_fp16_cuda_standard
```

Rewrite report:

```text
matmul_total: 326
weight_matmul_total: 253
rewritten_weight_matmul_nodes: 253
skipped_matmul_by_reason:
  rhs_not_initializer: 73
```

This means all MatMul ops whose RHS is a static initializer weight and whose shape matches `[B,S,K] x [K,N]` were rewritten.

### Gemma E4B

Example:

```text
gemma_e4b_it_b4_s32_fp16_cuda_standard
```

Rewrite report:

```text
matmul_total: 430
weight_matmul_total: 344
rewritten_weight_matmul_nodes: 344
skipped_matmul_by_reason:
  rhs_not_initializer: 86
```

Gemma has even more matching weight MatMul ops than Qwen2.5-3B, and the same rewrite rule applies cleanly.

## 5. Performance Summary

| Model | Experiment | B | S | Rewritten MatMul | Baseline ms | Flatten ms | Speedup | Latency Reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma | gemma_e4b_it_b4_s32_fp16_cuda_standard | 4 | 32 | 344 | 157.200 | 48.150 | 3.265x | 69.370% |
| Gemma | gemma_e4b_it_b4_s64_fp16_cuda_standard | 4 | 64 | 344 | 157.100 | 55.250 | 2.843x | 64.831% |
| Gemma | gemma_e4b_it_b4_s128_fp16_cuda_standard | 4 | 128 | 344 | 208.400 | 118.800 | 1.754x | 42.994% |
| Qwen | qwen25_3b_b4_s32_fp16_cuda_standard | 4 | 32 | 253 | 93.700 | 27.390 | 3.421x | 70.768% |
| Qwen | qwen25_3b_b4_s64_fp16_cuda_standard | 4 | 64 | 253 | 95.050 | 31.780 | 2.991x | 66.565% |
| Qwen | qwen25_3b_b4_s128_fp16_cuda_standard | 4 | 128 | 253 | 111.000 | 54.290 | 2.045x | 51.090% |
| Qwen | qwen25_3b_b8_s32_fp16_cuda_standard | 8 | 32 | 253 | 185.000 | 31.070 | 5.954x | 83.205% |
| Qwen | qwen25_3b_b8_s64_fp16_cuda_standard | 8 | 64 | 253 | 192.800 | 52.890 | 3.645x | 72.567% |
| Qwen | qwen25_3b_b8_s128_fp16_cuda_standard | 8 | 128 | 253 | 203.800 | 70.720 | 2.882x | 65.299% |

## 6. Interpretation

The core flatten MatMul optimization is not limited to DeepSeek.

Evidence:

```text
Qwen2.5-3B:
  253 weight MatMul ops rewritten.
  measured speedup: 2.045x to 5.954x.

Gemma E4B:
  344 weight MatMul ops rewritten.
  measured speedup: 1.754x to 3.265x.
```

This matches the expected transformer-wide pattern: large numbers of rank-3 activation x rank-2 weight MatMuls exist in Qwen and Gemma too.

However, the dynamic DeepSeek attention-context MMA path is still not proven portable.

Current status:

```text
safe_dynamic_m / dynamic-M projection style:
  likely general for dynamic transformer projection and MLP MatMuls.

attention-context dynamic-K MMA:
  not yet stable.
  model/layout dependent.
  should remain experimental.
```

## 7. Reproduction Commands

Summarize existing results:

```bash
cd /home/zhongjialin/projects

python3 - <<'PY'
import json, statistics
from pathlib import Path

roots = [
    Path('/home/zhongjialin/projects/iree/Gemma/experiments'),
    Path('/home/zhongjialin/projects/iree/Qwen/experiments'),
]

def load(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def mean_from_rows(data, variant):
    vals = []
    for r in data.get('rows', data.get('runs', [])):
        if r.get('variant') == variant and r.get('status') == 'ok':
            if isinstance(r.get('latency_ms'), (int, float)):
                vals.append(float(r['latency_ms']))
    return statistics.mean(vals) if vals else None

def summarize_bench(data):
    if not data:
        return None
    perf = data.get('performance')
    if isinstance(perf, dict) and 'baseline' in perf and 'flatten_matmul' in perf:
        b = perf['baseline'].get('latency_ms_mean')
        f = perf['flatten_matmul'].get('latency_ms_mean')
        if b and f:
            return float(b), float(f)
    b = mean_from_rows(data, 'baseline')
    f = mean_from_rows(data, 'flatten_matmul')
    return (b, f) if b and f else None

def rewritten_count(report):
    if not report:
        return None
    return report.get('rewritten_weight_matmul_nodes')

print('| Model | Experiment | B | S | Rewritten MatMul | Baseline ms | Flatten ms | Speedup | Latency Reduction |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for root in roots:
    model = 'Gemma' if 'Gemma' in str(root) else 'Qwen'
    for exp in sorted(root.iterdir() if root.exists() else []):
        if not exp.is_dir():
            continue
        meta = load(exp / 'export_metadata.json') or {}
        bench = summarize_bench(load(exp / 'benchmark_repeated_single_compare.json'))
        report = load(exp / 'flatten_matmul_rewrite_report.json') or load(exp / 'rewrite_report.json')
        rewritten = rewritten_count(report)
        if not bench:
            continue
        base_ms, flat_ms = bench
        speedup = base_ms / flat_ms
        reduction = (1 - flat_ms / base_ms) * 100
        print(f'| {model} | {exp.name} | {meta.get("batch")} | {meta.get("seq")} | {rewritten} | {base_ms:.3f} | {flat_ms:.3f} | {speedup:.3f}x | {reduction:.3f}% |')
PY
```

Rerun a Qwen benchmark:

```bash
cd /home/zhongjialin/projects/iree/Qwen

CUDA_VISIBLE_DEVICES=0 \
python3 /home/zhongjialin/projects/iree-optimization/scripts/qwen/run_qwen_repeated_single_benchmark.py \
  --experiment-name qwen25_3b_b4_s32_fp16_cuda_standard \
  --batch 4 \
  --seq 32 \
  --runs 10 \
  --gpu 0
```

Rerun a Gemma benchmark:

```bash
cd /home/zhongjialin/projects/iree/Gemma

CUDA_VISIBLE_DEVICES=0 \
python3 /home/zhongjialin/projects/iree-optimization/scripts/gemma/run_gemma_repeated_single_benchmark.py \
  --experiment-name gemma_e4b_it_b4_s32_fp16_cuda_standard \
  --batch 4 \
  --seq 32 \
  --runs 10 \
  --gpu 0
```

## 8. Current Conclusion

The optimization should be described in two layers:

```text
Layer 1: flatten rank-3 activation x rank-2 weight MatMul
  general across transformer LLMs.
  already validated on DeepSeek, Qwen2.5, and Gemma.

Layer 2: dynamic-shape attention context MMA
  not general yet.
  currently DeepSeek/Llama-layout motivated.
  latest candidate compiles but runtime timeout shows it is not production-ready.
```

Therefore the broader claim should be:

```text
The main flatten MatMul optimization is not DeepSeek-only.
It applies to common transformer projection/MLP MatMuls and has measured benefit on Qwen and Gemma.

The more aggressive dynamic attention optimization is still experimental and should not be claimed as generally effective yet.
```
