# Troubleshooting: Dynamic Attention Shape Issues

## 1. What Worked

The stable dynamic optimization is dynamic-M matmul:

```text
[?, K_static] x [K_static, N_static]
```

This covers projection / MLP / lm_head after flattening:

```text
[B,S,K] x [K,N] -> [B*S,K] x [K,N]
```

DeepSeek and Gemma both showed large speedups from this path.

## 2. What Did Not Fully Work

Dynamic attention context is harder:

```text
[B,H,S,S] x [B,H,S,D] -> [B,H,S,D]
```

Here dynamic `S` appears in reduction K as well as output dimensions.

Earlier experiments tried to push dynamic attention context toward MMA, but
some generated VMFBs timed out or appeared to hang at runtime.

Example observed behavior:

```bash
timeout 20s \
env CUDA_VISIBLE_DEVICES=2 \
/home/zhongjialin/projects/iree-build/tools/iree-run-module \
  --module=/tmp/iree_deepseek_attention_cuda_after_patch/dynamic_context.vmfb \
  --device=cuda \
  --function=main \
  --input=128x104x104xf16=1 \
  --input=128x104x128xf16=1
```

Exit status:

```text
124
```

Meaning the command timed out.

## 3. Current Boundary

Do not mix the attention-context experiments into the main flatten-rank3
matmul conclusion.

Safe conclusion:

```text
Projection / MLP dynamic-M matmul is stable and effective.
```

Open problem:

```text
Dynamic attention context with dynamic reduction K still needs separate work.
```

## 4. Relevant Experiment Directory

```text
/home/zhongjialin/projects/iree/dynamic_shape_matmul_experiment
```

Copied helper scripts:

```text
scripts/dynamic_shape_experiments/
```

## 5. Recommended Next Investigation

Start from small IR, not full model:

```text
1. Single dynamic attention scores matmul.
2. Single dynamic attention context matmul.
3. Dynamic reduction K with SIMT fallback.
4. Padding-to-tile-size experiment.
5. Guarded specialization by seq range.
```

Only after a small kernel is stable should it be tested in the full model.
