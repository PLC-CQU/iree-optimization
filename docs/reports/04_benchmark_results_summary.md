# 性能结果与实验结论汇总

本文汇总当前仓库中保留的核心性能结果，并说明这些结果支持哪些技术判断。

结果文件位于：

```text
results/deepseek/
results/figures/
```

## 1. Graph-level flatten matmul 结果

结果文件：

```text
results/deepseek/benchmark_b4_b8_seq32_64_128_flatten_matmul_summary.json
```

实验对象：

```text
DeepSeek-R1-Llama-8B
IREE CUDA
Graph-level MatMul flatten
B = 4 / 8
S = 32 / 64 / 128
```

总体结果：

```text
num_shapes: 6
num_completed: 6
best_speedup: 10.746x
min_speedup: 3.001x
mean_speedup: 6.155x
```

逐 shape 结果：

| Shape | baseline latency | flatten latency | speedup | latency reduction |
|---|---:|---:|---:|---:|
| b4_s32 | 314.07 ms | 39.53 ms | 7.94x | 87.41% |
| b4_s64 | 308.12 ms | 60.68 ms | 5.08x | 80.31% |
| b4_s128 | 342.77 ms | 114.23 ms | 3.00x | 66.68% |
| b8_s32 | 648.72 ms | 60.37 ms | 10.75x | 90.69% |
| b8_s64 | 608.49 ms | 111.46 ms | 5.46x | 81.68% |
| b8_s128 | 681.37 ms | 144.88 ms | 4.70x | 78.74% |

对应图：

```text
results/figures/benchmark_b4_b8_seq32_64_128_flatten_matmul_latency.png
results/figures/benchmark_b4_b8_seq32_64_128_flatten_matmul_tokens_per_second.png
```

结论：

```text
rank3 matmul flatten 对 projection / MLP / lm_head 这类权重 matmul 有稳定收益。
收益在小 seq 和大 batch 场景更明显。
seq 增大后 baseline 与 optimized 的差距缩小，但 optimized 仍保持明显优势。
```

## 2. B4/B8/B16 扩展结果

结果文件：

```text
results/deepseek/benchmark_b4_b8_b16_extended_flatten_matmul_summary.json
```

该文件保留了更大 batch 范围下的 flatten matmul 汇总，用来观察 batch 扩展时的趋势。它主要用于说明：

```text
flatten matmul 优化不是单个 shape 的偶然结果。
随着 batch/seq 变化，优化收益仍然存在。
不同 shape 的收益幅度受 matmul / attention / memory 行为比例影响。
```

## 3. 低层 codegen flag 对比

结果文件：

```text
results/deepseek/benchmark_b8_s64_optimization_compare.json
```

实验 shape：

```text
B = 8
S = 64
CUDA target = sm_86
```

baseline：

```text
latency: 602.819 ms
items/s: 1.659
```

候选优化：

| Variant | latency | speedup vs baseline | 说明 |
|---|---:|---:|---|
| prefetch1 | 602.36 ms | 1.0008x | 几乎持平 |
| prefetch3 | 603.03 ms | 0.9997x | 几乎持平 |
| vector_distribution | 603.17 ms | 0.9994x | 略慢 |
| shared_memory_reuse | 603.30 ms | 0.9992x | 略慢 |
| vector_shared_prefetch2 | 603.13 ms | 0.9995x | 几乎持平 |

结论：

```text
单独调 prefetch、vector distribution、shared memory reuse 这类底层 flag，不能解决主要瓶颈。
核心收益来自 IR 结构和 codegen pipeline 选择，而不是局部参数微调。
```

这也是项目转向 `rank3 -> rank2 flatten`、safe dynamic-M、attention context fallback 的原因。

## 4. Shape sweep 与 granularity 结果

保留文件：

```text
results/deepseek/benchmark_seq_sweep_compare.json
results/deepseek/benchmark_shape_granularity_compare.json
```

这些结果用于观察：

```text
不同 sequence length 下 latency / throughput 的变化。
不同 shape granularity 对 benchmark 结果的影响。
```

它们不是主结论来源，但用于辅助判断优化是否只对单个固定 shape 有效。

## 5. 正确性结果与性能结果的对应关系

性能收益必须和正确性结果一起看。当前保留的 correctness 文件显示：

```text
b4_s32:  max_abs_error = 0.0
b8_s64:  max_abs_error = 0.0
b8_s128: max_abs_error = 0.0
```

也就是说，保留的 flatten matmul 性能收益不是以 logits 变化为代价得到的。

## 6. 对当前优化方向的支撑

这些结果支持以下技术判断：

```text
1. Graph-level matmul flatten 是确定有效的主线优化。
2. 低层 flag 微调收益很小，不是主要方向。
3. 动态 shape 性能问题需要从 IR 结构和 CUDA lowering pipeline 入手。
4. projection / MLP / lm_head 的 safe dynamic-M 是稳定第一阶段。
5. attention context 仍是后续缩小动态与静态差距的重点。
```

## 7. 还需要补充的结果

后续建议补充：

```text
safe_dynamic_m 在 b1/b4/b8 多个 shape 下的统一 JSON 汇总。
attention context SIMT fallback 的独立 correctness + benchmark 表。
Qwen / Gemma 跨模型对照结果。
dynamic rank-5 attention MMA failure / timeout 的最小复现日志。
```

这些可以继续放入 `docs/reports/`，而大体积中间产物仍然不进入 Git。
