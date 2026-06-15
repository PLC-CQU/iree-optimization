# 动态 Shape Matmul 最小实验

本目录用一组小型 MLIR case 隔离动态 shape matmul 的性能问题。目标不是复刻完整 DeepSeek 图，而是把问题拆成更容易观察的结构：单个 matmul、QKV projection、reshape/transpose、attention scores、MLP gate/up/down，以及 guarded specialization。

## 文件说明

```text
static_rank3_matmul.mlir
dynamic_rank3_matmul.mlir
  [4,128,4096] x [4096,4096] 与 [?,?,4096] x [4096,4096] 的对比。

static_rank3_matmul_256.mlir
dynamic_rank3_matmul_256.mlir
  更小的 [*,*,256] 版本，方便 CPU / CUDA benchmark。

static_qkv_256.mlir
dynamic_qkv_256.mlir
  三个 projection 共享同一个 [B,S,256] lhs，模拟 Q/K/V 线性层。

static_project_transpose_256.mlir
dynamic_project_transpose_256.mlir
  projection 后接 reshape + transpose。

static_attention_scores_256.mlir
dynamic_attention_scores_256.mlir
  attention scores: [B,H,S,D] x [B,H,D,S] -> [B,H,S,S]。

static_mlp_256.mlir
dynamic_mlp_256.mlir
  MLP gate/up/down projection 与 elementwise 组合。

dynamic_specialized_rank3_matmul_256.mlir
  外部 ABI 保持动态，但内部 cast 到已知静态 shape，用于证明 codegen 关键点是 matmul problem size 是否静态。

dynamic_guarded_specialized_rank3_matmul_256.mlir
  动态函数内用 shape guard 分支：常见 shape 走静态 fast path，其它 shape 走动态 fallback。

summarize_googlebench.py
  汇总 Google Benchmark JSON 输出。
```

## 运行方式

基础 pass / Flow / HAL 对比：

```bash
cd /home/zhongjialin/projects/iree-optimization/experiments/dynamic_shape_matmul_experiment
IREE_BUILD=/path/to/iree-build bash run_compare.sh
IREE_BUILD=/path/to/iree-build bash run_extended_compare.sh
IREE_BUILD=/path/to/iree-build bash run_extended_hal_compare.sh
```

CUDA 对比：

```bash
cd /home/zhongjialin/projects/iree-optimization/experiments/dynamic_shape_matmul_experiment
IREE_BUILD=/path/to/iree-build CUDA_ARCH=sm_86 GPU=0 bash run_cuda_compare.sh
IREE_BUILD=/path/to/iree-build CUDA_ARCH=sm_86 GPU=0 bash run_extended_cuda_benchmark.sh
```

CUDA codegen IR 和 specialization probe：

```bash
IREE_BUILD=/path/to/iree-build CUDA_ARCH=sm_86 bash run_extended_cuda_ir_compare.sh
IREE_BUILD=/path/to/iree-build bash run_specialization_probe.sh
IREE_BUILD=/path/to/iree-build bash run_guarded_specialization_probe.sh
IREE_BUILD=/path/to/iree-build CUDA_ARCH=sm_86 GPU=0 bash run_specialization_cuda_benchmark.sh
```

更稳定的 specialization benchmark：

```bash
IREE_BUILD=/path/to/iree-build CUDA_ARCH=sm_86 GPU=0 \
BENCH_REPETITIONS=10 BENCH_MIN_TIME=10s BENCH_WARMUP_TIME=2.0 \
  bash run_specialization_cuda_benchmark.sh

python3 summarize_googlebench.py \
  /tmp/iree_dynamic_shape_specialization_cuda_benchmark/*.googlebench.json
```

## 主要观察

### 1. Flow 层 dispatch 数量不一定增加

在单个 matmul 和几个缩小版结构中，动态 shape 并没有增加真实 `flow.dispatch` 数量。需要注意：

```bash
grep -c "flow.dispatch" dynamic_flow.mlir
```

会把 `flow.dispatch.tie_shape` 也算进去。更准确的真实 dispatch 计数方式是：

```bash
grep -c " = flow.dispatch " /tmp/static_rank3_flow.mlir
grep -c " = flow.dispatch " /tmp/dynamic_rank3_flow.mlir
```

动态版本的主要额外开销来自 shape-carrying IR，而不是简单的 dispatch 数量增加。

### 2. HAL 层会出现更多动态 shape plumbing

静态版本通常可以使用预构建 / memoized command buffer，并且 workgroup count 与 byte size 是常量。动态版本需要读取 shape、计算 runtime size/workgroup，并把更多 shape 常量传给 dispatch。

典型区别：

```text
static:
  indirect execute
  workgroup count / binding size 多为常量

dynamic:
  direct execute
  runtime dim load
  runtime shape arithmetic
  更多 dispatch constants
```

### 3. CUDA codegen pipeline 是关键差异

动态 matmul 变慢的一个核心原因是：动态 shape 下，matmul 可能从静态版本的 `TileAndFuse` / MMA 路径退到更通用的 `VectorDistribute`。

缩小版 CUDA benchmark 中观察到的趋势：

```text
01_single_matmul:                dynamic 明显慢于 static
02_qkv_shared_lhs:               dynamic 明显慢于 static
03_projection_reshape_transpose: dynamic 明显慢于 static
04_attention_scores:             dynamic 带更多 shape constants
05_mlp_gate_up_down:             dynamic 最容易因为多个 matmul fallback 而放大差距
```

这说明优化目标不只是减少 dispatch，而是要让常见动态形状的 matmul 能恢复接近静态的 CUDA lowering。

### 4. 内部恢复静态 problem size 可以恢复大部分性能

`dynamic_specialized_rank3_matmul_256.mlir` 证明：即使外部函数签名仍然是动态 ABI，只要内部 matmul 看到的是静态 problem size，就能恢复 `TileAndFuse` / MMA 路径。

`dynamic_guarded_specialized_rank3_matmul_256.mlir` 进一步模拟了理想优化形式：

```mlir
%b = tensor.dim %lhs, %c0 : tensor<?x?x256xf32>
%s = tensor.dim %lhs, %c1 : tensor<?x?x256xf32>
%is_b4 = arith.cmpi eq, %b, %c4 : index
%is_s128 = arith.cmpi eq, %s, %c128 : index
%is_target_shape = arith.andi %is_b4, %is_s128 : i1
%out = scf.if %is_target_shape -> tensor<?x?x256xf32> {
  // static fast path
} else {
  // dynamic fallback
}
```

较长 CUDA benchmark 中，guarded static fast path 相比纯动态版本有明显加速。这说明一个可行方向是：入口保持动态，但在编译器内部为常见 shape 恢复静态 matmul kernel。

## 与当前项目的关系

这些最小实验支撑 DeepSeek 动态 shape 优化中的几个判断：

- 单个动态 matmul 不一定因为 dispatch 数量变多而慢。
- 真正关键的是 CUDA lowering pipeline、动态 shape arithmetic、dispatch constants 和 command submission 形式。
- projection / MLP / lm_head 这类 safe dynamic-M matmul 是第一阶段稳定优化对象。
- attention context 更复杂，需要保留 fallback 或用更保守的 SIMT TileAndFuse 方案，不能简单强行套用普通 dynamic-M MMA heuristic。
