# IREE 编译器优化设计说明

本文记录当前项目中 IREE 编译器侧的主要技术改动，以及这些改动解决的问题。

相关文件位于：

```text
iree-patches/tracked_changes.patch
iree-patches/new-files/
iree-patches/source-snapshots/
```

## 1. 问题背景

DeepSeek 动态 shape 模型中，输入 batch 和 sequence 是运行时维度：

```text
input_ids:      tensor<?x?xi32>
attention_mask: tensor<?x?xi32>
```

这些动态维度会传递到 projection、MLP、lm_head 和 attention 相关 matmul 中。原始动态路径常见问题包括：

```text
matmul 不能稳定进入 MMA pipeline
部分 contraction 退到 VectorDistribute
HAL 层出现更多 runtime shape arithmetic
dispatch constants / workload 变多
```

因此优化目标不是简单减少 dispatch 数量，而是尽量恢复动态 shape 下的高质量 CUDA lowering。

## 2. FlattenRank3MatmulPass

新增文件：

```text
compiler/src/iree/compiler/GlobalOptimization/FlattenRank3Matmul.cpp
```

pass 名称：

```text
iree-global-opt-flatten-rank3-matmul
```

### 2.1 识别对象

主要识别：

```text
lhs: [B, S, K]
rhs: [K, N]
out: [B, S, N]
```

数学语义：

```text
out[b, s, n] = sum_k lhs[b, s, k] * rhs[k, n]
```

### 2.2 改写形式

改写为：

```mlir
%flat_lhs = tensor.collapse_shape %lhs [[0, 1], [2]]
%flat_out = tensor.collapse_shape %out [[0, 1], [2]]
%matmul = linalg.matmul ins(%flat_lhs, %rhs) outs(%flat_out)
%expanded = tensor.expand_shape %matmul [[0, 1], [2]]
```

也就是：

```text
[B,S,K] -> [B*S,K]
[B*S,K] x [K,N] -> [B*S,N]
[B*S,N] -> [B,S,N]
```

### 2.3 为什么这样做

IREE CUDA 后端对标准 `linalg.matmul` 的 tiling、vectorization、MMA lowering 更成熟。把 rank-3 contraction 显式转成 rank-2 matmul 后，projection / MLP / lm_head 这类算子更容易进入静态或 safe dynamic-M 的优化路径。

### 2.4 覆盖的 IR 形式

pass 主要覆盖两类 pattern：

```text
rank-3 by rank-2 linalg.generic matmul-like contraction
broadcasted rank-2 weight + batch_matmul 形式
```

测试文件：

```text
compiler/src/iree/compiler/GlobalOptimization/test/flatten_rank3_matmul.mlir
```

## 3. AssumeInputShapeBoundsPass

新增文件：

```text
compiler/src/iree/compiler/GlobalOptimization/AssumeInputShapeBounds.cpp
```

pass 名称：

```text
iree-global-opt-assume-input-shape-bounds
```

### 3.1 作用

该 pass 在动态 rank-2 输入上插入 bounded shape assumption，例如：

```text
1 <= B <= maxBatch
1 <= S <= maxSeq
```

然后通过 `tensor.extract_slice` 等 IR 形式把 upper bound 信息传给后续优化和 codegen。

### 3.2 当前定位

bounded shape 对 codegen 分析有帮助，但不是单独的最终方案。当前结论是：

```text
普通 projection / MLP / lm_head:
  safe dynamic-M 已经足够有效，不一定依赖 bounded range。

attention context:
  直接用 bounded 信息强行进入 MMA 风险较高，需要更保守的 fallback 或进一步修复 dynamic rank-5 MMA lowering。
```

## 4. Safe dynamic-M matmul heuristic

修改文件：

```text
compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp
compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp
```

### 4.1 识别对象

只针对如下安全结构：

```text
tensor<?xK> x tensor<KxN> -> tensor<?xN>
```

约束：

```text
M 动态
K 静态
N 静态
不是 scaled contraction
dynamic K 与复杂 dynamic contraction 保留 fallback
```

### 4.2 核心策略

在 schedule selection 阶段使用代表性 M bound：

```text
kDynamicMHeuristicBound = 512
```

这样 matmul 仍然保持动态 ABI，但 codegen 可以选择接近静态 matmul 的 MMA config。动态 M 边界通过 padding / masking 处理。

### 4.3 安全保护

`KernelConfig.cpp` 中保留了对动态 K 的保护：

```text
如果 dynamic K 与 dynamic M/N 同时存在，则 fallback
```

这避免把 attention 这类复杂 dynamic batch_matmul 简单套入普通 dynamic-M 策略。

## 5. Attention context fallback

修改文件：

```text
compiler/src/iree/compiler/Codegen/Common/BlockDynamicDimensions.cpp
compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp
```

attention context 的形态更复杂：

```text
batch / M / K 可能都是动态
N = 128 通常固定
输入 f16，输出 f32
```

当前阶段发现：

```text
普通 dynamic-M heuristic 对 projection / MLP 有效。
attention context 如果直接强行进入 dynamic rank-5 MMA，完整模型中可能 timeout 或在后续 lowering 阶段失败。
```

因此当前策略是：

```text
识别 dynamic attention context
在不能安全进入 MMA 时，尝试 SIMT TileAndFuse fallback
保留完整 dynamic path 的正确性和可运行性
```

这也是阶段报告中 b1_s32 动态模型接近静态性能的关键改动之一。

## 6. Pipeline 接入

修改文件：

```text
compiler/src/iree/compiler/GlobalOptimization/Passes.cpp
compiler/src/iree/compiler/GlobalOptimization/Passes.td
compiler/src/iree/compiler/GlobalOptimization/BUILD.bazel
compiler/src/iree/compiler/GlobalOptimization/CMakeLists.txt
```

GlobalOptimization pipeline 中加入：

```text
createFlattenRank3MatmulPass
IREE::Flow::createCanonicalizePass
mlir::createCSEPass
```

目的是让 rank-3 matmul 在较早阶段显式变成标准 matmul，并尽快清理冗余 reshape / shape IR。

## 7. 当前技术边界

当前已经比较稳定的部分：

```text
rank3 -> rank2 flatten
safe dynamic-M matmul
projection / MLP / lm_head 的 MMA lowering
baseline vs flatten matmul 正确性验证
```

仍需继续研究的部分：

```text
dynamic rank-5 attention batch_matmul 的 MMA lowering
attention context 的 bufferization / LLVMGPU lowering
更大 B/S shape 下 dynamic attention context fallback 的稳定性
```

## 8. 关联实验

最小实验目录：

```text
experiments/dynamic_shape_matmul_experiment/
```

它用于证明：

- 动态 shape 不一定增加真实 Flow dispatch 数量。
- 动态 shape 会增加 HAL shape plumbing。
- codegen pipeline 从 `TileAndFuse` / MMA 退到 `VectorDistribute` 是关键性能差异。
- guarded specialization 可以恢复静态 matmul problem size 的大部分性能。
