# IREE 静态 Bucket 编译与 Matmul Flatten 优化总结

## 1. 背景与目标

当前工作的目标是验证并实现一种适合 IREE 部署大语言模型的优化路径：

1. 在 IREE 编译流程中自动识别模型里的 rank3 matmul / broadcasted batch matmul。
2. 将这类算子改写为标准的二维 `linalg.matmul`，使后端更容易生成高质量 GPU kernel。
3. 对真实动态输入使用 bucket 路由，将运行时的动态形状映射到一组静态编译的 VMFB。
4. 保持动态模型的使用体验，同时获得静态模型的编译优化收益。

最终希望达到的效果是：

```text
用户输入动态 batch/seq
  -> runtime router 选择合适静态 bucket
  -> pad 输入到 bucket shape
  -> 调用静态编译 VMFB
  -> 裁剪输出回真实请求语义
```

这样从用户视角看仍然支持动态输入；从 IREE 编译器视角看，每个 bucket 都是固定形状模型，可以进行更深度的静态优化。

## 2. 优化方法概述

LLM 中最主要的计算来自线性层，例如 Q/K/V projection、MLP gate/up/down projection。这些线性层在 IR 中常见形态是：

```text
lhs: [B, S, K]
rhs: [K, N]
out: [B, S, N]
```

其数学含义是：

```text
out[b, s, n] = sum_k lhs[b, s, k] * rhs[k, n]
```

该计算可以等价改写为：

```text
[B, S, K] -> [B*S, K]
[B*S, K] x [K, N] -> [B*S, N]
[B*S, N] -> [B, S, N]
```

也就是将 batch 维和 sequence 维合并，把 rank3 matmul 变成标准 rank2 matmul。

核心改写为：

```mlir
%flat_lhs = tensor.collapse_shape %lhs [[0, 1], [2]]
%flat_out = tensor.collapse_shape %out [[0, 1], [2]]
%matmul = linalg.matmul ins(%flat_lhs, %rhs) outs(%flat_out)
%expanded = tensor.expand_shape %matmul [[0, 1], [2]]
```

这个改写不改变数学语义，只改变 IR 形态。它的意义在于：IREE 后端对标准 `linalg.matmul` 的识别、tiling、vectorization、GPU lowering 通常比对 rank3 `linalg.generic` 或 broadcast 外壳下的 `batch_matmul` 更强。

## 3. Pass 实现说明

新增 pass 文件：

```text
compiler/src/iree/compiler/GlobalOptimization/FlattenRank3Matmul.cpp
```

该 pass 主要包含两类 pattern。

### 3.1 识别 rank3 by rank2 matmul

第一类 pattern 识别：

```text
[B, S, K] x [K, N] -> [B, S, N]
```

识别条件包括：

```text
lhs rank == 3
rhs rank == 2
output rank == 3
lhs.dim(2) == rhs.dim(0)
lhs.dim(0/1) == output.dim(0/1)
rhs.dim(1) == output.dim(2)
```

同时要求 affine map 匹配 matmul 语义：

```text
lhs:    (b, s, k)
rhs:    (k, n)
output: (b, s, n)
```

对应源码逻辑：

```cpp
SmallVector<AffineMap> expectedMaps = {
    AffineMap::get(4, 0, {bDim, sDim, kDim}, context),
    AffineMap::get(4, 0, {kDim, nDim}, context),
    AffineMap::get(4, 0, {bDim, sDim, nDim}, context),
};
```

并检查 iterator 类型：

```text
parallel, parallel, parallel, reduction
```

最后检查 body 是乘加归约：

```text
acc + lhs * rhs
```

满足条件后执行：

```cpp
bsSize = bSize * sSize;
flatLhs = collapse_shape(lhs, [[0, 1], [2]]);
flatOutput = collapse_shape(output, [[0, 1], [2]]);
matmul = linalg.matmul(flatLhs, rhs, flatOutput);
expanded = expand_shape(matmul, [[0, 1], [2]]);
```

### 3.2 识别 broadcasted batch matmul

第二类 pattern 识别如下形态：

```text
rhs: [K, N]
rhs3 = broadcast(rhs) -> [B, K, N]
linalg.batch_matmul(lhs: [B, S, K], rhs3: [B, K, N])
```

这种形态在导入/转换模型时较常见。实际上 rhs 没有真正的 batch 语义，只是同一权重矩阵被 broadcast 到 batch 维。

pass 会先确认 rhs3 的定义来自一个 broadcast-like `linalg.generic`，然后取回原始 rank2 rhs：

```text
broadcasted rhs: [B, K, N]
source rhs:      [K, N]
```

然后同样将：

```text
[B, S, K] x [K, N]
```

改写成：

```text
[B*S, K] x [K, N]
```

这可以消除不必要的权重 broadcast 外壳，使权重矩阵直接参与标准 matmul。

## 4. 最小 IR 证明

测试文件：

```text
compiler/src/iree/compiler/GlobalOptimization/test/flatten_rank3_matmul.mlir
```

### 4.1 rank3 generic matmul 示例

原始 IR：

```mlir
func.func @rank3_by_rank2_matmul(%lhs: tensor<2x3x4xf32>,
                                 %rhs: tensor<4x5xf32>,
                                 %out: tensor<2x3x5xf32>) -> tensor<2x3x5xf32> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%lhs, %rhs : tensor<2x3x4xf32>, tensor<4x5xf32>)
      outs(%out : tensor<2x3x5xf32>) {
    ^bb0(%l: f32, %r: f32, %acc: f32):
      %mul = arith.mulf %l, %r : f32
      %add = arith.addf %mul, %acc : f32
      linalg.yield %add : f32
    } -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
}
```

pass 后期望 IR：

```mlir
%flat_lhs = tensor.collapse_shape %lhs [[0, 1], [2]]
  : tensor<2x3x4xf32> into tensor<6x4xf32>

%flat_out = tensor.collapse_shape %out [[0, 1], [2]]
  : tensor<2x3x5xf32> into tensor<6x5xf32>

%matmul = linalg.matmul
  ins(%flat_lhs, %rhs : tensor<6x4xf32>, tensor<4x5xf32>)
  outs(%flat_out : tensor<6x5xf32>)

%expanded = tensor.expand_shape %matmul [[0, 1], [2]]
  output_shape [2, 3, 5]
  : tensor<6x5xf32> into tensor<2x3x5xf32>
```

这个测试证明了 pass 的基本正确性：原本的 rank3 `linalg.generic` 被改写为标准 `linalg.matmul`。

### 4.2 broadcasted batch matmul 示例

测试中还有如下形态：

```mlir
%rhs3 = linalg.generic
  ins(%rhs : tensor<4096x1024xf16>)
  outs(%rhs3_empty : tensor<4x4096x1024xf16>)

%0 = linalg.batch_matmul
  ins(%lhs, %rhs3 : tensor<4x32x4096xf16>, tensor<4x4096x1024xf16>)
  outs(%out : tensor<4x32x1024xf32>)
```

pass 后要求变为：

```mlir
%flat_lhs = tensor.collapse_shape %lhs [[0, 1], [2]]
  : tensor<4x32x4096xf16> into tensor<128x4096xf16>

%flat_out = tensor.collapse_shape %out [[0, 1], [2]]
  : tensor<4x32x1024xf32> into tensor<128x1024xf32>

%matmul = linalg.matmul
  ins(%flat_lhs, %rhs : tensor<128x4096xf16>, tensor<4096x1024xf16>)
  outs(%flat_out : tensor<128x1024xf32>)

%expanded = tensor.expand_shape %matmul [[0, 1], [2]]
  output_shape [4, 32, 1024]
```

这说明 pass 不仅能处理直接 rank3 matmul，也能处理权重被 broadcast 后形成的 batch matmul。

## 5. 真实 DeepSeek 模型 IR 证明

以 `b4_s32` bucket 为例，global optimization 后真实模型 IR 中出现：

```mlir
%collapsed_8 = tensor.collapse_shape %44 [[0, 1], [2]]
  : tensor<4x32x4096xf16> into tensor<128x4096xf16>

%47 = linalg.matmul
  ins(%collapsed_8, %onnx__MatMul_8048
  : tensor<128x4096xf16>, tensor<4096x4096xf16>)
  outs(%46 : tensor<128x4096xf32>)
  -> tensor<128x4096xf32>

%expanded_9 = tensor.expand_shape %47 [[0, 1], [2, 3]]
  output_shape [4, 32, 32, 128]
  : tensor<128x4096xf32> into tensor<4x32x32x128xf32>
```

这对应注意力中的 projection：

```text
[4, 32, 4096] -> [128, 4096]
[128, 4096] x [4096, 4096] -> [128, 4096]
```

MLP 中也能看到类似结构：

```text
gate/up projection:
  [4, 32, 4096] -> [128, 4096]
  [128, 4096] x [4096, 14336] -> [128, 14336]
  [128, 14336] -> [4, 32, 14336]

down projection:
  [4, 32, 14336] -> [128, 14336]
  [128, 14336] x [14336, 4096] -> [128, 4096]
  [128, 4096] -> [4, 32, 4096]
```

这些都是 LLM 中最重的线性层。优化命中了主计算路径，因此有实际性能意义。

## 6. 自动 pass 与已有 flatten 版本的一致性

使用加入 pass 后的 IREE 直接编译原始模型，不再依赖外部 rewrite 脚本，得到的 global optimization IR 与已有 flatten 版本算子数量一致：

```text
auto flatten:
  linalg.matmul       = 225
  linalg.batch_matmul = 64

existing flatten:
  linalg.matmul       = 225
  linalg.batch_matmul = 64
```

这说明：

```text
原始模型输入 IREE
  -> GlobalOptimization pass 自动识别目标算子
  -> 自动改写为 flatten matmul 形态
  -> 得到和外部 flatten 脚本一致的 IR 结构
```

因此，该优化已经真正接入 IREE 编译流程，而不是依赖模型外部预处理。

## 7. 为什么静态 bucket 编译会快

静态 bucket 编译时，模型中的关键形状都是编译期常量。例如：

```mlir
tensor<4x32x4096xf16>
tensor<128x4096xf16>
tensor<4096x14336xf16>
```

这意味着：

```text
B = 4
S = 32
B*S = 128
K = 4096
N = 14336
```

IREE 后端可以基于这些固定值做更深入的优化：

```text
固定 matmul M/N/K
固定 workgroup 数量
固定 dispatch workload
固定 buffer size
固定 layout 选择
更明确的 tiling/vectorization/tensor core lowering
更少 tensor.dim / arith.muli 等运行时 shape 计算
更少动态边界检查
更容易做 canonicalization 和 fusion
```

直观理解：

```text
静态模型：
  编译器知道题目的全部尺寸，可以提前生成专用解法。

动态模型：
  编译器只知道有一类题，不知道具体尺寸，必须生成更泛化、更保守的解法。
```

## 8. 为什么动态模型效果不好

动态模型中，IR 形状通常是：

```mlir
tensor<?x?x4096xf16>
```

flatten 后也只能得到：

```mlir
tensor<?x4096xf16>
```

也就是 `B*S` 在编译期未知。动态 IR 通常需要运行时 shape 计算：

```mlir
%B = tensor.dim %arg0, %c0
%S = tensor.dim %arg0, %c1
%BS = arith.muli %B, %S
```

这会导致：

```text
matmul 的 M 维是动态值
dispatch workload 是动态值
buffer shape 是动态值
layout 和 tiling 选择更保守
部分优化不能按具体 shape 展开
小 shape 和大 shape 需要共享一套泛化代码
```

所以动态模型并不是完全没有优化，而是后端无法针对真实热点 shape 做专门优化。

bucket 方法的价值在于：

```text
运行时仍支持动态输入
编译时看到的是固定 shape
```

它把动态问题拆成：

```text
运行时轻量路由
+ 多个静态专用模型
```

## 9. 性能数据

下面数据来自 dynamic vs bucket same-input benchmark。动态路径接收真实 shape，bucket 路径接收相同有效 token 区域并 pad 到静态 bucket。

```text
request   -> bucket   dynamic       bucket       speedup    padding
b1_s16    -> b4_s32    1709.82 ms     37.13 ms     46.05x     87.5%
b1_s32    -> b4_s32    3445.58 ms     37.16 ms     92.72x     75.0%
b1_s48    -> b4_s64    5134.83 ms     60.39 ms     85.02x     81.2%
b1_s64    -> b4_s64    6871.54 ms     60.42 ms    113.72x     75.0%
b1_s96    -> b4_s128  10329.42 ms    110.92 ms     93.13x     81.2%
b1_s128   -> b4_s128  13744.96 ms    110.84 ms    124.01x     75.0%
b2_s48    -> b4_s64   10328.74 ms     60.42 ms    170.95x     62.5%
b3_s80    -> b4_s128  25677.67 ms    110.80 ms    231.74x     53.1%
b4_s64    -> b4_s64   27496.20 ms     60.38 ms    455.35x      0.0%
b5_s96    -> b8_s128  51664.35 ms    140.40 ms    367.97x     53.1%
b8_s128   -> b8_s128 110040.35 ms    139.95 ms    786.26x      0.0%
```

汇总：

```text
平均加速：233.36x
最小加速：46.05x
最大加速：786.26x
```

这里有两个特别关键的观察。

### 9.1 即使 padding 很高，静态 bucket 仍然更快

例如：

```text
b1_s16 -> b4_s32
padding = 87.5%
dynamic = 1709.82 ms
bucket  = 37.13 ms
speedup = 46.05x
```

bucket 路径实际计算了更多 token slot，但仍然远快于动态路径。这说明主要收益不是“少算了”，而是静态编译模型的 kernel/dispatch 质量远高于动态路径。

### 9.2 无 padding 时，静态模型优势更干净

例如：

```text
b4_s64 -> b4_s64
padding = 0.0%
dynamic = 27496.20 ms
bucket  = 60.38 ms
speedup = 455.35x
```

这个 case 没有 padding 干扰，仍然有极大差距，说明静态 shape specialization 本身就是关键性能来源。

## 10. 正确性验证

dynamic vs bucket full grid 正确性结果：

```text
total: 11
passed: 11
failed: 0
```

容差：

```text
atol = 0.06
rtol = 0.01
```

逐项结果：

```text
b1_s16   -> b4_s32   passed max_abs=0.03515625 mean_abs=0.00527219
b1_s32   -> b4_s32   passed max_abs=0.04199219 mean_abs=0.00640448
b1_s48   -> b4_s64   passed max_abs=0.03125000 mean_abs=0.00526311
b1_s64   -> b4_s64   passed max_abs=0.02343750 mean_abs=0.00380199
b1_s96   -> b4_s128  passed max_abs=0.02636719 mean_abs=0.00470436
b1_s128  -> b4_s128  passed max_abs=0.02343750 mean_abs=0.00427244
b2_s48   -> b4_s64   passed max_abs=0.03320312 mean_abs=0.00521544
b3_s80   -> b4_s128  passed max_abs=0.03125000 mean_abs=0.00476594
b4_s64   -> b4_s64   passed max_abs=0.03515625 mean_abs=0.00455593
b5_s96   -> b8_s128  passed max_abs=0.04296875 mean_abs=0.00468089
b8_s128  -> b8_s128  passed max_abs=0.03125000 mean_abs=0.00396731
```

最大误差：

```text
b5_s96 -> b8_s128
max_abs_error = 0.04296875
```

仍低于 `atol=0.06`。

这证明 bucket 路径不是近似替代，而是在 padding、mask 和 output crop 后保持了动态路径语义。

## 11. 之前 correctness 问题的定位与修复

早期 bucket correctness 失败并不是 bucket 方法错误，而是 static ONNX 导出中的 RoPE `position_ids` 路径触发了 IREE CUDA correctness/layout 问题。

问题路径：

```text
Constant [1, S]
  -> Expand [B, S]
  -> Cast
```

该 no-input `position_ids` 路径在 fixed static ONNX 下容易引入布局/正确性问题。

修复方式：

```text
static bucket 导出时不要把 RoPE position_ids expand 到 batch
保持 [1, S] / [1, S, D]
让后续 q/k elementwise 自然 broadcast 到 batch
```

修复后，full grid 11/11 通过。

这说明：

```text
失败点是具体 IR/layout 问题
不是 bucket 部署方案本身错误
```

## 12. 部署方案

当前已实现的部署结构包括三部分：

```text
前端页面
  -> Python chat server / tokenizer / generation loop
  -> C++ IREE bucket serving service
  -> static VMFB bucket
```

### 12.1 Router 逻辑

对于请求 shape：

```text
request = [B, S]
```

选择最小可容纳 bucket：

```text
bucket.batch >= B
bucket.seq >= S
```

例如：

```text
b1_s56 -> b1_s64
b1_s88 -> b1_s128
b1_s129 -> b1_s256
b3_s80 -> b4_s128
b5_s96 -> b8_s128
```

然后：

```text
input_ids       pad 到 [bucket_B, bucket_S]
attention_mask  pad 区域填 0
last_token_idx  指向真实最后一个有效 token
```

模型输出后，只取真实 batch 对应行。

### 12.2 C++ serving

当前实现了 C++ IREE serving service：

```text
tools/iree-bucket-serving-main.cc
```

它负责：

```text
读取 manifest
选择 bucket
加载对应 VMFB 和参数
调用 IREE runtime
返回输出和 invoke latency
```

相比直接用 `iree-run-module` CLI，每 token 启动进程的开销大幅降低，更接近真实部署形态。

### 12.3 多 GPU 部署

服务器有 4 张 4090，可以启动多个 serving worker：

```text
GPU 0 -> port 8010
GPU 1 -> port 8011
GPU 2 -> port 8012
GPU 3 -> port 8013
```

Python server 通过多个 `--cpp-serving-url` 将请求分发到多个 worker。不同 bucket 可以稳定落在不同 worker 上，避免单进程同时加载过多 8B 参数导致 OOM。

## 13. 在线指标解释

前端中的指标可以这样理解。

### 13.1 Bucket Hits

例如：

```text
Bucket Hits: b1_s64:9, b1_s128:64, b1_s256:44
```

含义是生成过程中：

```text
9 次调用使用 b1_s64
64 次调用使用 b1_s128
44 次调用使用 b1_s256
```

因为生成时 sequence length 逐 token 增长：

```text
seq <= 64       -> b1_s64
65 <= seq <=128 -> b1_s128
129 <= seq <=256 -> b1_s256
```

这个指标证明 runtime 确实在根据动态输入选择静态 bucket。

### 13.2 Mean Invoke / Last Invoke

这两个指标表示 C++ serving 调用 IREE VMFB 的核心执行时间：

```text
Mean Invoke: 平均每 token IREE 执行耗时
Last Invoke: 最后一次 token 的 IREE 执行耗时
```

这是衡量静态 bucket VMFB 核心性能的主要指标。

### 13.3 IREE Tokens/s

该指标只看 IREE invoke 时间：

```text
IREE Tokens/s = generated_tokens / total_iree_invoke_time
```

它更能反映模型 VMFB 本身的执行效率。

### 13.4 Host Overhead

Host overhead 包括：

```text
Python generation loop
tokenizer encode/decode
HTTP request
.npy 文件 IO
数据拷贝
可能的冷启动加载
```

因此它不是编译优化本身的核心性能，而是当前 demo 级 serving 架构的额外开销。后续生产化需要继续降低这部分。

### 13.5 Avg Padding

该指标表示 bucket 计算中的 padding 比例：

```text
padding = 1 - active_token_slots / bucket_token_slots
```

它用于评估 bucket 设计是否贴近真实请求分布。padding 越低，说明 bucket 网格越合理。

## 14. 当前方案的优势与限制

### 14.1 优势

当前方案已经证明：

```text
IREE pass 可以自动识别并改写目标算子
静态 bucket 编译能显著提升 IREE CUDA 路径性能
dynamic 与 bucket 输出在 full grid 上一致
runtime router 可以支持真实动态输入
C++ serving 形态比 CLI 更接近实际部署
```

核心优势是：

```text
用少量 padding 换取静态编译优化
用有限 bucket 数量覆盖大量真实请求
避免为每个真实 shape 单独编译模型
```

### 14.2 当前限制

当前在线 demo 仍有一些非最终生产化因素：

```text
当前生成路径尚未实现 KV cache
每个 token 仍会重新跑当前完整 seq
Python + HTTP + npy 文件协议有明显 host overhead
部分 benchmark 来自 DEBUG build，绝对时间会受影响
```

因此当前在线前端的 wall time 不能直接代表最终生产吞吐。更合理的判断方式是分别看：

```text
IREE Invoke: 静态 VMFB 核心执行能力
Host Overhead: serving 工程开销
Bucket Hits / Padding: bucket 策略是否有效
```

后续如果实现 KV cache decode、二进制 tensor 协议、长驻 scheduler 和 release build，整体 serving 性能还会进一步接近生产形态。

## 15. 结论

本工作形成了完整证据链：

```text
1. IR 形态证明
   rank3/broadcast matmul 被改写为 collapse_shape + linalg.matmul + expand_shape

2. 真实模型证明
   DeepSeek 的 Q/K/V/MLP projection 已经变成固定 [B*S, K] x [K, N] GEMM

3. IREE 集成证明
   原始模型直接经过加入 pass 的 IREE 编译后，matmul 数量与已有 flatten 版本一致

4. 性能证明
   11 个 shape 上平均 233.36x 加速，最小 46.05x，最大 786.26x

5. 正确性证明
   dynamic vs bucket full grid 11/11 passed，最大误差 0.04296875 < atol 0.06

6. 部署证明
   在线 router 能根据真实动态 seq 选择 b1_s64 / b1_s128 / b1_s256 等静态 VMFB
```

最终结论：

```text
该优化方法有效，并且具有实际部署意义。

它的本质不是简单 padding，也不是单纯多编译几个模型，而是将用户侧动态 shape
转换为编译器侧静态 shape，让 IREE 后端能对主要 matmul 计算进行深度 specialization。

动态模型效果不好，是因为后端必须保留 shape 泛化能力；静态 bucket 快，是因为
关键 M/N/K、dispatch workload、buffer shape 都在编译期固定。

bucket 部署则在两者之间取得平衡：运行时保留动态输入能力，编译时获得静态模型性能。
```

## 16. 动态 Shape 与静态 Bucket 的中间 IR 对比实验

为了更直观看清动态 shape 和静态 bucket 在 IREE 中间 IR 中的差异，额外对比了同一模型在两个编译阶段的 IR：

```text
global-optimization
flow
```

使用的输入模型：

```text
动态模型：
dynamic_shape_b_s/build_flatten_matmul_input_i32_with_demote/deepseek_r1_8b_external_inlined.mlir

静态 bucket：
flatten_shape_b4_s128/build_flatten_matmul_last_nonpad_index_input_i32_with_demote/deepseek_r1_8b_external_inlined.mlir
```

### 16.1 编译命令

生成 dynamic global IR：

```bash
/home/zhongjialin/projects/iree-build/tools/iree-compile \
  --iree-input-type=onnx \
  --iree-input-demote-i64-to-i32 \
  --iree-opt-strip-assertions \
  --compile-to=global-optimization \
  /home/zhongjialin/projects/iree/deepseek-R1-Llama-8b/dynamic_shape_b_s/build_flatten_matmul_input_i32_with_demote/deepseek_r1_8b_external_inlined.mlir \
  -o /tmp/deepseek_dynamic_bs_global.mlir
```

生成 static global IR：

```bash
/home/zhongjialin/projects/iree-build/tools/iree-compile \
  --iree-input-type=onnx \
  --iree-input-demote-i64-to-i32 \
  --iree-opt-strip-assertions \
  --compile-to=global-optimization \
  /home/zhongjialin/projects/iree/deepseek-R1-Llama-8b/flatten_shape_b4_s128/build_flatten_matmul_last_nonpad_index_input_i32_with_demote/deepseek_r1_8b_external_inlined.mlir \
  -o /tmp/deepseek_b4_s128_global.mlir
```

生成 dynamic flow IR：

```bash
/home/zhongjialin/projects/iree-build/tools/iree-compile \
  --iree-input-type=onnx \
  --iree-input-demote-i64-to-i32 \
  --iree-opt-strip-assertions \
  --compile-to=flow \
  /home/zhongjialin/projects/iree/deepseek-R1-Llama-8b/dynamic_shape_b_s/build_flatten_matmul_input_i32_with_demote/deepseek_r1_8b_external_inlined.mlir \
  -o /tmp/deepseek_dynamic_bs_flow.mlir
```

生成 static flow IR：

```bash
/home/zhongjialin/projects/iree-build/tools/iree-compile \
  --iree-input-type=onnx \
  --iree-input-demote-i64-to-i32 \
  --iree-opt-strip-assertions \
  --compile-to=flow \
  /home/zhongjialin/projects/iree/deepseek-R1-Llama-8b/flatten_shape_b4_s128/build_flatten_matmul_last_nonpad_index_input_i32_with_demote/deepseek_r1_8b_external_inlined.mlir \
  -o /tmp/deepseek_b4_s128_flow.mlir
```

### 16.2 导出阶段的差异

动态模型在导出 ONNX 时显式声明了动态轴：

```python
dynamic_axes={
    "input_ids": {0: "batch", 1: "seq"},
    "attention_mask": {0: "batch", 1: "seq"},
    "logits": output_axes,
}
```

因此导入后的入口类型是：

```mlir
!torch.vtensor<[?,?],si32>
```

静态 bucket 导出时使用固定 dummy input：

```python
input_ids = torch.zeros((args.batch, args.seq), dtype=torch.long, device=device)
attention_mask = torch.ones((args.batch, args.seq), dtype=torch.long, device=device)
```

并且：

```python
dynamic_axes={}
```

因此 b4_s128 的入口类型是：

```mlir
!torch.vtensor<[4,128],si32>
```

这说明动态和静态的根本差异在导出阶段就已经确定：动态模型把 batch/seq 作为符号维度，静态 bucket 直接把 batch/seq 固定进模型。

### 16.3 Global Optimization 阶段统计

对比结果：

```text
global IR:

dynamic ? count:       16721
static b4_s128 ? count:    0

dynamic arith.muli:       10
static b4_s128 arith.muli: 0

dynamic linalg.matmul:       225
static b4_s128 linalg.matmul: 225

dynamic linalg.batch_matmul:       65
static b4_s128 linalg.batch_matmul: 64

dynamic tensor.reshape: 293
static tensor.reshape:    0

dynamic tensor.empty(%...): 56
static tensor.empty(%...):  0
```

这里最重要的结论是：

```text
flatten matmul pass 在动态和静态上都能生效。
两者都有 225 个 linalg.matmul。

真正区别不是“有没有 matmul”，而是 matmul 的 shape 是否静态。
```

动态版本虽然已经被改写成 `linalg.matmul`，但是类型仍然是：

```mlir
tensor<?x4096xf16>
tensor<?x4096xf32>
```

静态 bucket 版本则是：

```mlir
tensor<512x4096xf16>
tensor<512x4096xf32>
```

也就是说：

```text
dynamic:
  M = B*S，在编译期未知

static b4_s128:
  M = 4*128 = 512，在编译期固定
```

### 16.4 入口 import 的 IR 差异

动态 global IR 中，IREE 需要从 `hal.buffer_view` 读取 runtime shape：

```mlir
%0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
%1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
%2 = hal.tensor.import wait(%arg2) => %arg0
  : !hal.buffer_view -> tensor<?x?xi32>{%0, %1}

%4 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
%5 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
%6 = hal.tensor.import wait(%arg2) => %arg1
  : !hal.buffer_view -> tensor<?x?xi32>{%4, %5}
```

静态 b4_s128 global IR 中，输入 shape 已经固定：

```mlir
%0 = hal.tensor.import wait(%arg3) => %arg0
  : !hal.buffer_view -> tensor<4x128xi32>

%2 = hal.tensor.import wait(%arg3) => %arg1
  : !hal.buffer_view -> tensor<4x128xi32>

%4 = hal.tensor.import wait(%arg3) => %arg2
  : !hal.buffer_view -> tensor<4xi32>
```

这说明：

```text
动态模型：
  需要运行时读取输入 buffer_view 的 shape
  tensor 类型携带 {%0, %1} 这类动态 shape operand

静态 bucket：
  import 时直接恢复成 tensor<4x128xi32>
  没有 runtime shape operand
```

### 16.5 Embedding 与 reshape 的差异

动态 IR 中，embedding 后需要动态计算 `B*S`：

```mlir
%22 = arith.index_cast %0 : index to i32
%23 = arith.index_cast %1 : index to i32
%collapsed = tensor.collapse_shape %21 [[0, 1]]
  : tensor<?x?xi32> into tensor<?xi32>

%24 = arith.muli %0, %1 overflow<nsw> : index
%25 = tensor.empty(%24) : tensor<?x4096xf16>

%from_elements = tensor.from_elements %22, %23, %c4096_i32 : tensor<3xi32>
%reshape = tensor.reshape %26(%from_elements)
  : (tensor<?x4096xf16>, tensor<3xi32>) -> tensor<?x?x4096xf16>
```

静态 b4_s128 中，对应部分已经完全固定：

```mlir
%31 = linalg.generic ... -> tensor<4x128x4096xf16>

%collapsed_7 = tensor.collapse_shape %31 [[0, 1], [2]]
  : tensor<4x128x4096xf16> into tensor<512x4096xf16>

%32 = tensor.empty() : tensor<512x4096xf32>
%33 = linalg.fill ... outs(%32 : tensor<512x4096xf32>)
```

结论：

```text
dynamic:
  B*S 需要 arith.muli 在运行时计算
  empty/reshape 需要动态 shape operand

static:
  4*128 已经折叠成 512
  empty/reshape 都是固定类型
```

### 16.6 Matmul 的 IR 差异

动态 global IR 中典型 projection：

```mlir
%collapsed_36 = tensor.collapse_shape %139 [[0, 1], [2]]
  : tensor<?x?x4096xf16> into tensor<?x4096xf16>

%142 = linalg.matmul
  ins(%collapsed_36, %onnx__MatMul_9019
  : tensor<?x4096xf16>, tensor<4096x4096xf16>)
  outs(%141 : tensor<?x4096xf32>)
  -> tensor<?x4096xf32>
```

静态 b4_s128 global IR 中对应 projection：

```mlir
%collapsed_7 = tensor.collapse_shape %31 [[0, 1], [2]]
  : tensor<4x128x4096xf16> into tensor<512x4096xf16>

%34 = linalg.matmul
  ins(%collapsed_7, %onnx__MatMul_7815
  : tensor<512x4096xf16>, tensor<4096x4096xf16>)
  outs(%33 : tensor<512x4096xf32>)
  -> tensor<512x4096xf32>
```

这就是最直接的 IR 证据：

```text
dynamic matmul:
  matmul_Dx4096x4096
  M 是动态 D

static matmul:
  matmul_512x4096x4096
  M 固定为 512
```

### 16.7 Flow 阶段统计

flow 阶段对比结果：

```text
flow IR:

dynamic ? count:       5990
static b4_s128 ? count:   0

dynamic flow.dispatch: 780
static flow.dispatch:  550

dynamic flow.executable: 94
static flow.executable:  40
```

这说明动态 shape 到 flow 阶段仍然没有完全消除，且会生成更多泛化 dispatch/executable。

### 16.8 Flow 阶段 dispatch 名称差异

动态 flow IR 中的 matmul dispatch：

```mlir
flow.dispatch
  @main_graph$async_dispatch_29_matmul_Dx4096x4096_f16xf16xf32[%9]
  (%89, %onnx__MatMul_9019, %9)
  : (tensor<?x4096xf16>{%9}, tensor<4096x4096xf16>, index)
  -> tensor<?x4096xf16>{%9}
```

动态 dispatch export：

```mlir
flow.executable.export public
  @main_graph$async_dispatch_29_matmul_Dx4096x4096_f16xf16xf32
  workgroups(%arg0: index) -> (index, index, index)
```

注意这里的：

```text
Dx4096x4096
workgroups(%arg0: index)
```

说明 matmul 的 M 维和 workgroup 计算仍依赖 runtime 参数。

静态 b4_s128 flow IR 中的 matmul dispatch：

```mlir
flow.dispatch
  @main_graph$async_dispatch_3_matmul_512x4096x4096_f16xf16xf32
  (%6, %onnx__MatMul_7815)
  : (tensor<512x4096xf16>, tensor<4096x4096xf16>)
  -> tensor<512x4096xf16>
```

静态 dispatch export：

```mlir
flow.executable.export public
  @main_graph$async_dispatch_3_matmul_512x4096x4096_f16xf16xf32
  workgroups() -> (index, index, index)
```

注意这里的：

```text
512x4096x4096
workgroups()
```

说明 dispatch 的核心 shape 和 workload 都已经静态化，不需要 runtime shape 参数。

### 16.9 这一组测试的结论

这组中间 IR 测试证明：

```text
1. 动态和静态都能触发 flatten matmul pass。
   两者在 global 阶段都有 225 个 linalg.matmul。

2. 动态模型的问题不是 pass 不生效，而是 pass 后 matmul 仍然是动态 M：
   tensor<?x4096xf16>
   matmul_Dx4096x4096
   workgroups(%arg0)

3. 静态 bucket 的优势是 M/B/S 被写死：
   tensor<512x4096xf16>
   matmul_512x4096x4096
   workgroups()

4. 动态模型到 flow 阶段仍保留大量动态 shape：
   ? count = 5990
   flow.executable = 94

5. 静态 bucket 到 flow 阶段没有动态 shape：
   ? count = 0
   flow.executable = 40

6. 因此静态 bucket 更快的底层原因是：
   后端看到的是固定 shape dispatch，
   可以生成固定 workload、固定 tile、固定 buffer layout 的专用 kernel。
```

简言之：

```text
动态路径：
  flatten pass 生效，但结果是 matmul_Dx4096x4096。

静态 bucket：
  flatten pass 生效，结果是 matmul_512x4096x4096。

两者差异就在这个 D 和 512。
D 表示运行时未知，512 表示编译期已知。
```
