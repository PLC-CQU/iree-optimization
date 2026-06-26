# IREE 动态 Shape 优化阶段总结

## 1. 文档目的

本文总结当前阶段针对 IREE 动态 shape 模型的优化工作。重点不是 bucket 部署，而是直接优化动态 shape 编译路径，希望让动态模型接近静态模型的性能。

本阶段围绕三个问题展开：

1. 为什么原始动态模型比静态模型慢很多。
2. 哪些动态结构已经可以稳定优化。
3. 哪些结构仍然是动态模型和静态模型之间的主要差距。

最终阶段性结论是：

```text
safe_dynamic_m 是第一阶段稳定有效的动态 shape 优化版本。

projection / MLP / lm_head 这类 flattened dynamic-M matmul 已经可以稳定走 CUDA MMA。

attention context 不能简单依靠 upper-bound 直接切到 MMA；完整模型中会 timeout/hang。

但针对 attention context 增加 SIMT TileAndFuse fallback 后，b1_s32 动态模型已经接近静态性能：
40.298 ms vs 39.749 ms，仅慢 1.38%。
```

## 2. 背景

静态模型中，IREE 可以在编译期看到完整 shape，例如：

```text
B = 1
S = 32
K = 4096
N = 14336
```

因此 GPU codegen 可以直接选择 tile、workgroup、MMA intrinsic、padding 策略。

动态模型中，输入是：

```text
input_ids:      tensor<?x?xi32>
attention_mask: tensor<?x?xi32>
```

后续很多关键计算的 loop bound 都变成 SSA value，例如 `B`、`S`、`B*S`。如果 codegen 无法证明这些动态维度适合 MMA，就会退到更保守的 lowering。

原始动态模型的典型问题是：

```text
普通 matmul 不能稳定进入 MMA pipeline
attention batch_matmul 更容易退到 VectorDistribute
dispatch workload 和 buffer shape 都带大量动态参数
```

因此动态模型和静态模型之间出现数量级差距。

## 3. 优化目标

本阶段的目标不是 bucket routing，而是：

```text
单个动态 VMFB
  接受动态 B/S 输入
  在编译器内部识别可优化结构
  尽量生成接近静态模型的 CUDA lowering
```

当前希望达成的理想路径是：

```text
projection / MLP / lm_head:
  tensor<?xK> x tensor<KxN>
  -> dynamic-M matmul
  -> TileAndFuse + NV_MMA_SYNC

attention scores/context:
  rank-5 dynamic batch_matmul
  -> 未来也应进入安全的 MMA lowering
```

## 4. 已实现的优化一：safe dynamic-M matmul

### 4.1 识别对象

LLM 中大量线性层可以被 flatten 成：

```text
lhs: tensor<?xKxf16>
rhs: tensor<KxNxf16>
out: tensor<?xNxf32>
```

这里动态维度只出现在 M 维，也就是 token 数：

```text
M = B * S
```

K 和 N 是静态权重维度，例如：

```text
K = 4096
N = 4096 / 1024 / 14336 / 128256
```

这种结构比 attention batch_matmul 简单很多，因为 reduction K 是静态的，N 也是静态的，只有 M 动态。

### 4.2 Codegen 侧策略

在 `ConfigUtils.cpp` 中加入 dynamic-M heuristic：

```cpp
constexpr int64_t kDynamicMHeuristicBound = 512;
```

当满足：

```text
只有 M 是 dynamic
N 是 static
K 是 static
不是 scaled contraction
```

则使用代表性 M bound 选择 MMA config，同时强制 padding：

```text
padding = [32, 16, 128]
mma_kind = NV_MMA_SYNC_F32_16x8x16_F16
pipeline = TileAndFuse
```

这样动态 M 不再阻止普通 matmul 进入 MMA。

### 4.3 安全保护

在 `KernelConfig.cpp` 中保留安全限制：

```text
如果 dynamic K 和 dynamic M/N 同时存在，则 fallback
```

也就是说：

```text
tensor<?x4096> x tensor<4096xN>
  可以优化

tensor<?x?x?> dynamic batch_matmul
  暂不直接强行优化
```

这个保护避免了早期强行优化 attention dynamic batch_matmul 后出现 runtime hang。

## 5. safe_dynamic_m 性能与正确性

### 5.1 性能结果

此前完整模型测试显示，safe dynamic-M 对真实 DeepSeek 动态模型有明显收益：

| Shape | old_dynamic | safe_dynamic_m | static_exact | old -> safe | safe / static |
|---|---:|---:|---:|---:|---:|
| b1_s32 | 3364.666 ms | 82.213 ms | 37.309 ms | 40.926x | 2.204x |
| b1_s64 | 6707.30 ms | 140.05 ms | 41.77 ms | 47.89x | 3.35x |
| b4_s64 | 26959.59 ms | 510.86 ms | 121.08 ms | 52.77x | 4.22x |
| b8_s128 | 107987.25 ms | 2012.78 ms | 119.70 ms | 53.65x | 16.82x |

最近一次 b1_s32 复测结果：

```text
safe_dynamic_m: 88.826 ms
static_exact:   39.768 ms
safe/static:    2.234x
```

这说明 safe dynamic-M 已经把原始动态模型的主要灾难性开销消掉，但和静态模型仍有约 2.2x 以上差距。

### 5.2 正确性结果

已经对多个 shape 做过输出对比：

```text
b1_s64:
  old_dynamic vs safe_dynamic_m max_abs = 0.00781250
  safe_dynamic_m vs static_exact max_abs = 0.00781250
  allclose(atol=0.06, rtol=0.01) = True

b4_s64:
  old_dynamic vs safe_dynamic_m max_abs = 0.00781250
  safe_dynamic_m vs static_exact max_abs = 0.00781250
  allclose(atol=0.06, rtol=0.01) = True

b8_s128:
  old_dynamic vs safe_dynamic_m max_abs = 0.00781250
  safe_dynamic_m vs static_exact max_abs = 0.00781250
  allclose(atol=0.06, rtol=0.01) = True
```

因此 safe dynamic-M 当前既有性能收益，也保持了模型输出正确性。

## 6. 已尝试的优化二：bounded dynamic shape

### 6.1 思路

为了让动态模型获得更多静态信息，新增了一个 global optimization pass：

```text
compiler/src/iree/compiler/GlobalOptimization/AssumeInputShapeBounds.cpp
```

pass 名称：

```text
iree-global-opt-assume-input-shape-bounds
```

它在动态输入后插入：

```mlir
%b = util.assume.int %dim_b<umin = 1, umax = 8> : index
%s = util.assume.int %dim_s<umin = 1, umax = 128> : index
%bounded = tensor.extract_slice %input[0, 0] [%b, %s] [1, 1]
```

目标是：

```text
保持 ABI 仍然是动态输入
但告诉编译器 B <= 8, S <= 128
让 codegen 可以对动态 loop bound 推导上界
```

### 6.2 对真实 IREE global IR 的处理

真实 global IR 中入口不是直接 tensor 参数，而是：

```mlir
%0 = hal.buffer_view.dim<%arg0>[0] : index
%1 = hal.buffer_view.dim<%arg0>[1] : index
%2 = hal.tensor.import wait(%arg2) => %arg0
     : !hal.buffer_view -> tensor<?x?xi32>{%0, %1}
```

因此 pass 同时支持两类位置：

```text
函数 tensor 参数
hal.tensor.import 的 rank-2 dynamic tensor 结果
```

在真实 DeepSeek dynamic global IR 中，pass 输出类似：

```mlir
%2 = hal.tensor.import wait(%arg2) => %arg0
    : !hal.buffer_view -> tensor<?x?xi32>{%0, %1}
%dim = tensor.dim %2, %c0 : tensor<?x?xi32>
%dim_13 = tensor.dim %2, %c1 : tensor<?x?xi32>
%3 = util.assume.int %dim<umin = 1, umax = 8> : index
%4 = util.assume.int %dim_13<umin = 1, umax = 128> : index
%extracted_slice = tensor.extract_slice %2[0, 0] [%3, %4] [1, 1]
```

### 6.3 bounded loop bound 推导

在 `ConfigUtils.cpp` 中新增：

```cpp
inferDynamicLoopUpperBounds(linalg::LinalgOp op, SmallVectorImpl<int64_t> &bounds)
```

它会从 linalg operand 的 tensor dim bound 反推 loop bound 上界。

这样对于带 `util.assume.int` 的动态 tensor，codegen 可以把某些 dynamic loop 看成有静态 upper bound，从而选择更好的 lowering config。

## 7. bounded attention micro 实验

针对 attention context micro：

```text
scores: tensor<?x?x?xf16>
value:  tensor<?x?x128xf16>
out:    tensor<?x?x128xf32>
```

做过三个版本：

```text
dynamic_unbounded
dynamic_bounded
static
```

结果：

| Case | Pipeline | Time |
|---|---|---:|
| dynamic_unbounded | VectorDistribute | 20.2 ms |
| dynamic_bounded | TileAndFuse + NV_MMA_SYNC + padding | 0.583 ms |
| static | TileAndFuse + NV_MMA_SYNC + padding | 0.584 ms |

对应 IR 中，bounded dynamic context 可以得到：

```mlir
translation_info = #iree_codegen.translation_info<
  pipeline = #iree_gpu.pipeline<TileAndFuse>
>

lowering_config = #iree_gpu.lowering_config<{
  convert_acc_gemm,
  mma_kind = #iree_gpu.mma_layout<NV_MMA_SYNC_F32_16x8x16_F16>,
  padding = [1, 64, 64, 128],
  promote_operands = [0, 1],
  reduction = [0, 0, 0, 8],
  subgroup = [0, 2, 4, 0],
  workgroup = [1, 64, 64, 0]
}>
```

这个 micro 实验证明：

```text
attention context 的数学结构本身可以被 dynamic bounded MMA 优化到接近静态。
```

但是，micro 成功不等于完整模型成功。

## 8. 完整模型中的 bounded attention 问题

### 8.1 直接打开 bounded attention MMA

将 bounded shape pass 应用到真实 DeepSeek dynamic global IR 后，完整模型中的 attention context dispatch 变成：

```text
dispatch_41_batch_matmul_DxDx128xD_f16xf16xf32
pipeline = TileAndFuse
mma_kind = NV_MMA_SYNC_F32_16x8x16_F16
padding = [1, 1, 32, 16, 128]
```

但是完整模型 benchmark：

```text
bounded_dynamic_b8_s128.vmfb
  B=1, S=32
  timeout 60s
  status = 124
```

对比：

```text
safe_dynamic_m
  同样 B=1, S=32
  正常运行，约 84-89 ms
```

因此 timeout 不是参数、输入或 GPU 问题，而是 attention context 的完整模型 lowering 问题。

### 8.2 可能原因

完整模型中的 attention context 不只是一个干净的 standalone matmul。它包含：

```text
rank-5 dynamic tensor
expand_shape / transpose 后的 layout
indirect binding
dynamic workload operands
padding + MMA 的边界 tile 处理
多层重复 dispatch
```

这些条件叠加后，虽然 IR 可以生成 `TileAndFuse + MMA`，但 runtime 执行会 hang。

因此当前不能简单地说：

```text
只要给 B/S upper bound，attention 就能安全走 MMA
```

## 9. guarded attention 修复

为了避免“编译成功但运行 hang”，当前加入了保守保护：

```cpp
bool hasBoundedDynamicLoops =
    bounds.size() <= 3 && inferDynamicLoopUpperBounds(linalgOp, bounds);
```

含义是：

```text
rank <= 3 的普通/flattened matmul 可以使用 bounded dynamic upper-bound。
rank-5 attention batch_matmul 暂时不使用 bounded upper-bound 进入 MMA。
```

guarded 后，完整模型中：

```text
dispatch_41 attention context
  回到 VectorDistribute

projection / MLP / lm_head
  仍然保持 TileAndFuse + MMA
```

重新 benchmark：

```text
safe_dynamic_m:                  88.826 ms
bounded_dynamic_guarded:         95.755 ms
static_exact:                    39.768 ms
```

对应关系：

```text
bounded guarded / safe_dynamic_m = 1.078x
safe_dynamic_m / static_exact    = 2.234x
bounded guarded / static_exact   = 2.408x
```

因此 guarded 版本恢复了可运行性，但没有收益，反而略慢。

## 10. 当前 IR 对比结论

### 10.1 safe dynamic-M 当前有效

普通 matmul 已经进入：

```text
pipeline = TileAndFuse
mma_kind = NV_MMA_SYNC_F32_16x8x16_F16
padding = [32, 16, 128]
```

这解释了为什么原始动态模型从几秒级下降到几十/几百毫秒级。

### 10.2 attention context 是主要未解瓶颈

保守版本中：

```text
dispatch_41 attention context
  pipeline = VectorDistribute
```

强行 bounded MMA 版本中：

```text
dispatch_41 attention context
  pipeline = TileAndFuse + MMA
  完整模型 timeout
```

这说明 attention context 是后续优化的关键，但不能直接套用普通 dynamic-M matmul 的策略。

### 10.3 attention context SIMT TileAndFuse fallback

在 2026-06-14 的继续优化中，新增了一个很窄的 fallback：

```text
目标形态：
  (..., M, K) x (..., K, 128) -> (..., M, 128)

约束：
  lhs/rhs = f16
  output = f32
  N = 128
  K = dynamic
  M = dynamic
```

这个形态正对应 DeepSeek attention context：

```text
dispatch_41_batch_matmul_DxDx128xD_f16xf16xf32
```

优化前：

```text
dispatch_41:
  pipeline = VectorDistribute
```

优化后：

```text
dispatch_41:
  pipeline = TileAndFuse
  workgroup_size = [32, 1, 1]
  subgroup_size = 32
```

全局 pipeline 分布从：

```text
14 pipeline = #iree_gpu.pipeline<Distribute>
31 pipeline = #iree_gpu.pipeline<TileAndFuse>
 3 pipeline = #iree_gpu.pipeline<VectorDistribute>
```

变为：

```text
14 pipeline = #iree_gpu.pipeline<Distribute>
32 pipeline = #iree_gpu.pipeline<TileAndFuse>
 2 pipeline = #iree_gpu.pipeline<VectorDistribute>
```

这说明 attention context 不再走最慢的 generic VectorDistribute 路径。它没有强行使用动态 K 的 MMA，因此避开了前面完整模型 timeout 的问题；同时又比原来的 VectorDistribute 更接近静态路径。

### 10.4 最新 b1_s32 真实性能结果

对同一个真实 DeepSeek dynamic 模型，B=1、S=32，CUDA sm_86 上测试：

| Version | Time | 相对关系 |
|---|---:|---:|
| safe_dynamic_m | 85.820 ms | baseline |
| bounded dynamic + attention context SIMT TileAndFuse | 40.298 ms | 2.130x faster than safe_dynamic_m |
| static_exact | 39.749 ms | reference |

关键结论：

```text
bounded_dynamic_b8_s128_simt_context / static_exact = 1.014x
gap = 1.38%
```

这已经达到本阶段最初目标：

```text
单个动态 shape VMFB 在 b1_s32 上接近静态模型性能。
```

它也说明动态模型此前剩余的约 2.2x 差距，主要确实来自 attention context 的 slow lowering，而不是 dynamic ABI 本身不可优化。

### 10.5 full grid 性能结果

随后对多个 B/S shape 做 full grid benchmark，结果如下：

| Case | safe_dynamic_m | SIMT context dynamic | static_exact | SIMT / safe | speedup | SIMT / static |
|---|---:|---:|---:|---:|---:|---:|
| b1_s32 | 88.588 ms | 40.249 ms | 39.739 ms | 0.454x | 2.201x | 1.013x |
| b1_s64 | 143.906 ms | 72.263 ms | 44.723 ms | 0.502x | 1.991x | 1.616x |
| b1_s128 | 261.293 ms | 155.732 ms | 55.889 ms | 0.596x | 1.678x | 2.786x |
| b4_s32 | 263.339 ms | 153.062 ms | 106.194 ms | 0.581x | 1.720x | 1.441x |
| b4_s64 | 507.398 ms | 309.138 ms | 124.534 ms | 0.609x | 1.641x | 2.482x |
| b4_s128 | 1003.436 ms | 635.175 ms | 171.661 ms | 0.633x | 1.580x | 3.700x |
| b8_s32 | 511.559 ms | 307.034 ms | 58.614 ms | 0.600x | 1.666x | 5.238x |
| b8_s64 | 1010.143 ms | 624.763 ms | 102.365 ms | 0.618x | 1.617x | 6.103x |
| b8_s128 | 2000.556 ms | 1285.012 ms | 121.285 ms | 0.642x | 1.557x | 10.595x |

这个表说明了两件事。

第一，SIMT context fallback 对所有测试 shape 都有效：

```text
safe_dynamic_m -> SIMT context dynamic:
  最好 b1_s32: 2.201x
  最差 b8_s128: 1.557x
  所有 case 均有明确收益
```

第二，SIMT context fallback 不是最终性能形态：

```text
SIMT context dynamic -> static_exact:
  b1_s32:   1.013x，几乎等价
  b1_s128:  2.786x
  b4_s128:  3.700x
  b8_s128: 10.595x
```

随着 B/S 增大，attention context 的计算量快速增大。静态模型可以用更强的静态 attention lowering，而当前动态版本只把 context 从 `VectorDistribute` 提升到 SIMT `TileAndFuse`，没有进入 MMA。因此小 shape 上动态和静态接近，大 shape 上差距重新拉开。

更准确的阶段结论是：

```text
SIMT context fallback 证明了 attention context 是真实瓶颈，也证明了动态路径可继续优化。
但要让 b4/b8、s64/s128 也接近静态，还需要解决 dynamic attention context 的 MMA lowering。
```

### 10.6 dynamic attention context MMA 实验

为了确认大 shape 差距是否来自 “SIMT 而不是 MMA”，加入了一个显式实验开关：

```text
--iree-codegen-test-dynamic-attention-context-mma=true
```

该开关只用于实验，不影响默认 SIMT fallback 路径。它对动态 attention context：

```text
dispatch_41_batch_matmul_DxDx128xD_f16xf16xf32
```

临时使用代表性上界选择 MMA config：

```text
batch/head upper bound = 256
M upper bound          = 128
K upper bound          = 128
N                      = 128
```

生成 executable-configurations 时，`dispatch_41` 可以得到接近静态的 MMA config：

```mlir
translation_info = #iree_codegen.translation_info<
  pipeline = #iree_gpu.pipeline<TileAndFuse>
  workgroup_size = [128, 1, 1]
  subgroup_size = 32
>

lowering_config = #iree_gpu.lowering_config<{
  convert_acc_gemm,
  mma_kind = #iree_gpu.mma_layout<NV_MMA_SYNC_F32_16x8x16_F16>,
  promote_operands = [0, 1],
  reduction = [0, 0, 0, 0, 8],
  subgroup = [0, 0, 2, 4, 0],
  workgroup = [1, 1, 64, 64, 0]
}>
```

这和静态 b8_s128 的 attention context 很接近。静态 b8_s128 中对应 dispatch 是：

```text
dispatch_12_batch_matmul_256x128x128x128_f16xf16xf32
  pipeline = TileAndFuse
  workgroup_size = [128, 1, 1]
  mma_kind = NV_MMA_SYNC_F32_16x8x16_F16
  workgroup = [1, 64, 64, 0]
```

但是继续编译完整 VMFB 时失败：

```text
error: op was not bufferized
see current operation: iree_codegen.inner_tiled
kind = NV_MMA_SYNC_F32_16x8x16_F16
```

因此当前新的定位是：

```text
动态 attention context 的 MMA config 选择已经可以构造出来；
但 rank-5 dynamic attention context 进入 inner_tiled MMA 后，后续 bufferization/LLVMGPU lowering 不能完成。
```

这比之前 “MMA 版本 runtime timeout” 更进一步：当前阻塞点已经收窄到 `inner_tiled` 动态 rank-5 MMA 的 bufferization/lowering 支持。

### 10.7 bounded input-shape pass 不是单独最终方案

bounded pass 本身有价值，因为它证明了：

```text
动态输入 + 编译期 upper bound 可以帮助 codegen 看到更多信息。
```

但当前完整模型里：

```text
打开 attention MMA -> timeout
关闭 attention MMA -> 没有收益
改为 attention context SIMT TileAndFuse fallback -> b1_s32 接近静态
```

所以 bounded input-shape 只有和更合理的 attention context config 选择结合，才形成当前可用的动态 shape 优化路径。

## 11. 与 bucket 部署的区别

bucket 静态部署：

```text
b1_s32.vmfb
b1_s64.vmfb
b4_s128.vmfb
每个 VMFB 是固定 shape
```

bounded dynamic：

```text
一个 VMFB 接受 1 <= B <= 8, 1 <= S <= 128
输入 ABI 仍然是动态 shape
编译器基于 upper bound 选择 config
```

它们都需要某种 shape 范围管理，但粒度不同：

```text
bucket 是单点 shape
bounded dynamic 是范围 shape
```

本阶段发现：

```text
对于普通 matmul，不需要 bucket，也不需要 bounded range，safe dynamic-M 已经有效。
对于 attention context，直接 bounded MMA 在完整模型中不稳定，但 SIMT TileAndFuse fallback 已经能让 b1_s32 接近静态。
```

## 12. 当前使用的关键命令

### 12.1 生成 bounded global IR

```bash
/home/zhongjialin/projects/iree-build/tools/iree-opt \
  --pass-pipeline='builtin.module(util.func(iree-global-opt-assume-input-shape-bounds{max-batch=8 max-seq=128}))' \
  /tmp/deepseek_dynamic_bs_global.mlir \
  -o /tmp/deepseek_dynamic_bs_global_bounded_b8_s128.mlir
```

### 12.2 查看 executable config

```bash
/home/zhongjialin/projects/iree-build/tools/iree-compile \
  --iree-hal-target-backends=cuda \
  --iree-cuda-target=sm_86 \
  --iree-gpu-test-target=sm_86 \
  --compile-from=global-optimization \
  --compile-to=executable-configurations \
  /tmp/deepseek_dynamic_bs_global_bounded_b8_s128.mlir \
  -o /tmp/deepseek_dynamic_bs_bounded_b8_s128_guarded_attention_exec_config.mlir
```

### 12.3 查看 pipeline 分布

```bash
grep -o "pipeline = #iree_gpu.pipeline<[^>]*>" \
  /tmp/deepseek_dynamic_bs_bounded_b8_s128_guarded_attention_exec_config.mlir \
  | sort | uniq -c
```

当前 guarded 版本：

```text
14 pipeline = #iree_gpu.pipeline<Distribute>
31 pipeline = #iree_gpu.pipeline<TileAndFuse>
 3 pipeline = #iree_gpu.pipeline<VectorDistribute>
```

当前 attention context SIMT fallback 版本：

```text
14 pipeline = #iree_gpu.pipeline<Distribute>
32 pipeline = #iree_gpu.pipeline<TileAndFuse>
 2 pipeline = #iree_gpu.pipeline<VectorDistribute>
```

其中：

```text
dispatch_41_batch_matmul_DxDx128xD_f16xf16xf32
  pipeline = TileAndFuse
```

### 12.4 benchmark 当前动态优化版本

```bash
cd /home/zhongjialin/projects/iree/deepseek-R1-Llama-8b

GPU=2 B=1 S=32 REPETITIONS=3 MIN_TIME=1x WARMUP_TIME=0 BENCH_TIMEOUT=60s \
  /home/zhongjialin/projects/iree_llm_matmul_flatten_handoff/scripts/deepseek/benchmark_bounded_dynamic_shape.py
```

### 12.5 benchmark 多 shape grid

```bash
cd /home/zhongjialin/projects/iree/deepseek-R1-Llama-8b

GPU=2 REPETITIONS=3 MIN_TIME=1x WARMUP_TIME=0 BENCH_TIMEOUT=120s \
  python3 /path/to/future_dynamic_simt_context_grid.py
```

只测指定 shape：

```bash
GPU=2 B_VALUES="1 4" S_VALUES="32 64" \
  python3 /path/to/future_dynamic_simt_context_grid.py
```

## 13. 文件与代码改动

### 13.1 Codegen 修改

```text
compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp
```

主要内容：

```text
1. dynamic-M heuristic
2. inferDynamicLoopUpperBounds
3. bounded dynamic loop 推导
4. bounded dynamic loop 推导限制在 rank <= 4，避免 rank-5 attention batch_matmul 进入不稳定 MMA
5. attention context dynamic-K 形态使用 SIMT TileAndFuse fallback
```

```text
compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp
```

主要内容：

```text
1. dynamic dim 也计入 non-unit parallel dim
2. dynamic K 和 dynamic M/N 同时存在时 fallback
3. skinny dim 检查增加 static guard
```

### 13.2 GlobalOptimization pass

```text
compiler/src/iree/compiler/GlobalOptimization/AssumeInputShapeBounds.cpp
```

主要内容：

```text
1. 为 rank-2 dynamic tensor 参数插入 util.assume.int
2. 为 hal.tensor.import 后的 rank-2 dynamic tensor 插入 util.assume.int
3. 通过 tensor.extract_slice 将 tensor value 和 bounded dim 绑定
```

### 13.3 Benchmark 脚本

```text
iree_llm_matmul_flatten_handoff/scripts/deepseek/benchmark_bounded_dynamic_shape.py
```

主要内容：

```text
1. 动态 VMFB 使用 dynamic_shape_b_s 对应 irpa
2. 静态 VMFB 使用 exact static shape 对应 irpa
3. 静态 last_nonpad_index 模型自动传第三个输入
4. 增加 BENCH_TIMEOUT，避免 hang 占住 GPU
5. summary 输出 safe_vs_static 和 bounded_vs_static 差距
```

```text
historical shell wrapper: benchmark_dynamic_simt_context_grid; use the Python commands documented in troubleshooting/03_dynamic_shape_attention_issues.md for future work
```

主要内容：

```text
1. 循环测试多个 B/S shape
2. 每个 shape 对比 safe_dynamic_m / simt_context dynamic / static_exact
3. 缺少静态 exact vmfb 的 shape 会自动跳过
4. 汇总输出 simt_vs_safe 和 simt_vs_static
```

## 14. 阶段性结论

当前阶段可以明确分成两部分。

### 14.1 已经成功的部分

```text
projection / MLP / lm_head 的 dynamic-M matmul 优化是有效的。
attention context 的 SIMT TileAndFuse fallback 在 b1_s32 上有效。
```

证据：

```text
原始动态模型从几秒级下降到几十/几百毫秒级。
输出和 old dynamic / static exact 在 atol=0.06, rtol=0.01 下匹配。
IR 中普通 matmul 进入 TileAndFuse + NV_MMA_SYNC。
IR 中 dispatch_41 attention context 从 VectorDistribute 变成 TileAndFuse。
b1_s32: 85.820 ms -> 40.298 ms，距离 static_exact 39.749 ms 只差 1.38%。
full grid 中所有 shape 都比 safe_dynamic_m 快 1.557x 到 2.201x。
```

这部分可以作为当前动态 shape 优化的最新核心成果。

### 14.2 仍需验证和解决的部分

```text
attention context 的动态 K MMA lowering 仍然不能直接打开。
SIMT fallback 在 b4/b8、s64/s128 上仍明显慢于静态。
```

证据：

```text
bounded micro 可以接近静态。
完整模型中打开 attention context MMA 会 60s timeout。
实验开关版本可以生成 MMA executable-config，但完整 VMFB 编译失败于 inner_tiled bufferization。
SIMT fallback 能让 b1_s32 接近静态。
但 b8_s128 仍是 static_exact 的 10.595x。
```

因此下一阶段应该集中修 dynamic K attention context MMA，或者寻找比当前 SIMT fallback 更接近静态的动态 attention lowering。

## 15. 下一步建议

### 15.1 建立 attention context 最小复现

目标是从完整模型中抽取或构造更接近真实 dispatch 的 repro，而不是只用干净 micro。

重点保留：

```text
rank-5 shape
expand_shape / transpose 后 layout
indirect binding
dynamic workload operands
padding config
```

### 15.2 分别验证 dispatch_39 和 dispatch_41

当前重点 dispatch：

```text
dispatch_39_batch_matmul_DxDxDx128_f16xf16xf32   attention scores
dispatch_41_batch_matmul_DxDx128xD_f16xf16xf32   attention context
```

建议分别构造：

```text
scores MMA on, context conservative
scores conservative, context MMA on
```

这样可以确认 timeout 是否完全来自 context，还是 scores/context 组合导致。

### 15.3 修复 dynamic rank-5 MMA lowering

目标不是依赖 bucket 或 bounded deployment，而是让 IREE CUDA codegen 支持：

```text
rank-5 dynamic batch_matmul
dynamic B/S
dynamic workload
fixed MMA tile
runtime mask/padding
正确处理边界 tile
```

理想最终状态：

```text
单个动态 VMFB
projection / MLP / lm_head -> dynamic-M MMA
attention scores/context  -> dynamic batch_matmul MMA
softmax/reduction         -> 保守但稳定 lowering
```

这样才是真正意义上让动态 shape 模型接近静态模型性能。

## 16. 继续优化：flat attention context MMA

上一阶段的 context MMA 失败点是：

```text
dynamic attention context 能在 executable-configurations 阶段选到 MMA，
但是 BlockDynamicDimensions 会把原本 flatten 的 batch 维重新拆成 ? x 32，
最终形成 rank-5 dynamic inner_tiled：

tensor<?x32x?x?xf16> x tensor<?x32x?x128xf16>

完整 VMFB 编译失败于 inner_tiled bufferization。
```

本轮继续定位后发现，`executable-sources` 阶段的 dispatch_41 其实仍然是理想的 flatten 形态：

```mlir
%57 = iree_tensor_ext.dispatch.tensor.load ... -> tensor<?x?x?xf16>
%58 = iree_tensor_ext.dispatch.tensor.load ... -> tensor<?x?x128xf16>
%60 = linalg.fill ... outs(%59 : tensor<?x?x128xf32>)
%61 = linalg.batch_matmul
  ins(%57, %58 : tensor<?x?x?xf16>, tensor<?x?x128xf16>)
  outs(%60 : tensor<?x?x128xf32>)
  -> tensor<?x?x128xf32>
```

也就是说，rank-5 不是 global optimization 产生的，也不是 dispatch creation 必然产生的，而是 LLVMGPU codegen common configuration 中的 `BlockDynamicDimensionsPass` 根据 `udiv=32` 把动态维重新物化成：

```text
?  ->  ? x 32
```

这对某些 SIMT/vector 分发是有帮助的，但对当前 dynamic attention context MMA 会触发后续 rank-5 dynamic `inner_tiled` bufferization 问题。

### 16.1 本轮新增实验开关

新增开关：

```text
--iree-codegen-test-skip-block-dynamic-attention-context=true
```

作用：

```text
只对 (..., M, K) x (..., K, 128) -> (..., M, 128)
且 f16/f16 -> f32 的 dynamic attention context contraction
跳过 BlockDynamicDimensions。

其他 matmul / projection / MLP 不受影响。
```

同时修复了 `setMatmulLoweringConfig` 中一个 matcher 问题：

```text
inferDynamicLoopUpperBounds 会把原始 dynamic M/K 推成静态上界，
如果后续 matcher 再用这个已经修改过的 bounds 判断 “K 是否 dynamic”，
就会误判 attention context 不匹配。

修复后：
  originalBounds 用于识别是否是 dynamic attention context
  inferred bounds 用于选择 MMA schedule
```

### 16.2 新 IR 证据

使用两个实验开关：

```bash
/home/zhongjialin/projects/iree-build/tools/iree-compile \
  --iree-hal-target-backends=cuda \
  --iree-cuda-target=sm_86 \
  --iree-gpu-test-target=sm_86 \
  --iree-codegen-test-dynamic-attention-context-mma=true \
  --iree-codegen-test-skip-block-dynamic-attention-context=true \
  --compile-from=global-optimization \
  --compile-to=executable-configurations \
  /tmp/deepseek_dynamic_bs_global_bounded_b8_s128.mlir \
  -o /tmp/deepseek_dynamic_bs_bounded_b8_s128_mma_flat_context_exec_config.mlir
```

得到的 dispatch_41 已经是 flat rank-4 loop 的 MMA config：

```mlir
func.func @main_graph$async_dispatch_41_batch_matmul_DxDx128xD_f16xf16xf32()
  attributes {
    translation_info =
      #iree_codegen.translation_info<
        pipeline = #iree_gpu.pipeline<TileAndFuse>
        workgroup_size = [64, 1, 1]
        subgroup_size = 32
      >
  }

%61 = linalg.generic
  iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  ins(%57, %58 : tensor<?x?x?xf16>, tensor<?x?x128xf16>)
  outs(%60 : tensor<?x?x128xf32>)
  attrs = {
    lowering_config = #iree_gpu.lowering_config<{
      convert_acc_gemm,
      mma_kind = #iree_gpu.mma_layout<NV_MMA_SYNC_F32_16x8x16_F16>,
      padding = [1, 32, 16, 128],
      promote_operands = [0, 1],
      reduction = [0, 0, 0, 8],
      subgroup = [0, 1, 2, 0],
      workgroup = [1, 32, 16, 0]
    }>
  }
```

这个结果和上一轮 rank-5 MMA 失败相比，关键区别是：

```text
上一轮：
  tensor<?x32x?x?xf16>
  tensor<?x32x?x128xf16>
  rank-5 inner_tiled
  VMFB 编译失败

本轮：
  tensor<?x?x?xf16>
  tensor<?x?x128xf16>
  flat rank-4 contraction loops
  VMFB 编译成功
```

### 16.3 新候选 VMFB

完整编译命令：

```bash
/home/zhongjialin/projects/iree-build/tools/iree-compile \
  --iree-hal-target-backends=cuda \
  --iree-cuda-target=sm_86 \
  --iree-gpu-test-target=sm_86 \
  --iree-codegen-test-dynamic-attention-context-mma=true \
  --iree-codegen-test-skip-block-dynamic-attention-context=true \
  --compile-from=global-optimization \
  /tmp/deepseek_dynamic_bs_global_bounded_b8_s128.mlir \
  -o /tmp/deepseek_dynamic_bs_bounded_b8_s128_mma_flat_context.vmfb
```

结果：

```text
/tmp/deepseek_dynamic_bs_bounded_b8_s128_mma_flat_context.vmfb
编译成功。
```

### 16.4 CUDA 验证指令

先跑一个小 shape sanity：

```bash
cd /home/zhongjialin/projects/iree

GPU=2 B=1 S=32 REPETITIONS=3 MIN_TIME=1x WARMUP_TIME=0 BENCH_TIMEOUT=120s \
BOUNDED_DYNAMIC_NAME=mma_flat_context \
BOUNDED_DYNAMIC_VMFB=/tmp/deepseek_dynamic_bs_bounded_b8_s128_mma_flat_context.vmfb \
OUT_DIR=/tmp/iree_mma_flat_context_benchmark_b1_s32 \
  iree_llm_matmul_flatten_handoff/scripts/deepseek/benchmark_bounded_dynamic_shape.py
```

再跑完整 grid：

```bash
cd /home/zhongjialin/projects/iree

GPU=2 REPETITIONS=3 MIN_TIME=1x WARMUP_TIME=0 BENCH_TIMEOUT=180s \
BOUNDED_DYNAMIC_NAME=mma_flat_context \
BOUNDED_DYNAMIC_VMFB=/tmp/deepseek_dynamic_bs_bounded_b8_s128_mma_flat_context.vmfb \
OUT_ROOT=/tmp/iree_dynamic_mma_flat_context_grid \
  historical shell wrapper: benchmark_dynamic_simt_context_grid; use the Python commands documented in troubleshooting/03_dynamic_shape_attention_issues.md for future work
```

正确性验证：

```bash
cd /home/zhongjialin/projects/iree

GPU=2 B=1 S=32 \
CANDIDATE_DYNAMIC_VMFB=/tmp/deepseek_dynamic_bs_bounded_b8_s128_mma_flat_context.vmfb \
OUT_DIR=/tmp/iree_mma_flat_context_correctness_b1_s32 \
  iree_llm_matmul_flatten_handoff/scripts/deepseek/check_safe_dynamic_m_correctness.py
```

建议至少补三组：

```bash
GPU=2 B=1 S=64  CANDIDATE_DYNAMIC_VMFB=/tmp/deepseek_dynamic_bs_bounded_b8_s128_mma_flat_context.vmfb OUT_DIR=/tmp/iree_mma_flat_context_correctness_b1_s64  iree_llm_matmul_flatten_handoff/scripts/deepseek/check_safe_dynamic_m_correctness.py
GPU=2 B=4 S=64  CANDIDATE_DYNAMIC_VMFB=/tmp/deepseek_dynamic_bs_bounded_b8_s128_mma_flat_context.vmfb OUT_DIR=/tmp/iree_mma_flat_context_correctness_b4_s64  iree_llm_matmul_flatten_handoff/scripts/deepseek/check_safe_dynamic_m_correctness.py
GPU=2 B=8 S=128 CANDIDATE_DYNAMIC_VMFB=/tmp/deepseek_dynamic_bs_bounded_b8_s128_mma_flat_context.vmfb OUT_DIR=/tmp/iree_mma_flat_context_correctness_b8_s128 iree_llm_matmul_flatten_handoff/scripts/deepseek/check_safe_dynamic_m_correctness.py
```

### 16.5 当前结论

```text
动态模型继续优化已经从 SIMT fallback 进入到真正的 flat dynamic attention context MMA。

这不是 bucket，也不是多静态模型路由：
  仍然是单个 dynamic B/S VMFB。

当前最新候选：
  safe_dynamic_m + dynamic attention context flat-MMA

下一步只需要用 CUDA grid 验证性能和正确性。
如果 b4/b8 大 shape 明显下降，就说明之前 static gap 的主要来源之一确实是 context dispatch 没有走 MMA。
```
