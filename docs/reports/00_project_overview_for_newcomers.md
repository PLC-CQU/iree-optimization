# 项目总览：给第一次接触 IREE 的读者

本文是本仓库的入门说明。目标是让没有接触过 IREE 的读者也能快速理解：这个项目是什么、为什么要做、具体做了什么、现在效果如何，以及接下来应该看哪些文件。

## 1. 一句话概括

本项目研究 **如何让 IREE 在运行 DeepSeek 等大模型时，更好地处理动态 batch / sequence 输入，并生成更快的 CUDA GPU 代码**。

更具体地说：

```text
同一个模型，输入长度 S 和 batch B 可能每次请求都不同。
这种动态 shape 对用户很方便，但对编译器优化很不友好。
本项目通过改写模型 IR 和修改 IREE 编译器，让常见 matmul 重新走上高性能 GPU lowering 路径。
```

## 2. IREE 是什么

IREE 可以理解为一个面向机器学习模型的编译和运行系统。它把模型从高层表示逐步降到具体硬件可以执行的形式。

在本项目里，主要流程是：

```text
PyTorch / HuggingFace 模型
  -> 导出 ONNX
  -> IREE import ONNX
  -> IREE 编译优化
  -> 生成 CUDA VMFB
  -> 在 GPU 上运行 benchmark / correctness
```

几个常见词：

```text
ONNX
  一种模型交换格式。本项目先把 DeepSeek 模型导出成 ONNX。

MLIR
  IREE 内部使用的多层 IR 表示。编译器优化大多发生在这一层。

VMFB
  IREE 编译后的可执行模块文件，可以被 iree-run-module 或 iree-benchmark-module 运行。

IRPA
  IREE external parameter 文件，用来存放大模型参数。

CUDA lowering
  IREE 把高层 matmul / tensor 计算降到 NVIDIA GPU 可执行代码的过程。
```

## 3. 这个项目为什么需要做

LLM 推理中最重的计算通常是 matmul。对于 DeepSeek 这类模型，常见输入是：

```text
input_ids:      tensor<?x?xi32>
attention_mask: tensor<?x?xi32>
```

这里的 `?` 表示动态维度：

```text
B = batch size，运行时才知道
S = sequence length，运行时才知道
```

动态 shape 对模型服务很自然，因为用户请求长度不同。但对编译器来说，动态 shape 会让很多优化变困难：

```text
静态 shape:
  编译器知道 B/S/K/N 的具体数值
  更容易选择 GPU tile、workgroup、MMA intrinsic
  更容易生成高性能 matmul kernel

动态 shape:
  编译器只知道某些维度运行时才确定
  可能无法证明某个 matmul 适合 MMA
  可能退到更保守、更慢的 lowering
```

项目最初观察到的问题是：动态模型比对应静态形状模型慢很多。后续分析发现，核心原因不是简单的 dispatch 数量增加，而是：

```text
matmul 的 IR 形态不够标准
动态维度导致 CUDA codegen 选择更保守的 pipeline
HAL 层需要更多 runtime shape arithmetic
attention context 的动态 batch_matmul 更难安全进入 MMA
```

## 4. 本项目做了什么

项目目前主要做了四类工作。

### 4.1 把 rank-3 matmul 改写成标准二维 matmul

LLM 中很多线性层长这样：

```text
[B, S, K] x [K, N] -> [B, S, N]
```

这本质上等价于：

```text
[B, S, K] -> [B*S, K]
[B*S, K] x [K, N] -> [B*S, N]
[B*S, N] -> [B, S, N]
```

本项目在 ONNX 层和 IREE GlobalOptimization 层都围绕这个方向做了处理：

```text
scripts/deepseek/rewrite_onnx_flatten_matmul.py
iree-patches/new-files/compiler/src/iree/compiler/GlobalOptimization/FlattenRank3Matmul.cpp
```

这样做的意义是：IREE CUDA 后端对标准 `linalg.matmul` 更容易生成高质量 GPU kernel。

### 4.2 加入 safe dynamic-M matmul 优化

很多 projection / MLP / lm_head 最终会变成：

```text
tensor<?xK> x tensor<KxN> -> tensor<?xN>
```

这里只有 M 是动态的，K 和 N 是静态权重维度。这类情况相对安全。

项目在 IREE CUDA codegen 中加入了一个 heuristic：

```text
如果只有 M 动态，K/N 静态：
  用代表性 M bound 选择 MMA config
  让它继续走 TileAndFuse / MMA 路径
```

这样既保留动态输入 ABI，又能恢复接近静态 matmul 的 GPU lowering。

### 4.3 分析 bounded dynamic shape 和 attention context

项目还尝试让编译器知道：

```text
1 <= B <= maxBatch
1 <= S <= maxSeq
```

对应 pass：

```text
AssumeInputShapeBounds.cpp
```

但 attention context 比普通 projection 更复杂。它往往涉及 dynamic batch、dynamic M、dynamic K 等组合，不能简单套用普通 dynamic-M 策略。

当前结论是：

```text
projection / MLP / lm_head:
  safe dynamic-M 是稳定有效的第一阶段优化。

attention context:
  需要更保守的 SIMT TileAndFuse fallback，或者继续修复 dynamic rank-5 MMA lowering。
```

### 4.4 做 correctness 和 benchmark 验证

仓库中保留了两类结果：

```text
correctness:
  验证优化前后 logits 是否一致。

benchmark:
  验证优化是否真的提升性能。
```

重要脚本：

```text
scripts/deepseek/verify_flatten_matmul_correctness.py
scripts/deepseek/check_safe_dynamic_m_correctness.sh
scripts/deepseek/benchmark_b1_last_cuda.py
scripts/deepseek/run_dynamic_shape_pipeline.py
```

## 5. 当前效果如何

### 5.1 Flatten matmul 的性能收益

当前保留的 DeepSeek 结果显示，graph-level flatten matmul 在 6 个 B/S shape 上都有明显收益。

结果摘要：

```text
平均加速: 6.155x
最大加速: 10.746x
最小加速: 3.001x
```

逐 shape 结果：

| Shape | baseline latency | flatten latency | speedup |
|---|---:|---:|---:|
| b4_s32 | 314.07 ms | 39.53 ms | 7.94x |
| b4_s64 | 308.12 ms | 60.68 ms | 5.08x |
| b4_s128 | 342.77 ms | 114.23 ms | 3.00x |
| b8_s32 | 648.72 ms | 60.37 ms | 10.75x |
| b8_s64 | 608.49 ms | 111.46 ms | 5.46x |
| b8_s128 | 681.37 ms | 144.88 ms | 4.70x |

对应结果文件：

```text
results/deepseek/benchmark_b4_b8_seq32_64_128_flatten_matmul_summary.json
```

### 5.2 Correctness 结果

当前保留的 correctness 结果：

| Shape | output shape | max_abs_error |
|---|---:|---:|
| b4_s32 | `[4, 128256]` | 0.0 |
| b8_s64 | `[8, 128256]` | 0.0 |
| b8_s128 | `[8, 128256]` | 0.0 |

也就是说，当前保留的 flatten matmul 优化在这些 case 上没有改变 logits。

### 5.3 底层 flag 微调效果有限

项目也测试过 prefetch、vector distribution、shared memory reuse 等底层 flag。结果显示这些微调基本持平，不能解决主要瓶颈。

这说明主要矛盾在于：

```text
IR 结构
动态 shape 信息
CUDA lowering pipeline 选择
```

而不是单独调几个底层参数。

## 6. 仓库里各目录怎么看

建议阅读顺序：

```text
README.md
  快速了解仓库结构和运行方式。

docs/reports/00_project_overview_for_newcomers.md
  面向新人的项目总览，也就是本文。

docs/reports/01_end_to_end_pipeline.md
  看完整流程怎么跑。

docs/reports/04_benchmark_results_summary.md
  看现在效果怎么样。

docs/reports/03_correctness_validation_and_bugfixes.md
  看 correctness 怎么做，以及过程中修过哪些验证问题。

docs/reports/02_compiler_optimization_design.md
  看 IREE 编译器具体改了什么。

docs/reports/iree_dynamic_shape_optimization_stage_report.md
  看更详细的阶段性技术总结。
```

主要目录：

```text
scripts/deepseek/
  DeepSeek 主流程脚本。

experiments/dynamic_shape_matmul_experiment/
  最小 MLIR 实验，用于解释 dynamic shape 为什么慢。

iree-patches/
  IREE 编译器补丁和源码快照。

results/
  精简后的核心结果。

scripts/gemma/
scripts/qwen/
  跨模型对照测试脚本。
```

## 7. 现在项目处于什么状态

已经完成并有结果支撑的部分：

```text
DeepSeek graph-level flatten matmul benchmark
baseline vs flatten matmul correctness
IREE FlattenRank3Matmul pass
safe dynamic-M matmul codegen heuristic
bounded dynamic shape / attention context 的阶段性分析
Qwen / Gemma 对照测试脚本整理
```

仍在继续推进的部分：

```text
attention context 的 dynamic rank-5 MMA lowering
safe_dynamic_m 多 shape 统一结果汇总
Qwen / Gemma 正式结果汇总
更完整的 dynamic B/S 单 VMFB 性能评估
```

## 8. 最短复现实验路线

如果只是想快速理解项目，可以先不跑完整 DeepSeek，只看最小 MLIR 实验：

```bash
cd experiments/dynamic_shape_matmul_experiment
IREE_BUILD=/path/to/iree-build bash run_extended_cuda_ir_compare.sh
IREE_BUILD=/path/to/iree-build CUDA_ARCH=sm_86 GPU=0 bash run_specialization_cuda_benchmark.sh
```

如果要跑 DeepSeek 主流程：

```bash
cd scripts/deepseek
python3 run_dynamic_shape_pipeline.py --action plan --model-path /path/to/model
python3 run_dynamic_shape_pipeline.py --action all --model-path /path/to/model --gpu 0
```

模型权重、ONNX、VMFB、IRPA 和 build 产物不进入本仓库，需要在本地生成。

## 9. 读者应该带走的核心认识

这个项目不是单纯“跑一个模型 benchmark”，而是在回答一个编译器问题：

```text
当 LLM 输入 shape 是动态时，IREE 为什么生成慢代码？
哪些动态结构可以安全恢复高性能 GPU matmul lowering？
哪些结构仍然需要更深入的 compiler / codegen 支持？
```

当前最明确的结果是：

```text
rank3 matmul flatten 和 safe dynamic-M 是有效方向。
attention context 是下一阶段缩小动态模型与静态模型差距的重点。
```
