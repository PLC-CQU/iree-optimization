# IREE 优化实验

本仓库整理了围绕 **IREE 动态 shape LLM 推理优化** 的代码、实验、报告和编译器补丁。当前重点是 DeepSeek 模型在动态 batch/sequence 输入下的性能问题，以及如何通过 IREE 编译器侧优化让动态模型尽量接近静态模型的 CUDA lowering 质量。

如果你之前没有接触过 IREE，建议先读：

```text
docs/reports/00_project_overview_for_newcomers.md
```

它会先解释 IREE、ONNX、VMFB、动态 shape、CUDA lowering 等背景，再说明本项目在解决什么问题。

## 当前结果概览

当前最明确的结果来自 DeepSeek 的 graph-level flatten matmul 实验：

| Shape | baseline latency | flatten latency | speedup |
|---|---:|---:|---:|
| b4_s32 | 314.07 ms | 39.53 ms | 7.94x |
| b4_s64 | 308.12 ms | 60.68 ms | 5.08x |
| b4_s128 | 342.77 ms | 114.23 ms | 3.00x |
| b8_s32 | 648.72 ms | 60.37 ms | 10.75x |
| b8_s64 | 608.49 ms | 111.46 ms | 5.46x |
| b8_s128 | 681.37 ms | 144.88 ms | 4.70x |

6 个 shape 的平均加速为 `6.155x`。同时，当前保留的 b4_s32 / b8_s64 / b8_s128 correctness 结果中，baseline 与 flatten matmul logits 的 `max_abs_error = 0.0`。

更完整的结果见 [性能结果与实验结论汇总](docs/reports/04_benchmark_results_summary.md)。

## 项目目标

动态模型入口通常具有如下输入形态：

```text
input_ids:      tensor<?x?xi32>
attention_mask: tensor<?x?xi32>
```

这会使后续很多关键计算的 loop bound 变成运行时 SSA value，例如 `B`、`S`、`B*S`。如果 IREE codegen 无法证明这些动态维度适合 MMA，就容易退到更保守的 lowering，导致动态模型明显慢于静态模型。

本项目当前围绕三件事展开：

1. 分析动态 shape 模型为什么慢，尤其是 projection / MLP / lm_head / attention context 中的 matmul 和 batch matmul。
2. 在 IREE 中实现并验证 `rank3 matmul -> flattened rank2 matmul`、safe dynamic-M matmul、bounded dynamic shape 等优化。
3. 用 DeepSeek 模型和最小 MLIR case 复现、定位、量化这些优化的效果与风险。

阶段性结论见 [动态 shape 优化阶段总结](docs/reports/iree_dynamic_shape_optimization_stage_report.md)。

## 仓库结构

```text
docs/reports/
  当前阶段报告，主要记录 DeepSeek 动态 shape 优化现状、性能结果和限制。

experiments/dynamic_shape_matmul_experiment/
  最小 MLIR 实验，用于隔离 static / dynamic shape 在 Flow、HAL、CUDA codegen 中的差异。

scripts/deepseek/
  DeepSeek 主流程脚本：动态 ONNX 导出、ONNX matmul rewrite、IREE CUDA 编译、benchmark 和正确性检查。

scripts/gemma/
scripts/qwen/
  Gemma / Qwen 对照测试脚本，用于验证 flatten / static shape 实验是否能迁移到其它模型。

results/deepseek/
  精简后的核心 benchmark / correctness 汇总结果。

results/figures/
  主体性能图。

iree-patches/
  IREE 编译器补丁、新增 pass/test 文件，以及相关源码快照。
```

本仓库不是完整产物归档。模型权重、ONNX、VMFB、IRPA、build 目录、日志和临时 tensor dump 都不放入 Git，需要本地重新生成或放到外部存储。

## 主要流程

### 1. 应用 IREE 编译器补丁

在本地 IREE checkout 中执行：

```bash
cd /path/to/iree
cp -a /path/to/iree-optimization/iree-patches/new-files/* .
git apply /path/to/iree-optimization/iree-patches/tracked_changes.patch
```

补丁中的主要内容：

- `FlattenRank3Matmul.cpp`：把 `[B,S,K] x [K,N] -> [B,S,N]` 这类 rank-3 matmul-like contraction 改写成标准二维 `linalg.matmul`。
- `AssumeInputShapeBounds.cpp`：为动态输入添加 bounded shape assumption，并支持 attention context barrier 实验。
- CUDA codegen 改动：针对 safe dynamic-M matmul 使用代表性 M bound 选择 MMA config，同时保留 dynamic K / complex dynamic contraction 的 fallback。
- 相关 BUILD / CMake / lit test 更新。

### 2. 运行最小 MLIR 实验

```bash
cd experiments/dynamic_shape_matmul_experiment
IREE_BUILD=/path/to/iree-build bash run_compare.sh
IREE_BUILD=/path/to/iree-build CUDA_ARCH=sm_86 GPU=0 bash run_extended_cuda_benchmark.sh
```

这些实验用于观察：

- Flow dispatch 数量是否变化。
- HAL 层是否从 indirect execute 变成 direct execute。
- 动态 shape 是否引入更多 shape arithmetic / dispatch constants。
- CUDA codegen 是否从 `TileAndFuse` / MMA 退到 `VectorDistribute`。
- guarded specialization 是否能恢复静态 matmul problem size。

### 3. 运行 DeepSeek 动态 shape 主流程

模型文件不放在仓库内。请将模型放在本地目录，例如 `/path/to/model`，然后执行：

```bash
cd scripts/deepseek
python3 run_dynamic_shape_pipeline.py --action plan --model-path /path/to/model
python3 run_dynamic_shape_pipeline.py --action all --model-path /path/to/model --gpu 0
```

主流程包括：

```text
导出动态 ONNX
  -> rewrite rank3 matmul / flatten matmul
  -> import + compile 为 IREE CUDA VMFB
  -> 在多个 B/S 请求上 benchmark
```

### 4. 检查正确性

代表性脚本：

```bash
cd scripts/deepseek
python3 verify_flatten_matmul_correctness.py
bash check_safe_dynamic_m_correctness.sh
```

这些检查用于确认 flatten matmul rewrite 和 safe dynamic-M 优化没有破坏模型输出语义。

## 核心脚本

```text
scripts/deepseek/export_dynamic_onnx.py
  导出动态 B/S ONNX。

scripts/deepseek/rewrite_onnx_flatten_matmul.py
  对 ONNX 图做 flatten matmul rewrite。

scripts/deepseek/compile_onnx_iree_cuda.py
  ONNX -> IREE import -> CUDA VMFB 编译。

scripts/deepseek/run_dynamic_shape_pipeline.py
  串联导出、rewrite、编译和 benchmark 的主入口。

scripts/deepseek/benchmark_b1_last_cuda.py
  运行 DeepSeek last-token CUDA benchmark。

scripts/deepseek/check_safe_dynamic_m_correctness.sh
scripts/deepseek/verify_flatten_matmul_correctness.py
  正确性检查。

scripts/gemma/run_gemma_flatten_compare.py
scripts/gemma/run_gemma_static_shape_experiment.py
scripts/gemma/run_gemma_repeated_single_benchmark.py
  Gemma flatten / static shape / repeated benchmark 测试。

scripts/qwen/run_qwen25_standard_flatten.py
scripts/qwen/run_qwen35_flatten_test.py
scripts/qwen/run_qwen_repeated_single_benchmark.py
  Qwen flatten 与 repeated benchmark 测试。
```

## 技术文档

```text
docs/reports/00_project_overview_for_newcomers.md
  面向第一次接触 IREE 的读者，解释项目背景、目标、方法和当前效果。

docs/reports/01_end_to_end_pipeline.md
  端到端流程说明：动态 ONNX 导出、rewrite、IREE 编译、benchmark、正确性验证。

docs/reports/02_compiler_optimization_design.md
  IREE 编译器优化设计：FlattenRank3Matmul、AssumeInputShapeBounds、safe dynamic-M、attention fallback。

docs/reports/03_correctness_validation_and_bugfixes.md
  正确性验证方法和验证过程中修正/规避的问题。

docs/reports/04_benchmark_results_summary.md
  当前保留性能结果的表格汇总和结论。

docs/reports/05_cross_model_test_scripts.md
  Qwen / Gemma 对照测试脚本说明。

docs/reports/iree_dynamic_shape_optimization_stage_report.md
  动态 shape 优化阶段总结。
```

## 环境说明

- 主要实验环境使用 CUDA target `sm_86`，其他 GPU 可通过 `CUDA_ARCH` 或 `--cuda-target` 覆盖。
- 需要本地可用的 IREE 工具：`iree-compile`、`iree-import-onnx`、`iree-run-module`、`iree-benchmark-module`。
- Python 依赖见 `requirements.txt`，实际运行还需要与本地模型、CUDA、IREE build 保持一致。

## 上传说明

见 [UPLOAD.md](UPLOAD.md)。
