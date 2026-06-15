# Qwen / Gemma 对照测试脚本说明

本文记录仓库中保留的 Qwen 和 Gemma 测试脚本。这些脚本不是 DeepSeek 主流程的一部分，但用于验证 flatten / static shape / repeated benchmark 相关方法是否能迁移到其它模型。

## 1. 目录

```text
scripts/gemma/
scripts/qwen/
```

## 2. Gemma 脚本

```text
scripts/gemma/run_gemma_flatten_compare.py
```

用于对比 Gemma 模型在标准路径与 flatten 路径下的性能或输出行为。

```text
scripts/gemma/run_gemma_static_shape_experiment.py
```

用于构造 Gemma static shape 实验，观察固定 B/S 情况下 IREE CUDA lowering 和 benchmark 结果。

```text
scripts/gemma/run_gemma_standard_iree.py
```

用于运行 Gemma 的标准 IREE 编译/执行流程，作为对照基线。

```text
scripts/gemma/run_gemma_repeated_single_benchmark.py
```

用于对单个 Gemma case 做重复 benchmark，降低偶然波动对判断的影响。

```text
scripts/gemma/summarize_gemma_benchmarks.py
scripts/gemma/summarize_gemma_repeated_sweep.py
```

用于汇总 Gemma benchmark 结果。

## 3. Qwen 脚本

```text
scripts/qwen/run_qwen25_standard_flatten.py
```

用于 Qwen2.5 standard / flatten 路径对比。

```text
scripts/qwen/run_qwen35_flatten_test.py
```

用于 Qwen3.5 flatten 测试。

```text
scripts/qwen/run_qwen_repeated_single_benchmark.py
```

用于 Qwen 单 case 重复 benchmark。

## 4. 使用方式

这些脚本依赖本地 Qwen / Gemma 模型目录、IREE build、CUDA 环境和对应生成产物。模型权重和生成产物不放入本仓库。

建议使用方式：

```bash
cd scripts/gemma
python3 run_gemma_flatten_compare.py --help

cd ../qwen
python3 run_qwen25_standard_flatten.py --help
```

先查看脚本参数，再根据本地模型路径和 IREE build 配置运行。

## 5. 与 DeepSeek 主线的关系

DeepSeek 是当前主要优化对象，Qwen / Gemma 脚本用于回答两个问题：

```text
rank3 matmul flatten 的收益是否只出现在 DeepSeek。
static shape / repeated benchmark 方法是否能作为跨模型验证工具。
```

如果后续 Qwen / Gemma 产生稳定结果，建议新增独立结果汇总文档，而不要把大体积 VMFB、ONNX、IRPA 或日志放入 Git。
