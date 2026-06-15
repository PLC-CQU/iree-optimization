# 端到端流程技术说明

本文记录当前项目从 DeepSeek 模型导出到 IREE CUDA benchmark / correctness 的主流程。目标是让后续复现实验时能清楚知道每一步输入、输出和关键脚本。

## 1. 输入与输出

### 输入

```text
DeepSeek HuggingFace 模型目录
本地 IREE build
CUDA GPU
```

模型权重不放入本仓库。运行时通过 `--model-path` 指向本地模型目录。

### 主要输出

```text
dynamic_shape_b_s/deepseek_dynamic_b_s_last.onnx
dynamic_shape_b_s/deepseek_dynamic_b_s_last_flatten_matmul.onnx
dynamic_shape_b_s/rewrite_pass_dynamic_b_s.json
dynamic_shape_b_s/build_flatten_matmul/deepseek_r1_8b_params.irpa
deepseek_dynamic_b_s_last_flatten_matmul.vmfb
dynamic_shape_b_s/benchmark_dynamic_b*_s*_flatten_matmul.json
```

这些都是生成产物，不进入 Git。

## 2. 主入口

主入口脚本是：

```text
scripts/deepseek/run_dynamic_shape_pipeline.py
```

它把流程拆成 5 个 action：

```text
plan       只打印命令，不执行
export     导出动态 ONNX
rewrite    对 ONNX 做 flatten matmul rewrite
compile    通过 IREE import/compile 生成 CUDA VMFB
benchmark  对多个 B/S 请求 benchmark
all        顺序执行完整流程
```

推荐先用 `plan` 检查路径和命令：

```bash
cd scripts/deepseek
python3 run_dynamic_shape_pipeline.py \
  --action plan \
  --model-path /path/to/model \
  --gpu 0 \
  --cuda-target sm_86
```

确认无误后执行：

```bash
python3 run_dynamic_shape_pipeline.py \
  --action all \
  --model-path /path/to/model \
  --gpu 0 \
  --cuda-target sm_86
```

## 3. Step 1：导出动态 ONNX

脚本：

```text
scripts/deepseek/export_dynamic_onnx.py
```

主流程中对应命令形态：

```bash
python3 export_dynamic_onnx.py \
  --model-path /path/to/model \
  --out dynamic_shape_b_s/deepseek_dynamic_b_s_last.onnx \
  --sample-batch 1 \
  --sample-seq 32 \
  --device cpu \
  --last-token-only
```

导出目标是保留动态 B/S：

```text
input_ids:      tensor<?x?xi32 或 tensor<?x?xi64>
attention_mask: tensor<?x?xi32 或 tensor<?x?xi64>
```

后续 compile 时可以通过 IREE flag 或输入 patch 将整数输入统一到当前实验需要的 dtype。

## 4. Step 2：ONNX flatten matmul rewrite

脚本：

```text
scripts/deepseek/rewrite_onnx_flatten_matmul.py
```

它识别 ONNX 中如下形式：

```text
MatMul([B, S, K], [K, N]) -> [B, S, N]
```

并改写成：

```text
Reshape([B, S, K] -> [B*S, K])
MatMul([B*S, K], [K, N]) -> [B*S, N]
Reshape([B*S, N] -> [B, S, N])
```

动态 shape 模式下，`B`、`S` 和 `B*S` 通过 ONNX `Shape/Gather/Mul/Concat` 在图中构造，避免依赖固定 batch/seq。

主流程命令形态：

```bash
python3 rewrite_onnx_flatten_matmul.py \
  --input dynamic_shape_b_s/deepseek_dynamic_b_s_last.onnx \
  --output dynamic_shape_b_s/deepseek_dynamic_b_s_last_flatten_matmul.onnx \
  --dynamic-shape \
  --check \
  --report dynamic_shape_b_s/rewrite_pass_dynamic_b_s.json \
  --max-report-records 5
```

rewrite 的边界：

- 只改写 RHS 为权重 initializer 的 rank-3 activation matmul。
- attention matmul 不在 ONNX 层强行改写，因为 RHS 是运行时 tensor，不是静态权重。
- contracting dim 和输出 dim 必须可验证匹配。

## 5. Step 3：IREE import 与 CUDA 编译

脚本：

```text
scripts/deepseek/compile_onnx_iree_cuda.py
scripts/deepseek/inline_onnx_dense_resources.py
```

流程：

```text
ONNX
  -> iree-import-onnx --large-model --externalize-params
  -> external MLIR + IRPA params
  -> inline ONNX dense resources
  -> iree-compile --iree-hal-target-backends=cuda
  -> VMFB
```

主流程命令形态：

```bash
python3 compile_onnx_iree_cuda.py \
  --onnx dynamic_shape_b_s/deepseek_dynamic_b_s_last_flatten_matmul.onnx \
  --build-dir dynamic_shape_b_s/build_flatten_matmul \
  --output deepseek_dynamic_b_s_last_flatten_matmul.vmfb \
  --cuda-target sm_86 \
  --batch 1 \
  --seq 32 \
  --optimization-preset baseline
```

`compile_onnx_iree_cuda.py` 中保留了几个重要开关：

```text
--force-import
--force-compile
--no-demote-i64-to-i32
--extra-compile-flag
--optimization-preset
```

其中 `--extra-compile-flag` 用于注入 IREE 实验 flag，例如 dynamic-M、attention fallback、bounded shape 等 codegen 实验。

## 6. Step 4：Benchmark

主 benchmark 脚本：

```text
scripts/deepseek/benchmark_b1_last_cuda.py
```

`run_dynamic_shape_pipeline.py` 默认测试以下请求：

```text
b1_s16
b1_s32
b1_s48
b1_s64
b1_s96
b1_s128
b2_s48
b3_s80
b4_s64
b5_s96
b8_s128
```

单个请求命令形态：

```bash
python3 benchmark_b1_last_cuda.py \
  --module deepseek_dynamic_b_s_last_flatten_matmul.vmfb \
  --params dynamic_shape_b_s/build_flatten_matmul/deepseek_r1_8b_params.irpa \
  --gpu 0 \
  --batch 4 \
  --seq 64 \
  --input-dtype i32 \
  --repetitions 5 \
  --min-time 10x \
  --warmup-time 1.0 \
  --out dynamic_shape_b_s/benchmark_dynamic_b4_s64_flatten_matmul.json
```

safe dynamic-M 的专项 benchmark 使用：

```text
scripts/deepseek/benchmark_safe_dynamic_m.sh
```

它对比：

```text
old_dynamic
safe_dynamic_m
static_exact（如果对应 shape 的静态模型存在）
```

## 7. Step 5：正确性验证

当前保留两个主要验证入口：

```text
scripts/deepseek/verify_flatten_matmul_correctness.py
scripts/deepseek/check_safe_dynamic_m_correctness.sh
```

`verify_flatten_matmul_correctness.py` 对比 baseline VMFB 与 flatten matmul VMFB 的 logits，输出 max abs、mean abs、RMSE、relative error 和 allclose 结果。

`check_safe_dynamic_m_correctness.sh` 对比：

```text
old_dynamic vs safe_dynamic_m
safe_dynamic_m vs static_exact
candidate_dynamic vs static_exact（可选）
```

当前保留的结果见：

```text
results/deepseek/correctness_b4_s32_flatten_matmul.json
results/deepseek/correctness_b8_s64_flatten_matmul.json
results/deepseek/correctness_b8_s128_flatten_matmul.json
```

三个 shape 的 baseline 与 flatten matmul logits 都是 `max_abs_error = 0.0`。

## 8. Qwen / Gemma 对照脚本

除 DeepSeek 主流程外，仓库中保留 Qwen 和 Gemma 的测试脚本，用于验证 flatten / static shape 实验是否能迁移到其它模型：

```text
scripts/gemma/
scripts/qwen/
```

这些脚本不作为 DeepSeek 主线流程的一部分，但用于跨模型 sanity check 和 benchmark 对照。
