# 正确性验证与问题修正说明

本文整理当前项目中与正确性验证相关的技术细节，以及过程中修正或规避的典型问题。

## 1. 验证目标

当前主要验证两类改动：

```text
ONNX / MLIR 中的 flatten matmul rewrite 是否保持 logits 一致。
safe dynamic-M / candidate dynamic VMFB 是否与旧 dynamic 或 static exact 输出一致。
```

核心原则：

```text
性能优化不能改变模型输出语义。
对比对象必须使用相同输入、相同参数文件、相同 last-token 语义。
```

## 2. baseline vs flatten matmul 验证

脚本：

```text
scripts/deepseek/verify_flatten_matmul_correctness.py
```

它运行两个 VMFB：

```text
baseline VMFB
flatten_matmul VMFB
```

并对输出 logits 计算：

```text
max_abs_error
mean_abs_error
rmse
max_relative_error
mean_relative_error
allclose(atol=1e-3, rtol=1e-3)
allclose(atol=1e-2, rtol=1e-2)
```

当前保留结果：

| Shape | 输出 shape | max_abs_error | mean_abs_error | allclose |
|---|---:|---:|---:|---|
| b4_s32 | `[4, 128256]` | 0.0 | 0.0 | true |
| b8_s64 | `[8, 128256]` | 0.0 | 0.0 | true |
| b8_s128 | `[8, 128256]` | 0.0 | 0.0 | true |

对应文件：

```text
results/deepseek/correctness_b4_s32_flatten_matmul.json
results/deepseek/correctness_b8_s64_flatten_matmul.json
results/deepseek/correctness_b8_s128_flatten_matmul.json
```

结论：对这些 shape，flatten matmul rewrite 与 baseline logits 完全一致。

## 3. safe dynamic-M 验证

脚本：

```text
scripts/deepseek/check_safe_dynamic_m_correctness.sh
```

验证对象：

```text
old_dynamic
safe_dynamic_m
static_exact
candidate_dynamic（可选）
```

对比关系：

```text
old_dynamic vs safe_dynamic_m
safe_dynamic_m vs static_exact
candidate_dynamic vs static_exact（如果提供）
```

默认容差：

```text
ATOL=0.06
RTOL=0.01
```

这个容差用于动态模型与静态模型之间的 CUDA 数值差异；baseline vs flatten matmul 的同构图验证则更严格，并且当前结果为 0 误差。

## 4. 已修正 / 规避的验证问题

### 4.1 输入 dtype 不一致

早期导出的 ONNX / IREE graph 可能使用 `xi64` 输入，而后续 dynamic path / demote path 使用 `xi32` 输入。

问题表现：

```text
同一个模型变体运行命令里的 input dtype 不一致。
baseline 与 optimized 可能实际走了不同输入 ABI。
```

当前处理：

- `compile_onnx_iree_cuda.py` 默认使用 `--iree-input-demote-i64-to-i32`。
- 默认编译出的 VMFB 按 int32 输入运行；`run_dynamic_shape_pipeline.py` 默认给 benchmark 传 `--input-dtype i32`。
- 如果 ONNX 图输入必须保持 int64，可用 `--no-demote-i64-to-i32` 禁用 demote，并在运行时传 `--input-dtype i64`。
- correctness 脚本中显式写出输入 dtype，避免隐式猜测。

相关脚本：

```text
scripts/deepseek/compile_onnx_iree_cuda.py
scripts/deepseek/check_safe_dynamic_m_correctness.sh
scripts/deepseek/verify_flatten_matmul_correctness.py
```

### 4.2 static exact 的 last-token 输入不同

动态 last-token-only 模型通常只需要：

```text
input_ids
attention_mask
```

而某些 static exact VMFB 额外需要：

```text
last_token_indices: tensor<Bxi32>
```

如果没有传入该输入，static exact 对比会失败或比较语义不一致。

当前处理：

`check_safe_dynamic_m_correctness.sh` 中 static path 显式传入：

```bash
--input="${B}xi32=$((S - 1))"
```

这保证 static exact 与 dynamic last-token 输出语义一致。

### 4.3 输出 `.npy` dtype / raw byte view 处理

IREE 输出到 `--output=@file.npy` 后，读取时可能出现 `uint8` view 的情况。

问题表现：

```text
np.load 后 dtype 是 uint8
直接比较会得到错误 shape 或错误数值
```

当前处理：

`verify_flatten_matmul_correctness.py` 中：

```python
if arr.dtype == np.uint8:
    arr = arr.view(np.float32)
else:
    arr = arr.astype(np.float32, copy=False)
```

并在一维输出时按 batch reshape：

```python
if arr.ndim == 1 and arr.size % batch == 0:
    arr = arr.reshape(batch, arr.size // batch)
```

这避免了因为输出文件编码形式导致的误判。

### 4.4 输出 shape mismatch 不能继续比较

如果 baseline 与 optimized 输出 shape 不一致，继续算 diff 会产生误导。

当前处理：

`verify_flatten_matmul_correctness.py` 在比较前检查 shape：

```text
baseline.shape == optimized.shape
```

如果不一致，报告中记录 `shape mismatch`，而不是继续给出错误的数值差异。

### 4.5 attention context 不能简单用普通 matmul 判断正确性

projection / MLP / lm_head 的 safe dynamic-M 优化相对稳定，但 attention context 的 dynamic batch_matmul 更复杂。

当前处理策略：

```text
普通 dynamic-M matmul 使用严格 correctness 对比。
attention context candidate 需要同时经过 correctness 和 benchmark。
遇到 runtime timeout / lowering failure 时不把它当成通过结果。
```

这也是当前阶段保留 SIMT TileAndFuse fallback，而不把 dynamic rank-5 attention MMA 作为默认路径的原因。

## 5. 推荐验证顺序

### 5.1 验证 ONNX rewrite

```bash
cd scripts/deepseek
python3 verify_flatten_matmul_correctness.py \
  --baseline-module deepseek_r1_8b_onnx_iree_cuda_b8_s64_last_baseline.vmfb \
  --baseline-params seq64_b8_opt/build_common/deepseek_r1_8b_params.irpa \
  --optimized-module deepseek_r1_8b_onnx_iree_cuda_b8_s64_last_flatten_matmul.vmfb \
  --optimized-params seq64_b8_opt/build_flatten_matmul/deepseek_r1_8b_params.irpa \
  --batch 8 \
  --seq 64
```

### 5.2 验证 safe dynamic-M

```bash
cd scripts/deepseek
GPU=0 B=4 S=64 bash check_safe_dynamic_m_correctness.sh
```

### 5.3 验证新的 candidate dynamic VMFB

```bash
GPU=0 B=4 S=64 \
CANDIDATE_DYNAMIC_VMFB=/tmp/candidate.vmfb \
OUT_DIR=/tmp/iree_candidate_correctness_b4_s64 \
  bash check_safe_dynamic_m_correctness.sh
```

这样可以在同一个脚本中同时比较：

```text
safe_dynamic_m vs candidate_dynamic
candidate_dynamic vs static_exact
```

## 6. 当前结论

现有结果支持以下判断：

```text
flatten matmul rewrite 对保留的 b4_s32 / b8_s64 / b8_s128 结果没有改变 logits。
safe dynamic-M 的正确性验证需要同时看 old dynamic 与 static exact。
候选 attention context 优化必须先通过 correctness，再讨论性能收益。
```
