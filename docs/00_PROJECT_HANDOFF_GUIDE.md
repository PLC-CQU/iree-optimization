# IREE LLM MatMul Flatten 优化项目交接技术报告

日期：2026-06-27

## 1. 项目一句话概括

本项目围绕 IREE 编译大语言模型时的 GPU 性能问题展开，核心发现是：

```text
Transformer 中大量线性层在 IR 里表现为：
  [B, S, K] x [K, N] -> [B, S, N]

如果显式改写为：
  [B*S, K] x [K, N] -> [B*S, N] -> [B, S, N]

IREE 更容易将其 lowering 到标准二维 matmul 路径，从而更稳定地使用 CUDA MMA / tensor core。
```

项目从静态 shape 模型开始验证优化收益，然后将优化写入 IREE pass，最后扩展到动态 B/S 模型，并在 DeepSeek、Gemma、Qwen 等模型上验证其泛化能力。

当前最重要的结论是：

```text
rank3 activation x rank2 weight 的 flatten MatMul 优化不是 DeepSeek 专用。

它对 Transformer projection / MLP / lm_head 这类结构具有通用意义。
动态 shape 下，只要动态主要集中在 M = B*S，而 K/N 是静态权重维度，优化仍然可以明显生效。
```

## 2. 仓库和重要路径

主要工程路径：

```text
/home/zhongjialin/projects/iree
/home/zhongjialin/projects/iree-build
/home/zhongjialin/projects/iree-optimization
```

IREE 工具路径：

```text
/home/zhongjialin/projects/iree-build/tools/iree-compile
/home/zhongjialin/projects/iree-build/tools/iree-opt
/home/zhongjialin/projects/iree-build/tools/iree-run-module
/home/zhongjialin/projects/iree-build/tools/iree-benchmark-module
```

Python / pip 版 IREE 工具：

```text
/home/zhongjialin/projects/.venv/bin/iree-compile
/home/zhongjialin/projects/.venv/bin/iree-import-onnx
```

当前源码 build 里的 IREE 带有本项目新增 pass。`.venv` 里的 `iree-compile` 可作为旧版本 / no-pass baseline 使用，但注意它和当前源码 build 不是完全同一版本。

核心 pass 文件：

```text
/home/zhongjialin/projects/iree/compiler/src/iree/compiler/GlobalOptimization/FlattenRank3Matmul.cpp
/home/zhongjialin/projects/iree/compiler/src/iree/compiler/GlobalOptimization/Passes.td
/home/zhongjialin/projects/iree/compiler/src/iree/compiler/GlobalOptimization/Passes.cpp
/home/zhongjialin/projects/iree/compiler/src/iree/compiler/GlobalOptimization/test/flatten_rank3_matmul.mlir
```

动态 shape / codegen 相关改动：

```text
/home/zhongjialin/projects/iree/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp
/home/zhongjialin/projects/iree/compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp
/home/zhongjialin/projects/iree/compiler/src/iree/compiler/Codegen/Common/BlockDynamicDimensions.cpp
```

DeepSeek 相关实验：

```text
/home/zhongjialin/projects/iree/deepseek-R1-Llama-8b
/home/zhongjialin/projects/iree_llm_matmul_flatten_handoff/scripts/deepseek/benchmark_safe_dynamic_m.py
/home/zhongjialin/projects/iree_llm_matmul_flatten_handoff/scripts/deepseek/check_safe_dynamic_m_correctness.py
/home/zhongjialin/projects/iree/deepseek-R1-Llama-8b/iree_dynamic_shape_optimization_stage_report.md
/home/zhongjialin/projects/iree/deepseek-R1-Llama-8b/iree_static_bucket_optimization_report.md
```

Gemma 相关实验：

```text
/home/zhongjialin/projects/iree/Gemma
/home/zhongjialin/projects/iree/Gemma/run_gemma_dynamic_shape_experiment.py
/home/zhongjialin/projects/iree/Gemma/gemma_dynamic_shape_flatten_report.md
/home/zhongjialin/projects/iree/Gemma/dynamic_experiments/gemma_e2b_dynamic_bs_fp16_cuda
```

跨模型总结：

```text
/home/zhongjialin/projects/iree/other_models_flatten_generalization_report.md
```

单算子 / 动态 shape 分析实验：

```text
/home/zhongjialin/projects/iree/dynamic_shape_matmul_experiment
```

## 3. IREE 基础编译流程说明

IREE 的完整模型部署流程可以粗略理解为：

```text
PyTorch / HuggingFace
  -> ONNX
  -> iree-import-onnx
  -> IREE input/global optimization MLIR
  -> Flow / Stream / HAL / Codegen
  -> VMFB
  -> iree-run-module / iree-benchmark-module
```

项目中最常用的几个产物：

```text
.onnx
  模型导出的 ONNX 文件。

*_external.mlir
  iree-import-onnx 导入后的 MLIR，通常带 external parameter 引用。

*_external_inlined.mlir
  将 dense_resource 引用改写后的 MLIR，便于 iree-compile 处理大参数。

.irpa
  IREE 参数归档。大模型权重通常不直接塞进 VMFB，而是通过 --parameters=model=xxx.irpa 加载。

.vmfb
  IREE 编译出的可执行模型文件。

.googlebench.json
  iree-benchmark-module 输出的 benchmark 结果。
```

## 4. 标准编译运行流程

下面是一套标准流程，以已有 ONNX 为输入。

### 4.1 ONNX 导入为 IREE MLIR

```bash
/home/zhongjialin/projects/.venv/bin/iree-import-onnx \
  /path/to/model.onnx \
  --large-model \
  --externalize-params \
  --num-elements-threshold 2 \
  --param-gb-threshold 2 \
  --save-params-to /path/to/model_params.irpa \
  -o /path/to/model_external.mlir
```

参数说明：

```text
--large-model
  大模型导入模式。

--externalize-params
  将大权重外部化，不直接写入 MLIR。

--save-params-to
  保存外部参数到 .irpa 文件。
```

### 4.2 inline dense resources

项目中常用脚本：

```text
/home/zhongjialin/projects/iree-optimization/scripts/deepseek/inline_onnx_dense_resources.py
/home/zhongjialin/projects/iree/deepseek-R1-Llama-8b/inline_onnx_dense_resources.py
```

命令：

```bash
python3 /home/zhongjialin/projects/iree-optimization/scripts/deepseek/inline_onnx_dense_resources.py \
  /path/to/model_external.mlir \
  -o /path/to/model_external_inlined.mlir
```

### 4.3 编译为 CUDA VMFB

```bash
/home/zhongjialin/projects/iree-build/tools/iree-compile \
  --iree-input-type=onnx \
  --iree-input-demote-i64-to-i32 \
  --iree-opt-strip-assertions \
  --iree-hal-target-backends=cuda \
  --iree-cuda-target=sm_86 \
  --iree-codegen-llvmgpu-use-reduction-vector-distribution=false \
  /path/to/model_external_inlined.mlir \
  -o /path/to/model.vmfb
```

注意：

```text
RTX 4090 实测 sm_86 可以运行。
之前 sm_89 曾出现目标架构问题，因此当前实验多数使用 sm_86。
```

### 4.4 运行模型

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/zhongjialin/projects/iree-build/tools/iree-run-module \
  --module=/path/to/model.vmfb \
  --parameters=model=/path/to/model_params.irpa \
  --parameter_mode=file \
  --device=cuda \
  --function=main_graph \
  --input=1x32xi64=0 \
  --input=1x32xi64=1
```

若模型输入已经 demote 到 i32，则输入改为：

```bash
--input=1x32xi32=0
--input=1x32xi32=1
```

当前 DeepSeek/Gemma 实验中，多数命令仍使用：

```bash
--input=BxSxi64=0
--input=BxSxi64=1
```

因为编译时使用了：

```text
--iree-input-demote-i64-to-i32
```

### 4.5 benchmark

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/zhongjialin/projects/iree-build/tools/iree-benchmark-module \
  --module=/path/to/model.vmfb \
  --parameters=model=/path/to/model_params.irpa \
  --parameter_mode=file \
  --device=cuda \
  --function=main_graph \
  --input=1x32xi64=0 \
  --input=1x32xi64=1 \
  --benchmark_repetitions=5 \
  --benchmark_min_time=1x \
  --benchmark_min_warmup_time=0 \
  --benchmark_time_unit=ms \
  --benchmark_out=/tmp/result.googlebench.json \
  --benchmark_out_format=json
```

如果 benchmark 长时间没有输出，需要先确认：

```bash
nvidia-smi
```

看 GPU 是否 100% 占用，是否有旧的 `iree-benchmark-module` 进程没退出。

### 4.6 编译到 global-optimization 阶段查看 IR

这是分析 pass 是否生效最常用的方法：

```bash
/home/zhongjialin/projects/iree-build/tools/iree-compile \
  --iree-input-type=onnx \
  --iree-input-demote-i64-to-i32 \
  --iree-opt-strip-assertions \
  --compile-to=global-optimization \
  /path/to/model_external_inlined.mlir \
  -o /tmp/model_global_opt.mlir
```

统计关键 op：

```bash
python3 - <<'PY'
from pathlib import Path
import re

path = Path("/tmp/model_global_opt.mlir")
text = path.read_text(errors="replace")
for name in [
    "linalg.matmul",
    "linalg.batch_matmul",
    "linalg.generic",
    "tensor.collapse_shape",
    "tensor.expand_shape",
]:
    print(name, len(re.findall(r"\b" + re.escape(name) + r"\b", text)))
PY
```

## 5. 优化方法基础解释

### 5.1 原始问题

Transformer 线性层通常写作：

```text
hidden_states: [B, S, K]
weight:        [K, N]
output:        [B, S, N]
```

数学上：

```text
output[b, s, n] = sum_k hidden_states[b, s, k] * weight[k, n]
```

这个操作本质上可以看成对 `B*S` 个 token 同时做同一个矩阵乘：

```text
[B*S, K] x [K, N] -> [B*S, N]
```

### 5.2 为什么改成二维 matmul 会快

CUDA MMA / tensor core 的硬件指令本质是二维 tile 的矩阵乘。三维 / batch matmul 也可以走 MMA，但前提是编译器能把它稳定拆成标准二维 tile。

对 IREE 来说，标准 `linalg.matmul` 是最成熟的入口。把 rank3 matmul 改写为 rank2 matmul 后：

```text
原始：
  linalg.generic 或 linalg.batch_matmul

优化后：
  tensor.collapse_shape
  linalg.matmul
  tensor.expand_shape
```

后端更容易选择高质量的 tiling、padding、MMA lowering。

这不是改变数学语义，而是改变 IR 形态，让后端更容易看懂。

### 5.3 静态和动态 shape 的区别

静态 shape：

```text
B、S、K、N 都是编译期常量。
IREE 可以直接根据完整 shape 选择最合适的 tile / workgroup / MMA config。
```

动态 shape：

```text
B、S 是运行时值。
M = B*S 也是运行时 SSA value。
```

动态 shape 不代表不能用 MMA。关键是动态维度出现在哪里：

```text
容易优化：
  [?, K_static] x [K_static, N_static]
  只有 M 动态，K/N 静态。

困难：
  attention context 中 S 同时影响 M 和 reduction K。
  dynamic K 会让 MMA tile / padding / pipeline 选择困难很多。
```

当前项目最稳定有效的方向是：

```text
只把 projection / MLP / lm_head 中的 rank3 x rank2 matmul flatten 成 dynamic-M rank2 matmul。
暂时不要把 attention context dynamic batch_matmul 混进同一个结论里。
```

## 6. Pass 实现说明

### 6.1 Pass 名称

```text
iree-global-opt-flatten-rank3-matmul
```

定义位置：

```text
compiler/src/iree/compiler/GlobalOptimization/Passes.td
```

实现位置：

```text
compiler/src/iree/compiler/GlobalOptimization/FlattenRank3Matmul.cpp
```

接入 pipeline 位置：

```text
compiler/src/iree/compiler/GlobalOptimization/Passes.cpp
```

当前接入点在：

```text
FoldReshapesIntoTensorBarriers
-> FlattenRank3Matmul
-> Canonicalize
-> CSE
```

### 6.2 匹配 rank3 x rank2 matmul

匹配对象：

```text
lhs rank = 3:    [B, S, K]
rhs rank = 2:    [K, N]
out rank = 3:    [B, S, N]
```

要求 affine maps 符合：

```text
lhs: (b, s, k)
rhs: (k, n)
out: (b, s, n)
```

要求 iterator 类型：

```text
parallel, parallel, parallel, reduction
```

要求 body 是乘加归约：

```text
acc + lhs * rhs
```

改写逻辑：

```text
bSize = dim(lhs, 0)
sSize = dim(lhs, 1)
bsSize = bSize * sSize

flatLhs = collapse_shape(lhs, [[0, 1], [2]])
flatOut = collapse_shape(out, [[0, 1], [2]])
matmul = linalg.matmul(flatLhs, rhs, flatOut)
expanded = expand_shape(matmul, [[0, 1], [2]])
```

动态 shape 下，`B*S` 用 SSA index arithmetic 表示：

```text
%b = tensor.dim %lhs, %c0
%s = tensor.dim %lhs, %c1
%bs = arith.muli %b, %s
```

### 6.3 匹配 broadcasted batch_matmul

有些导入路径会出现：

```text
rhs:  [K, N]
rhs3: broadcast(rhs) -> [B, K, N]
batch_matmul(lhs: [B, S, K], rhs3: [B, K, N])
```

这个 batch 维只是广播出来的，不是真正每个 batch 一份不同权重。pass 会识别 broadcast source，把它还原成 rank2 weight，然后改写为标准 matmul。

### 6.4 最小测试

运行 pass lit 风格测试：

```bash
/home/zhongjialin/projects/iree-build/tools/iree-opt \
  --split-input-file \
  --pass-pipeline='builtin.module(func.func(iree-global-opt-flatten-rank3-matmul))' \
  /home/zhongjialin/projects/iree/compiler/src/iree/compiler/GlobalOptimization/test/flatten_rank3_matmul.mlir
```

如果出现：

```text
'iree-global-opt-flatten-rank3-matmul' does not refer to a registered pass
```

说明 pass 没有被正确注册，优先检查：

```text
Passes.td
Passes.cpp
CMakeLists.txt
BUILD.bazel
```

## 7. 静态 shape 阶段

### 7.1 目标

先在静态 shape 下验证优化是否有效。静态 shape 的好处是：

```text
B/S/K/N 都是编译期常量。
如果 flatten 后性能提升明显，说明 IR 形态确实影响 IREE 后端 codegen。
```

### 7.2 DeepSeek 静态结果

DeepSeek 静态 flatten 后，在 b4_s32 等典型 shape 上达到约 `37 ms` 量级。

之前对比过：

```text
auto flatten VMFB:
  mean around 37.3 ms

existing flatten VMFB:
  mean around 37.3 ms
```

这证明 pass 接入 IREE 后，直接编译原始模型也能自动得到已有 flatten rewrite 的效果。

### 7.3 Qwen / Gemma 静态泛化

静态 fixed-shape 实验显示：

```text
Qwen2.5-3B:
  rewritten weight MatMul: 253
  speedup: 2.045x 到 5.954x

Gemma E4B:
  rewritten weight MatMul: 344
  speedup: 1.754x 到 3.265x
```

这说明优化不是 DeepSeek 特例。

详细报告：

```text
/home/zhongjialin/projects/iree/other_models_flatten_generalization_report.md
```

## 8. Bucket 部署阶段和 correctness 问题

中间阶段曾探索过 bucket 部署：

```text
真实输入动态 B/S
-> runtime 选择合适 bucket
-> pad 到静态 bucket shape
-> 调用静态 VMFB
-> 裁剪输出
```

该方向从部署上是可行的，并且 full grid correctness 通过：

```text
total: 11
passed: 11
failed: 0
max_abs: 0.04296875
atol=0.06, rtol=0.01
```

遇到的重要问题：

```text
fixed static ONNX 中：
Constant [1,S] -> Expand [B,S] -> Cast 的 no-input position_ids 路径
在 IREE CUDA 下存在 correctness / layout 问题。
```

解决方法：

```text
static bucket 导出时不要把 RoPE position_ids expand 到 batch。
保持 [1,S] / [1,S,D]，让后续 q/k elementwise 自然 broadcast 到 batch。
```

后来项目重点从 bucket 部署切回动态 shape 编译优化。bucket 方向可以作为工程部署备选，但不是当前主线。

## 9. 动态 shape 阶段

### 9.1 目标

目标是：

```text
不依赖多个静态 bucket VMFB。
使用一个动态 B/S VMFB。
在编译器内部识别并优化可 flatten 的 rank3 matmul。
让动态模型性能尽量接近静态模型。
```

### 9.2 单算子逐级定位

为了定位动态模型慢在哪里，做过逐级实验：

```text
实验 1：single matmul
  [B,S,K] x [K,N]

实验 2：三个 projection
  q/k/v 三个 matmul 共用一个 flatten lhs

实验 3：projection + reshape + transpose

实验 4：attention scores / context batch_matmul

实验 5：MLP gate/up/down 三个 matmul
```

实验目录：

```text
/home/zhongjialin/projects/iree/dynamic_shape_matmul_experiment
```

阶段性判断：

```text
projection / MLP:
  适合 flatten 为 dynamic-M rank2 matmul。

attention scores/context:
  更复杂，尤其是 dynamic reduction K 和 batch_matmul lowering。
```

### 9.3 safe dynamic-M 优化

DeepSeek 上的关键结果：

```text
b1_s32:
old_dynamic_ms:        3364.666
safe_dynamic_m_ms:       82.213
static_exact_ms:         37.309
speedup:                40.926x
safe_vs_static:          2.204x
```

正确性：

```text
old_dynamic vs safe_dynamic_m:
  max_abs: 0.0078125
  allclose(atol=0.06, rtol=0.01): True

safe_dynamic_m vs static_exact:
  max_abs: 0.0078125
  allclose(atol=0.06, rtol=0.01): True
```

更大 shape 上也验证过：

```text
b1_s64
b4_s64
b8_s128
```

结论：

```text
dynamic-M matmul 是当前动态 shape 下最稳定、最有效的优化方向。
```

## 10. Gemma 动态 shape 泛化验证

为验证动态优化不是 DeepSeek 专用，新增 Gemma 动态实验脚本：

```text
/home/zhongjialin/projects/iree/Gemma/run_gemma_dynamic_shape_experiment.py
```

动态 ONNX 输入：

```text
input_ids:      tensor<?x?xi64>
attention_mask: tensor<?x?xi64>
output logits:  tensor<?x262144xf32>
```

结果：

| Shape | baseline dynamic | optimized dynamic | speedup | correctness |
| --- | ---: | ---: | ---: | --- |
| b1_s32 | 136.603 ms | 25.102 ms | 5.442x | pass |
| b1_s64 | 198.063 ms | 35.870 ms | 5.522x | pass |
| b4_s32 | 396.723 ms | 40.125 ms | 9.887x | pass |

IR 证据：

| IR | `linalg.matmul` | `linalg.batch_matmul` |
| --- | ---: | ---: |
| baseline global opt | 0 | 279 |
| optimized global opt | 277 | 2 |

这说明：

```text
Gemma 动态模型中的 rank3 / batch-like matmul 大量被改写为标准 linalg.matmul。
优化后性能提升 5x 到 10x，正确性通过。
```

详细报告：

```text
/home/zhongjialin/projects/iree/Gemma/gemma_dynamic_shape_flatten_report.md
```

## 11. 复现 Gemma 动态实验

编译两个动态版本：

```bash
cd /home/zhongjialin/projects/iree/Gemma

/home/zhongjialin/projects/iree/Gemma/run_gemma_dynamic_shape_experiment.py \
  --model-path /home/zhongjialin/projects/iree/Gemma/googlegemma-4-E2B-it \
  --experiment-name gemma_e2b_dynamic_bs_fp16_cuda \
  --export-device cpu \
  --gpu 2 \
  --shapes b1_s32 \
  --repetitions 1 \
  --min-time 1x \
  --warmup-time 0 \
  --benchmark-timeout-seconds 120 \
  --action compile
```

跑 b1_s32：

```bash
cd /home/zhongjialin/projects/iree/Gemma

/home/zhongjialin/projects/iree/Gemma/run_gemma_dynamic_shape_experiment.py \
  --model-path /home/zhongjialin/projects/iree/Gemma/googlegemma-4-E2B-it \
  --experiment-name gemma_e2b_dynamic_bs_fp16_cuda \
  --export-device cpu \
  --gpu 2 \
  --shapes b1_s32 \
  --repetitions 3 \
  --min-time 1x \
  --warmup-time 0 \
  --benchmark-timeout-seconds 180 \
  --action all
```

跑更多动态 shape：

```bash
cd /home/zhongjialin/projects/iree/Gemma

/home/zhongjialin/projects/iree/Gemma/run_gemma_dynamic_shape_experiment.py \
  --model-path /home/zhongjialin/projects/iree/Gemma/googlegemma-4-E2B-it \
  --experiment-name gemma_e2b_dynamic_bs_fp16_cuda \
  --export-device cpu \
  --gpu 2 \
  --shapes b1_s64 b4_s32 \
  --repetitions 3 \
  --min-time 1x \
  --warmup-time 0 \
  --benchmark-timeout-seconds 180 \
  --action all
```

查看汇总：

```bash
cat /home/zhongjialin/projects/iree/Gemma/dynamic_experiments/gemma_e2b_dynamic_bs_fp16_cuda/dynamic_shape_full_summary.json
```

## 12. 生成 IR 对比

baseline 使用 `.venv` 里的 no-pass IREE：

```bash
/home/zhongjialin/projects/.venv/bin/iree-compile \
  --iree-input-type=onnx \
  --iree-input-demote-i64-to-i32 \
  --iree-opt-strip-assertions \
  --compile-to=global-optimization \
  /home/zhongjialin/projects/iree/Gemma/dynamic_experiments/gemma_e2b_dynamic_bs_fp16_cuda/imported/gemma_dynamic_external_inlined.mlir \
  -o /tmp/gemma_e2b_dynamic_baseline_global.mlir
```

optimized 使用当前源码 build：

```bash
/home/zhongjialin/projects/iree-build/tools/iree-compile \
  --iree-input-type=onnx \
  --iree-input-demote-i64-to-i32 \
  --iree-opt-strip-assertions \
  --compile-to=global-optimization \
  /home/zhongjialin/projects/iree/Gemma/dynamic_experiments/gemma_e2b_dynamic_bs_fp16_cuda/imported/gemma_dynamic_external_inlined.mlir \
  -o /tmp/gemma_e2b_dynamic_optimized_global.mlir
```

统计：

```bash
python3 - <<'PY'
from pathlib import Path
import re

for label, path in [
    ("baseline", "/tmp/gemma_e2b_dynamic_baseline_global.mlir"),
    ("optimized", "/tmp/gemma_e2b_dynamic_optimized_global.mlir"),
]:
    text = Path(path).read_text(errors="replace")
    print(label)
    print("  linalg.matmul:", len(re.findall(r"\blinalg\.matmul\b", text)))
    print("  linalg.batch_matmul:", len(re.findall(r"\blinalg\.batch_matmul\b", text)))
    print("  linalg.generic:", len(re.findall(r"\blinalg\.generic\b", text)))
    print("  tensor.collapse_shape:", len(re.findall(r"\btensor\.collapse_shape\b", text)))
    print("  tensor.expand_shape:", len(re.findall(r"\btensor\.expand_shape\b", text)))
PY
```

预期：

```text
baseline:
  linalg.matmul: 0
  linalg.batch_matmul: 279

optimized:
  linalg.matmul: 277
  linalg.batch_matmul: 2
```

## 13. 常见问题和解决方法

### 13.1 pass 未注册

报错：

```text
'iree-global-opt-flatten-rank3-matmul' does not refer to a registered pass
```

检查：

```text
Passes.td 是否定义 pass
Passes.cpp 是否注册 pass
CMakeLists.txt / BUILD.bazel 是否加入 cpp 文件
是否重新编译 iree-opt / iree-compile
```

### 13.2 rg 不存在

服务器可能没有 `rg`：

```text
Command 'rg' not found
```

可以用：

```bash
grep -E "linalg\\.(batch_)?matmul" file.mlir
```

或者用 Python 正则统计。

### 13.3 CUDA invalid context

曾遇到：

```text
CUDA_ERROR_INVALID_CONTEXT
```

常见原因：

```text
编译目标架构不匹配，例如 sm_89 问题。
运行时 device / CUDA_VISIBLE_DEVICES 不一致。
多个 GPU 进程上下文混乱。
```

当前实验一般使用：

```text
--iree-cuda-target=sm_86
CUDA_VISIBLE_DEVICES=<gpu>
```

### 13.4 CUDA no device

在 Codex sandbox 内跑 benchmark 可能出现：

```text
CUDA_ERROR_NO_DEVICE
```

这是 sandbox 看不到 GPU，不是模型错误。用户自己在服务器 shell 里运行同样命令即可，或在工具里申请非 sandbox GPU 运行。

### 13.5 benchmark 长时间无结果

先看：

```bash
nvidia-smi
```

如果 GPU 100% 且进程存在，可能是真的 kernel 很慢或 hang。之前 attention context 的某些 dynamic MMA 尝试会 timeout。建议用：

```bash
timeout 60s <command>
```

### 13.6 Gemma 动态导出慢

Gemma4 动态 ONNX export 会比较慢，并且 Python 环境中系统 transformers 太旧，不认识 `model_type=gemma4`。

解决方法：

```text
脚本会优先使用 /home/zhongjialin/projects/iree/Qwen/pydeps 里的 transformers。
导出使用 --export-device cpu。
IREE benchmark 仍然使用 CUDA。
```

### 13.7 position_ids correctness 问题

静态 bucket 中曾遇到 RoPE position_ids 展开到 batch 后出现 correctness 问题。

修法：

```text
不要把 Constant [1,S] expand 到 [B,S]。
保持 [1,S] / [1,S,D]，让 elementwise 自然 broadcast。
```

## 14. 当前局限性

### 14.1 attention context 仍未完全解决

当前最稳定的优化对象是：

```text
projection / MLP / lm_head
[B,S,K] x [K,N]
```

attention context 形如：

```text
[B,H,S,S] x [B,H,S,D] -> [B,H,S,D]
```

这里动态 `S` 同时影响 M 和 reduction K。dynamic reduction K 对 MMA lowering 很不友好，之前强行优化可能出现 timeout / hang。

因此，attention context 应作为下一阶段单独研究主题。

### 14.2 Gemma A/B 不是完全单变量

Gemma 动态实验中：

```text
baseline_dynamic 使用 .venv iree-compile
optimized_dynamic 使用 iree-build/tools/iree-compile
```

这能说明优化方向有效，但严格论文实验最好给当前 IREE 加一个开关：

```text
--iree-global-opt-disable-flatten-rank3-matmul
```

然后用同一份 compiler 做 pass on/off 对比。

### 14.3 动态 shape 覆盖范围仍需扩大

目前已验证典型点：

```text
DeepSeek: b1_s32, b1_s64, b4_s64, b8_s128 等
Gemma: b1_s32, b1_s64, b4_s32
```

建议扩展到：

```text
b1/b2/b4/b8
s16/s32/s64/s128/s256
```

### 14.4 还缺少更底层的 MMA 证据

目前证据主要是：

```text
性能提升
global IR 中 batch_matmul -> matmul
正确性 allclose
```

还可以进一步补：

```text
LLVMGPU IR
PTX 中 mma.sync / tensor core 相关指令
dispatch-level profiling
```

## 15. 下一步建议

### 15.1 做严格 pass on/off A/B

新增一个全局开关：

```text
--iree-global-opt-disable-flatten-rank3-matmul
```

目标：

```text
同一份 iree-build/tools/iree-compile
pass off -> baseline
pass on  -> optimized
```

这样可以排除不同 IREE 版本造成的干扰。

### 15.2 补 MMA / PTX 证据

研究问题：

```text
optimized 后是否确实进入 MMA lowering？
具体使用了哪些 MMA intrinsic？
哪些 dispatch 仍然没有走 MMA？
```

建议输出：

```text
global opt IR
dispatch IR
LLVMGPU IR
PTX / cubin 反汇编
```

### 15.3 attention context 单独研究

不要把 attention context 和 projection / MLP flatten 混在一起。

建议从小 IR 开始：

```text
dynamic attention scores
dynamic attention context
dynamic reduction K
padding to tile size
SIMT fallback
guarded specialization
```

已有实验目录：

```text
/home/zhongjialin/projects/iree/dynamic_shape_matmul_experiment
```

### 15.4 扩展更多模型

建议继续验证：

```text
Qwen dynamic B/S
Gemma E4B dynamic B/S
Llama / Mistral 类模型
不同 hidden size / intermediate size / head dim
```

目标是把结论写成：

```text
该 pass 对 decoder-only Transformer 的 projection / MLP rank3 x rank2 MatMul 具有普适优化效果。
```

## 16. 新同学接手建议阅读顺序

建议按下面顺序读：

```text
1. 本文档
2. compiler/src/iree/compiler/GlobalOptimization/FlattenRank3Matmul.cpp
3. compiler/src/iree/compiler/GlobalOptimization/test/flatten_rank3_matmul.mlir
4. deepseek-R1-Llama-8b/iree_static_bucket_optimization_report.md
5. deepseek-R1-Llama-8b/iree_dynamic_shape_optimization_stage_report.md
6. Gemma/gemma_dynamic_shape_flatten_report.md
7. other_models_flatten_generalization_report.md
8. dynamic_shape_matmul_experiment 里的单算子脚本
```

然后建议先复现两个最小实验：

```text
1. iree-opt 跑 flatten_rank3_matmul.mlir
2. Gemma E2B dynamic b1_s32 benchmark
```

如果这两个都跑通，新同学基本就掌握了：

```text
pass 如何工作
模型如何编译运行
如何判断优化是否生效
如何做 correctness 和 benchmark
```

## 17. 当前项目结论

当前项目已经形成比较完整的闭环：

```text
从模型编译运行
到静态 shape 优化
到 IREE pass 实现
到动态 shape 性能修复
到跨模型验证
```

最重要的技术结论是：

```text
将 Transformer 中的 rank3 activation x rank2 weight MatMul
显式 flatten 成标准 rank2 linalg.matmul，
可以显著改善 IREE CUDA 后端对这些线性层的 lowering。

动态 shape 下，只要动态主要在 M = B*S，K/N 保持静态，
该优化仍然可以稳定生效，并明显缩小动态模型和静态模型之间的性能差距。
```

当前最值得继续做的是：

```text
严格 pass on/off A/B
补 MMA/PTX 证据
扩大动态 shape 和模型覆盖
单独研究 attention context dynamic-K 问题
```
