# Handoff Package Manifest

## Main Documentation

```text
docs/00_PROJECT_HANDOFF_GUIDE.md
  Main project handoff guide. Renamed from:
  /home/zhongjialin/projects/iree/project_handoff_technical_report.md

docs/01_STANDARD_COMPILE_RUN_FLOW.md
  Standard IREE compile/run/benchmark flow with commands.

docs/02_PASS_IMPLEMENTATION_AND_VALIDATION.md
  How the flatten-rank3-matmul pass works and how to validate it.

docs/03_CROSS_MODEL_GENERALIZATION.md
  Existing Qwen/Gemma fixed-shape generalization evidence.

docs/04_DEEPSEEK_DYNAMIC_SHAPE_OPTIMIZATION.md
  DeepSeek dynamic shape optimization stage report.

docs/05_STATIC_BUCKET_DEPLOYMENT_AND_CORRECTNESS.md
  Static bucket deployment and correctness debugging report.

docs/06_GEMMA_DYNAMIC_GENERALIZATION.md
  Gemma E2B dynamic B/S validation report.
```

## Troubleshooting Notes

```text
docs/troubleshooting/01_standard_compile_run_issues.md
  iree-import-onnx, iree-compile, iree-run-module, CUDA target, params, and
  benchmark issues encountered during standard compile/run.

docs/troubleshooting/02_correctness_issues.md
  Correctness issues encountered during static bucket and dynamic-vs-static
  verification.

docs/troubleshooting/03_dynamic_shape_attention_issues.md
  Dynamic attention-context experiments, timeout/hang, and current boundary.

docs/troubleshooting/04_gemma_dynamic_export_issues.md
  Gemma4 dynamic ONNX export environment problems and solutions.
```

## Source Snapshots

```text
src/iree_pass/FlattenRank3Matmul.cpp
src/iree_pass/Passes.td
src/iree_pass/Passes.cpp
src/iree_pass/CMakeLists.txt
src/iree_pass/BUILD.bazel
src/iree_pass/flatten_rank3_matmul.mlir

src/codegen_dynamic/ConfigUtils.cpp
src/codegen_dynamic/KernelConfig.cpp
src/codegen_dynamic/BlockDynamicDimensions.cpp

src/gemma/run_gemma_dynamic_shape_experiment.py
src/gemma/run_gemma_standard_iree.py
src/gemma/run_gemma_flatten_compare.py
```

## Scripts

```text
scripts/deepseek/benchmark_safe_dynamic_m.py
scripts/deepseek/check_safe_dynamic_m_correctness.py
scripts/deepseek/benchmark_bounded_dynamic_shape.py
scripts/deepseek/rewrite_onnx_flatten_matmul.py
scripts/deepseek/inline_onnx_dense_resources.py
scripts/deepseek/iree_bucket_router.py
scripts/deepseek/bucket_manifest.example.json
scripts/dynamic_shape_experiments/summarize_googlebench.py
scripts/dynamic_shape_experiments/shape_specializing_compile_cache.py
```

No shell-wrapper files are included in this handoff package. Former shell
wrappers were either replaced by Python scripts above or converted into
explicit commands in the documentation.

## Result Summaries

```text
results/deepseek/
results/gemma/
results/cross_model/
```

Only lightweight JSON summaries are included. Large generated artifacts remain
in the original experiment directories.
