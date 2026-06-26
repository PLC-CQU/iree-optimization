# Results Directory

This directory contains lightweight result summaries only.

Large generated artifacts are intentionally excluded:

```text
.onnx
.vmfb
.irpa
.safetensors
large MLIR dumps
```

Those files remain in the original experiment directories referenced by:

```text
docs/00_PROJECT_HANDOFF_GUIDE.md
docs/01_STANDARD_COMPILE_RUN_FLOW.md
docs/06_GEMMA_DYNAMIC_GENERALIZATION.md
```

## Included Result Types

```text
deepseek/
  JSON summaries from static flatten, dynamic-vs-bucket, correctness, and IR
  diagnostic experiments.

gemma/
  Gemma E2B dynamic B/S summary and static flatten summary.

cross_model/
  Reserved for future Qwen/Gemma/Llama cross-model summaries.
```
