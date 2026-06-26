# Troubleshooting: Gemma Dynamic Export Issues

## 1. Old Transformers Cannot Load Gemma4

Observed error:

```text
ValueError: The checkpoint you are trying to load has model type `gemma4`
but Transformers does not recognize this architecture.
```

Cause:

```text
System transformers was too old.
```

Fix used:

```text
Use the newer transformers package located in:
/home/zhongjialin/projects/iree/Qwen/pydeps
```

The script:

```text
/home/zhongjialin/projects/iree/Gemma/run_gemma_dynamic_shape_experiment.py
```

adds that path before importing transformers.

## 2. PyTorch CUDA Not Visible During Export

Observed:

```text
torch.cuda.is_available() == False
```

Even with CUDA-capable GPUs available to IREE runtime.

Fix used:

```text
Export Gemma ONNX on CPU:
--export-device cpu
```

This only affects PyTorch ONNX export. IREE benchmark still runs with:

```text
--device=cuda
```

## 3. NumPy / onnxruntime Warning During Import Chain

Observed warning:

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
AttributeError: _ARRAY_API not found
```

In this project it appeared through an optional `onnxruntime` import path while
loading transformers/torch modules. It did not stop Gemma export in the final
successful run.

If it becomes fatal, avoid relying on onnxruntime in that Python path or align
NumPy/onnxruntime versions in a clean environment.

## 4. Dynamic ONNX Export Takes A Long Time

Gemma4 dynamic B/S export is slow. During the successful run, the process spent
substantial time in:

```text
PyTorch ONNX tracing
iree-import-onnx external parameter archiving
dense resource inlining
```

Useful signs of progress:

```bash
ls -lh /home/zhongjialin/projects/iree/Gemma/dynamic_experiments/gemma_e2b_dynamic_bs_fp16_cuda/gemma_dynamic_last_token.onnx
du -sh /home/zhongjialin/projects/iree/Gemma/dynamic_experiments/gemma_e2b_dynamic_bs_fp16_cuda/imported
```

The final `.irpa` can be around 10GB for Gemma E2B.

## 5. Recommended Gemma Command

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
