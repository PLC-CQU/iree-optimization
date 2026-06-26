# FlattenRank3Matmul Pass Implementation And Validation

## 1. Purpose

The pass rewrites Transformer linear-layer matmul from rank-3 activation form
to standard rank-2 matmul form:

```text
[B,S,K] x [K,N] -> [B,S,N]

becomes

collapse [B,S,K] -> [B*S,K]
linalg.matmul [B*S,K] x [K,N] -> [B*S,N]
expand [B*S,N] -> [B,S,N]
```

The main reason is to let IREE use its mature `linalg.matmul` lowering and CUDA
MMA/tensor-core path.

## 2. Source Files

Main implementation:

```text
/home/zhongjialin/projects/iree/compiler/src/iree/compiler/GlobalOptimization/FlattenRank3Matmul.cpp
```

Pass declaration:

```text
/home/zhongjialin/projects/iree/compiler/src/iree/compiler/GlobalOptimization/Passes.td
```

Pipeline registration:

```text
/home/zhongjialin/projects/iree/compiler/src/iree/compiler/GlobalOptimization/Passes.cpp
```

Build files:

```text
/home/zhongjialin/projects/iree/compiler/src/iree/compiler/GlobalOptimization/CMakeLists.txt
/home/zhongjialin/projects/iree/compiler/src/iree/compiler/GlobalOptimization/BUILD.bazel
```

Minimal test:

```text
/home/zhongjialin/projects/iree/compiler/src/iree/compiler/GlobalOptimization/test/flatten_rank3_matmul.mlir
```

Snapshots of these files are copied under:

```text
src/iree_pass/
```

## 3. Pattern 1: Rank3 Generic MatMul

The first pattern matches a `linalg.generic` equivalent to:

```text
lhs: [B,S,K]
rhs: [K,N]
out: [B,S,N]
```

Required indexing maps:

```text
lhs: (b, s, k)
rhs: (k, n)
out: (b, s, n)
```

Required iterators:

```text
parallel, parallel, parallel, reduction
```

Required body:

```text
acc + lhs * rhs
```

Dynamic dimensions are allowed. The pass uses `tensor.dim` and `arith.muli` to
materialize `B*S` when B or S is dynamic.

## 4. Pattern 2: Broadcasted BatchMatmul

The second pattern handles:

```text
rhs:  [K,N]
rhs3: broadcast(rhs) -> [B,K,N]
batch_matmul(lhs: [B,S,K], rhs3: [B,K,N])
```

This is semantically the same as `[B,S,K] x [K,N]`. The pass recovers the
original rank-2 RHS and emits the same collapse/matmul/expand structure.

## 5. Minimal Validation Command

```bash
/home/zhongjialin/projects/iree-build/tools/iree-opt \
  --split-input-file \
  --pass-pipeline='builtin.module(func.func(iree-global-opt-flatten-rank3-matmul))' \
  /home/zhongjialin/projects/iree/compiler/src/iree/compiler/GlobalOptimization/test/flatten_rank3_matmul.mlir
```

The output should contain `linalg.matmul` and `tensor.collapse_shape` /
`tensor.expand_shape`.

## 6. Full Pipeline Validation

The pass is integrated into global optimization after:

```text
FoldReshapesIntoTensorBarriers
```

and before:

```text
Canonicalize
CSE
```

Generate global opt IR:

```bash
/home/zhongjialin/projects/iree-build/tools/iree-compile \
  --iree-input-type=onnx \
  --iree-input-demote-i64-to-i32 \
  --iree-opt-strip-assertions \
  --compile-to=global-optimization \
  /path/to/model_external_inlined.mlir \
  -o /tmp/model_global_opt.mlir
```

Then count:

```text
linalg.matmul
linalg.batch_matmul
linalg.generic
tensor.collapse_shape
tensor.expand_shape
```

For Gemma E2B dynamic B/S, the observed change was:

```text
baseline:
  linalg.matmul:       0
  linalg.batch_matmul: 279

optimized:
  linalg.matmul:       277
  linalg.batch_matmul: 2
```

## 7. Important Boundary

This pass is for projection / MLP / lm_head style matmul:

```text
[B,S,K] x [K,N]
```

It should not be treated as a complete solution for attention context:

```text
[B,H,S,S] x [B,H,S,D] -> [B,H,S,D]
```

Attention context has dynamic reduction K when S is dynamic, and that needs a
separate optimization strategy.
