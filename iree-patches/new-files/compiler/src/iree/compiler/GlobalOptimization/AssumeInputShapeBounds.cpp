// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_ASSUMEINPUTSHAPEBOUNDSPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

struct BoundedTensorValue {
  Value value;
  SmallVector<Operation *, 5> internalOps;
};

static BoundedTensorValue assumeRank2TensorBounds(Value tensorValue,
                                                 int64_t maxBatch,
                                                 int64_t maxSeq,
                                                 OpBuilder &builder,
                                                 Location loc) {
  auto tensorType = dyn_cast<RankedTensorType>(tensorValue.getType());
  if (!tensorType || tensorType.getRank() != 2 ||
      (!tensorType.isDynamicDim(0) && !tensorType.isDynamicDim(1))) {
    return {};
  }

  MLIRContext *context = builder.getContext();
  auto batchDimOp = tensor::DimOp::create(builder, loc, tensorValue, 0);
  auto seqDimOp = tensor::DimOp::create(builder, loc, tensorValue, 1);
  auto batchAssumption = IREE::Util::IntAssumptionAttr::get(
      context, /*umin=*/1, /*umax=*/maxBatch, /*udiv=*/std::nullopt);
  auto seqAssumption = IREE::Util::IntAssumptionAttr::get(
      context, /*umin=*/1, /*umax=*/maxSeq, /*udiv=*/std::nullopt);
  auto boundedBatchOp = IREE::Util::AssumeIntOp::create(
      builder, loc, batchDimOp.getResult(), batchAssumption);
  auto boundedSeqOp = IREE::Util::AssumeIntOp::create(
      builder, loc, seqDimOp.getResult(), seqAssumption);

  SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(0),
                                       builder.getIndexAttr(0)};
  SmallVector<OpFoldResult> sizes = {boundedBatchOp.getResult(0),
                                     boundedSeqOp.getResult(0)};
  SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1),
                                       builder.getIndexAttr(1)};
  auto sliceOp = tensor::ExtractSliceOp::create(builder, loc, tensorType,
                                                tensorValue, offsets, sizes,
                                                strides);
  return {sliceOp.getResult(),
          {batchDimOp.getOperation(), seqDimOp.getOperation(),
           boundedBatchOp.getOperation(), boundedSeqOp.getOperation(),
           sliceOp.getOperation()}};
}

static void replaceUsesWithBoundedTensor(Value tensorValue,
                                         BoundedTensorValue boundedValue) {
  tensorValue.replaceUsesWithIf(boundedValue.value, [&](OpOperand &use) {
    Operation *owner = use.getOwner();
    return !llvm::is_contained(boundedValue.internalOps, owner);
  });
}

static bool isFlattenedAttentionContext(linalg::BatchMatmulOp op) {
  auto lhsType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
  auto rhsType = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!lhsType || !rhsType || !resultType || lhsType.getRank() != 3 ||
      rhsType.getRank() != 3 || resultType.getRank() != 3) {
    return false;
  }
  if (!lhsType.getElementType().isF16() || !rhsType.getElementType().isF16() ||
      !resultType.getElementType().isF32()) {
    return false;
  }
  return lhsType.isDynamicDim(0) && lhsType.isDynamicDim(1) &&
         lhsType.isDynamicDim(2) && rhsType.isDynamicDim(0) &&
         rhsType.isDynamicDim(1) && rhsType.getDimSize(2) == 128 &&
         resultType.isDynamicDim(0) && resultType.isDynamicDim(1) &&
         resultType.getDimSize(2) == 128;
}

static void insertAttentionContextBarriers(FunctionOpInterface funcOp) {
  SmallVector<linalg::BatchMatmulOp> candidates;
  funcOp.walk([&](linalg::BatchMatmulOp op) {
    if (isFlattenedAttentionContext(op)) {
      candidates.push_back(op);
    }
  });

  for (linalg::BatchMatmulOp op : candidates) {
    Value result = op.getResult(0);
    if (llvm::any_of(result.getUsers(), [](Operation *user) {
          return isa<IREE::TensorExt::ComputeBarrierEndOp>(user);
        })) {
      continue;
    }
    OpBuilder builder(op->getContext());
    builder.setInsertionPointAfter(op);
    auto resultType = cast<RankedTensorType>(result.getType());
    SmallVector<Value> dynamicDims;
    SmallVector<Operation *> internalOps;
    for (int64_t i = 0, e = resultType.getRank(); i < e; ++i) {
      if (resultType.isDynamicDim(i)) {
        auto dimOp = tensor::DimOp::create(builder, op.getLoc(), result, i);
        dynamicDims.push_back(dimOp.getResult());
        internalOps.push_back(dimOp.getOperation());
      }
    }
    auto barrier = IREE::TensorExt::ComputeBarrierEndOp::create(
        builder, op.getLoc(), result.getType(), result, dynamicDims);
    internalOps.push_back(barrier.getOperation());
    result.replaceUsesWithIf(barrier.getResult(), [&](OpOperand &use) {
      return !llvm::is_contained(internalOps, use.getOwner());
    });
  }
}

struct AssumeInputShapeBoundsPass final
    : impl::AssumeInputShapeBoundsPassBase<AssumeInputShapeBoundsPass> {
  using Base::Base;

  void runOnOperation() override {
    if (maxBatch <= 0 || maxSeq <= 0) {
      if (barrierAttentionContext) {
        insertAttentionContextBarriers(getOperation());
      }
      return;
    }

    FunctionOpInterface funcOp = getOperation();
    if (funcOp.empty()) {
      return;
    }

    Location loc = funcOp.getLoc();

    for (BlockArgument arg : funcOp.getArguments()) {
      OpBuilder builder(&funcOp.front(), funcOp.front().begin());
      BoundedTensorValue boundedArg =
          assumeRank2TensorBounds(arg, maxBatch, maxSeq, builder, loc);
      if (boundedArg.value) {
        replaceUsesWithBoundedTensor(arg, boundedArg);
      }
    }

    SmallVector<Operation *> tensorImports;
    funcOp.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "hal.tensor.import" &&
          op->getNumResults() == 1) {
        tensorImports.push_back(op);
      }
    });
    for (Operation *importOp : tensorImports) {
      Value importedTensor = importOp->getResult(0);
      OpBuilder builder(importOp->getContext());
      builder.setInsertionPointAfter(importOp);
      BoundedTensorValue boundedImport = assumeRank2TensorBounds(
          importedTensor, maxBatch, maxSeq, builder, importOp->getLoc());
      if (boundedImport.value) {
        replaceUsesWithBoundedTensor(importedTensor, boundedImport);
      }
    }

    if (barrierAttentionContext) {
      insertAttentionContextBarriers(funcOp);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::GlobalOptimization
