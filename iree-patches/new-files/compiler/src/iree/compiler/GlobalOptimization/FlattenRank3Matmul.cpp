// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_FLATTENRANK3MATMULPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc" // IWYU pragma: export

namespace {

static bool isCompatibleDim(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs;
}

static OpFoldResult getMixedDim(OpBuilder &builder, Location loc, Value value,
                                int64_t dim) {
  auto type = cast<RankedTensorType>(value.getType());
  int64_t staticDim = type.getDimSize(dim);
  if (!ShapedType::isDynamic(staticDim)) {
    return builder.getIndexAttr(staticDim);
  }
  return tensor::DimOp::create(builder, loc, value, dim).getResult();
}

static Value materializeIndexValue(OpBuilder &builder, Location loc,
                                   OpFoldResult dim) {
  if (auto attr = dyn_cast<Attribute>(dim)) {
    return arith::ConstantIndexOp::create(builder, loc,
                                          cast<IntegerAttr>(attr).getInt());
  }
  return cast<Value>(dim);
}

static OpFoldResult multiplyDims(OpBuilder &builder, Location loc,
                                 OpFoldResult lhs, OpFoldResult rhs) {
  auto lhsAttr = dyn_cast<Attribute>(lhs);
  auto rhsAttr = dyn_cast<Attribute>(rhs);
  if (lhsAttr && rhsAttr) {
    int64_t lhsValue = cast<IntegerAttr>(lhsAttr).getInt();
    int64_t rhsValue = cast<IntegerAttr>(rhsAttr).getInt();
    return builder.getIndexAttr(lhsValue * rhsValue);
  }
  Value lhsValue = materializeIndexValue(builder, loc, lhs);
  Value rhsValue = materializeIndexValue(builder, loc, rhs);
  return arith::MulIOp::create(builder, loc, lhsValue, rhsValue).getResult();
}

static int64_t getStaticDim(OpFoldResult dim) {
  if (auto attr = dyn_cast<Attribute>(dim)) {
    return cast<IntegerAttr>(attr).getInt();
  }
  return ShapedType::kDynamic;
}

static bool isRank3ByRank2MatmulBody(linalg::GenericOp genericOp) {
  Block *body = genericOp.getBody();
  if (!body || body->getNumArguments() != 3) {
    return false;
  }

  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yieldOp || yieldOp->getNumOperands() != 1) {
    return false;
  }

  Operation *addOp = yieldOp->getOperand(0).getDefiningOp();
  if (!addOp || !isa<arith::AddFOp, arith::AddIOp>(addOp)) {
    return false;
  }

  Value accArg = body->getArgument(2);
  Value maybeMul;
  if (addOp->getOperand(0) == accArg) {
    maybeMul = addOp->getOperand(1);
  } else if (addOp->getOperand(1) == accArg) {
    maybeMul = addOp->getOperand(0);
  } else {
    return false;
  }

  Operation *mulOp = maybeMul.getDefiningOp();
  if (!mulOp || !isa<arith::MulFOp, arith::MulIOp>(mulOp)) {
    return false;
  }

  Value lhsArg = body->getArgument(0);
  Value rhsArg = body->getArgument(1);
  return (mulOp->getOperand(0) == lhsArg && mulOp->getOperand(1) == rhsArg) ||
         (mulOp->getOperand(0) == rhsArg && mulOp->getOperand(1) == lhsArg);
}

static Value getRank2BroadcastSource(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1 ||
      genericOp->getNumResults() != 1) {
    return {};
  }

  Value source = genericOp.getDpsInputs()[0];
  Value output = genericOp.getDpsInits()[0];
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  auto outputType = dyn_cast<RankedTensorType>(output.getType());
  if (!sourceType || !outputType || sourceType.getRank() != 2 ||
      outputType.getRank() != 3) {
    return {};
  }

  if (!isCompatibleDim(sourceType.getDimSize(0), outputType.getDimSize(1)) ||
      !isCompatibleDim(sourceType.getDimSize(1), outputType.getDimSize(2))) {
    return {};
  }

  MLIRContext *context = genericOp.getContext();
  AffineExpr bDim = getAffineDimExpr(0, context);
  AffineExpr kDim = getAffineDimExpr(1, context);
  AffineExpr nDim = getAffineDimExpr(2, context);
  SmallVector<AffineMap> expectedMaps = {
      AffineMap::get(3, 0, {kDim, nDim}, context),
      AffineMap::get(3, 0, {bDim, kDim, nDim}, context),
  };
  if (genericOp.getIndexingMapsArray() != expectedMaps) {
    return {};
  }

  SmallVector<utils::IteratorType> expectedIterators = {
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::parallel};
  if (genericOp.getIteratorTypesArray() != expectedIterators) {
    return {};
  }

  Block *body = genericOp.getBody();
  if (!body || body->getNumArguments() != 2) {
    return {};
  }
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yieldOp || yieldOp->getNumOperands() != 1 ||
      yieldOp->getOperand(0) != body->getArgument(0)) {
    return {};
  }

  return source;
}

struct FlattenRank3Matmul final : OpRewritePattern<linalg::GenericOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1 ||
        genericOp->getNumResults() != 1) {
      return failure();
    }

    Value lhs = genericOp.getDpsInputs()[0];
    Value rhs = genericOp.getDpsInputs()[1];
    Value output = genericOp.getDpsInits()[0];

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!lhsType || !rhsType || !outputType || lhsType.getRank() != 3 ||
        rhsType.getRank() != 2 || outputType.getRank() != 3) {
      return failure();
    }

    if (!isCompatibleDim(lhsType.getDimSize(2), rhsType.getDimSize(0)) ||
        !isCompatibleDim(lhsType.getDimSize(0), outputType.getDimSize(0)) ||
        !isCompatibleDim(lhsType.getDimSize(1), outputType.getDimSize(1)) ||
        !isCompatibleDim(rhsType.getDimSize(1), outputType.getDimSize(2))) {
      return failure();
    }

    MLIRContext *context = genericOp.getContext();
    AffineExpr bDim = getAffineDimExpr(0, context);
    AffineExpr sDim = getAffineDimExpr(1, context);
    AffineExpr nDim = getAffineDimExpr(2, context);
    AffineExpr kDim = getAffineDimExpr(3, context);
    SmallVector<AffineMap> expectedMaps = {
        AffineMap::get(4, 0, {bDim, sDim, kDim}, context),
        AffineMap::get(4, 0, {kDim, nDim}, context),
        AffineMap::get(4, 0, {bDim, sDim, nDim}, context),
    };
    if (genericOp.getIndexingMapsArray() != expectedMaps) {
      return failure();
    }

    SmallVector<utils::IteratorType> expectedIterators = {
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::parallel, utils::IteratorType::reduction};
    if (genericOp.getIteratorTypesArray() != expectedIterators) {
      return failure();
    }

    if (!isRank3ByRank2MatmulBody(genericOp)) {
      return failure();
    }

    Location loc = genericOp.getLoc();
    OpFoldResult bSize = getMixedDim(rewriter, loc, lhs, 0);
    OpFoldResult sSize = getMixedDim(rewriter, loc, lhs, 1);
    OpFoldResult bsSize = multiplyDims(rewriter, loc, bSize, sSize);
    OpFoldResult kSize = getMixedDim(rewriter, loc, lhs, 2);
    OpFoldResult nSize = getMixedDim(rewriter, loc, rhs, 1);

    SmallVector<ReassociationIndices> lhsReassoc = {{0, 1}, {2}};
    SmallVector<ReassociationIndices> outputReassoc = {{0, 1}, {2}};
    auto flatLhsType = RankedTensorType::get(
        {getStaticDim(bsSize), getStaticDim(kSize)}, lhsType.getElementType());
    auto flatOutputType = RankedTensorType::get(
        {getStaticDim(bsSize), getStaticDim(nSize)},
        outputType.getElementType());

    Value flatLhs = tensor::CollapseShapeOp::create(
        rewriter, loc, flatLhsType, lhs, lhsReassoc);
    Value flatOutput = tensor::CollapseShapeOp::create(
        rewriter, loc, flatOutputType, output, outputReassoc);
    auto matmulOp = linalg::MatmulOp::create(
        rewriter, loc, flatOutputType, ValueRange{flatLhs, rhs},
        ValueRange{flatOutput});
    SmallVector<OpFoldResult> expandedShape = {bSize, sSize, nSize};
    Value expanded = tensor::ExpandShapeOp::create(
        rewriter, loc, outputType, matmulOp.getResult(0), outputReassoc,
        expandedShape);

    rewriter.replaceOp(genericOp, ArrayRef<Value>{expanded});
    return success();
  }
};

struct FlattenBroadcastedBatchMatmul final
    : OpRewritePattern<linalg::BatchMatmulOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp batchMatmulOp,
                                PatternRewriter &rewriter) const override {
    if (batchMatmulOp.getNumDpsInputs() != 2 ||
        batchMatmulOp.getNumDpsInits() != 1 ||
        batchMatmulOp->getNumResults() != 1) {
      return failure();
    }

    Value lhs = batchMatmulOp.getDpsInputs()[0];
    Value broadcastedRhs = batchMatmulOp.getDpsInputs()[1];
    Value output = batchMatmulOp.getDpsInits()[0];

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto broadcastedRhsType =
        dyn_cast<RankedTensorType>(broadcastedRhs.getType());
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!lhsType || !broadcastedRhsType || !outputType ||
        lhsType.getRank() != 3 || broadcastedRhsType.getRank() != 3 ||
        outputType.getRank() != 3) {
      return failure();
    }

    if (!isCompatibleDim(lhsType.getDimSize(0),
                         broadcastedRhsType.getDimSize(0)) ||
        !isCompatibleDim(lhsType.getDimSize(0), outputType.getDimSize(0)) ||
        !isCompatibleDim(lhsType.getDimSize(1), outputType.getDimSize(1)) ||
        !isCompatibleDim(lhsType.getDimSize(2),
                         broadcastedRhsType.getDimSize(1)) ||
        !isCompatibleDim(broadcastedRhsType.getDimSize(2),
                         outputType.getDimSize(2))) {
      return failure();
    }

    MLIRContext *context = batchMatmulOp.getContext();
    AffineExpr bDim = getAffineDimExpr(0, context);
    AffineExpr mDim = getAffineDimExpr(1, context);
    AffineExpr nDim = getAffineDimExpr(2, context);
    AffineExpr kDim = getAffineDimExpr(3, context);
    SmallVector<AffineMap> expectedMaps = {
        AffineMap::get(4, 0, {bDim, mDim, kDim}, context),
        AffineMap::get(4, 0, {bDim, kDim, nDim}, context),
        AffineMap::get(4, 0, {bDim, mDim, nDim}, context),
    };
    if (batchMatmulOp.getIndexingMapsArray() != expectedMaps) {
      return failure();
    }

    auto broadcastOp = broadcastedRhs.getDefiningOp<linalg::GenericOp>();
    if (!broadcastOp) {
      return failure();
    }
    Value rhs = getRank2BroadcastSource(broadcastOp);
    if (!rhs) {
      return failure();
    }
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!rhsType || rhsType.getRank() != 2 ||
        !isCompatibleDim(lhsType.getDimSize(2), rhsType.getDimSize(0)) ||
        !isCompatibleDim(rhsType.getDimSize(1), outputType.getDimSize(2))) {
      return failure();
    }

    Location loc = batchMatmulOp.getLoc();
    OpFoldResult bSize = getMixedDim(rewriter, loc, lhs, 0);
    OpFoldResult mSize = getMixedDim(rewriter, loc, lhs, 1);
    OpFoldResult bmSize = multiplyDims(rewriter, loc, bSize, mSize);
    OpFoldResult kSize = getMixedDim(rewriter, loc, lhs, 2);
    OpFoldResult nSize = getMixedDim(rewriter, loc, rhs, 1);

    SmallVector<ReassociationIndices> lhsReassoc = {{0, 1}, {2}};
    SmallVector<ReassociationIndices> outputReassoc = {{0, 1}, {2}};
    auto flatLhsType = RankedTensorType::get(
        {getStaticDim(bmSize), getStaticDim(kSize)}, lhsType.getElementType());
    auto flatOutputType = RankedTensorType::get(
        {getStaticDim(bmSize), getStaticDim(nSize)},
        outputType.getElementType());

    Value flatLhs = tensor::CollapseShapeOp::create(
        rewriter, loc, flatLhsType, lhs, lhsReassoc);
    Value flatOutput = tensor::CollapseShapeOp::create(
        rewriter, loc, flatOutputType, output, outputReassoc);
    auto matmulOp = linalg::MatmulOp::create(
        rewriter, loc, flatOutputType, ValueRange{flatLhs, rhs},
        ValueRange{flatOutput});
    SmallVector<OpFoldResult> expandedShape = {bSize, mSize, nSize};
    Value expanded = tensor::ExpandShapeOp::create(
        rewriter, loc, outputType, matmulOp.getResult(0), outputReassoc,
        expandedShape);

    rewriter.replaceOp(batchMatmulOp, ArrayRef<Value>{expanded});
    return success();
  }
};

class FlattenRank3MatmulPass final
    : public impl::FlattenRank3MatmulPassBase<FlattenRank3MatmulPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<FlattenBroadcastedBatchMatmul, FlattenRank3Matmul>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::GlobalOptimization
