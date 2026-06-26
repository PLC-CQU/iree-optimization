// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-global-opt-flatten-rank3-matmul))' %s | FileCheck %s

func.func @rank3_by_rank2_matmul(%lhs: tensor<2x3x4xf32>,
                                 %rhs: tensor<4x5xf32>,
                                 %out: tensor<2x3x5xf32>) -> tensor<2x3x5xf32> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%lhs, %rhs : tensor<2x3x4xf32>, tensor<4x5xf32>)
      outs(%out : tensor<2x3x5xf32>) {
    ^bb0(%l: f32, %r: f32, %acc: f32):
      %mul = arith.mulf %l, %r : f32
      %add = arith.addf %mul, %acc : f32
      linalg.yield %add : f32
    } -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
}

// CHECK-LABEL: func.func @rank3_by_rank2_matmul
// CHECK-SAME:    %[[LHS:.+]]: tensor<2x3x4xf32>
// CHECK-SAME:    %[[RHS:.+]]: tensor<4x5xf32>
// CHECK-SAME:    %[[OUT:.+]]: tensor<2x3x5xf32>
// CHECK:         %[[FLAT_LHS:.+]] = tensor.collapse_shape %[[LHS]] {{\[\[}}0, 1], [2]] : tensor<2x3x4xf32> into tensor<6x4xf32>
// CHECK:         %[[FLAT_OUT:.+]] = tensor.collapse_shape %[[OUT]] {{\[\[}}0, 1], [2]] : tensor<2x3x5xf32> into tensor<6x5xf32>
// CHECK:         %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:      ins(%[[FLAT_LHS]], %[[RHS]]
// CHECK-SAME:      outs(%[[FLAT_OUT]]
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[MATMUL]] {{\[\[}}0, 1], [2]] output_shape [2, 3, 5] : tensor<6x5xf32> into tensor<2x3x5xf32>
// CHECK:         return %[[EXPANDED]]

// -----

func.func @not_a_matmul_body(%lhs: tensor<2x3x4xf32>,
                             %rhs: tensor<4x5xf32>,
                             %out: tensor<2x3x5xf32>) -> tensor<2x3x5xf32> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%lhs, %rhs : tensor<2x3x4xf32>, tensor<4x5xf32>)
      outs(%out : tensor<2x3x5xf32>) {
    ^bb0(%l: f32, %r: f32, %acc: f32):
      %add = arith.addf %l, %acc : f32
      linalg.yield %add : f32
    } -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
}

// CHECK-LABEL: func.func @not_a_matmul_body
// CHECK:       linalg.generic
// CHECK-NOT:   linalg.matmul

// -----

func.func @broadcast_rhs_batch_matmul(%lhs: tensor<4x32x4096xf16>,
                                      %rhs: tensor<4096x1024xf16>,
                                      %out: tensor<4x32x1024xf32>) -> tensor<4x32x1024xf32> {
  %rhs3_empty = tensor.empty() : tensor<4x4096x1024xf16>
  %rhs3 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%rhs : tensor<4096x1024xf16>)
      outs(%rhs3_empty : tensor<4x4096x1024xf16>) {
    ^bb0(%r: f16, %unused: f16):
      linalg.yield %r : f16
    } -> tensor<4x4096x1024xf16>
  %0 = linalg.batch_matmul ins(%lhs, %rhs3 : tensor<4x32x4096xf16>, tensor<4x4096x1024xf16>)
                             outs(%out : tensor<4x32x1024xf32>) -> tensor<4x32x1024xf32>
  return %0 : tensor<4x32x1024xf32>
}

// CHECK-LABEL: func.func @broadcast_rhs_batch_matmul
// CHECK-SAME:    %[[LHS:.+]]: tensor<4x32x4096xf16>
// CHECK-SAME:    %[[RHS:.+]]: tensor<4096x1024xf16>
// CHECK-SAME:    %[[OUT:.+]]: tensor<4x32x1024xf32>
// CHECK:         %[[FLAT_LHS:.+]] = tensor.collapse_shape %[[LHS]] {{\[\[}}0, 1], [2]] : tensor<4x32x4096xf16> into tensor<128x4096xf16>
// CHECK:         %[[FLAT_OUT:.+]] = tensor.collapse_shape %[[OUT]] {{\[\[}}0, 1], [2]] : tensor<4x32x1024xf32> into tensor<128x1024xf32>
// CHECK:         %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:      ins(%[[FLAT_LHS]], %[[RHS]]
// CHECK-SAME:      outs(%[[FLAT_OUT]]
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[MATMUL]] {{\[\[}}0, 1], [2]] output_shape [4, 32, 1024] : tensor<128x1024xf32> into tensor<4x32x1024xf32>
// CHECK:         return %[[EXPANDED]]
