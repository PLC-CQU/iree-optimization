// Same as dynamic_bounded_flattened_heads_attn_context.mlir, but inserts a
// tensor compute barrier after the flattened batch_matmul. The goal is to keep
// dispatch creation from fusing the following expand/truncation back into the
// matmul dispatch, which otherwise reconstructs a rank-5 lowering shape.
func.func @main(%probs: tensor<?x32x?x?xf16>,
                %v: tensor<?x32x?x128xf16>) -> tensor<?x32x?x128xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.0 : f32

  %b = tensor.dim %v, %c0 : tensor<?x32x?x128xf16>
  %s = tensor.dim %v, %c2 : tensor<?x32x?x128xf16>
  %b_bounded = util.assume.int %b<umin = 1, umax = 8> : index
  %s_bounded = util.assume.int %s<umin = 1, umax = 128> : index

  %probs_bounded = tensor.extract_slice %probs[0, 0, 0, 0]
      [%b_bounded, 32, %s_bounded, %s_bounded] [1, 1, 1, 1]
      : tensor<?x32x?x?xf16> to tensor<?x32x?x?xf16>
  %v_bounded = tensor.extract_slice %v[0, 0, 0, 0]
      [%b_bounded, 32, %s_bounded, 128] [1, 1, 1, 1]
      : tensor<?x32x?x128xf16> to tensor<?x32x?x128xf16>

  %probs_flat = tensor.collapse_shape %probs_bounded [[0, 1], [2], [3]]
      : tensor<?x32x?x?xf16> into tensor<?x?x?xf16>
  %v_flat = tensor.collapse_shape %v_bounded [[0, 1], [2], [3]]
      : tensor<?x32x?x128xf16> into tensor<?x?x128xf16>
  %empty4 = tensor.empty(%b_bounded, %s_bounded) : tensor<?x32x?x128xf32>
  %empty3 = tensor.collapse_shape %empty4 [[0, 1], [2], [3]]
      : tensor<?x32x?x128xf32> into tensor<?x?x128xf32>
  %init = linalg.fill ins(%cst : f32)
      outs(%empty3 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
  %context_flat = linalg.batch_matmul
      ins(%probs_flat, %v_flat
          : tensor<?x?x?xf16>, tensor<?x?x128xf16>)
      outs(%init : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
  %bh = arith.muli %b_bounded, %c32 : index
  %barrier = iree_tensor_ext.compute_barrier.start %context_flat
      : tensor<?x?x128xf32>{%bh, %s_bounded} -> tensor<?x?x128xf32>
  %context = tensor.expand_shape %barrier [[0, 1], [2], [3]]
      output_shape [%b_bounded, 32, %s_bounded, 128]
      : tensor<?x?x128xf32> into tensor<?x32x?x128xf32>
  return %context : tensor<?x32x?x128xf32>
}
