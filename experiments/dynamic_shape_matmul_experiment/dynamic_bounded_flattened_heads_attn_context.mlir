// Bounded dynamic attention context with B and heads flattened:
//   input ABI:
//     probs: [B, 32, S, S]
//     v:     [B, 32, S, 128]
//   internal matmul:
//     probs: [B*32, S, S]
//     v:     [B*32, S, 128]
//     out:   [B*32, S, 128]
//
// This keeps the caller-visible rank-5 shape but presents codegen with the
// rank-4/batch_matmul structure that already worked in the standalone bounded
// attention context microbenchmark.
func.func @main(%probs: tensor<?x32x?x?xf16>,
                %v: tensor<?x32x?x128xf16>) -> tensor<?x32x?x128xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
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
  %context = tensor.expand_shape %context_flat [[0, 1], [2], [3]]
      output_shape [%b_bounded, 32, %s_bounded, 128]
      : tensor<?x?x128xf32> into tensor<?x32x?x128xf32>
  return %context : tensor<?x32x?x128xf32>
}
