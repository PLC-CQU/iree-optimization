// Bounded dynamic rank-5 DeepSeek-like attention context:
//   probs: [B, 32, S, S]
//   v:     [B, 32, S, 128]
//   out:   [B, 32, S, 128]
//
// This keeps the full-model rank-5 loop structure. It is useful as a contrast
// with dynamic_bounded_flattened_heads_attn_context.mlir.
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

  %empty = tensor.empty(%b_bounded, %s_bounded) : tensor<?x32x?x128xf32>
  %init = linalg.fill ins(%cst : f32)
      outs(%empty : tensor<?x32x?x128xf32>) -> tensor<?x32x?x128xf32>
  %context = linalg.generic {
      indexing_maps = [
        affine_map<(b, h, q, n, k) -> (b, h, q, k)>,
        affine_map<(b, h, q, n, k) -> (b, h, k, n)>,
        affine_map<(b, h, q, n, k) -> (b, h, q, n)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
    } ins(%probs_bounded, %v_bounded
        : tensor<?x32x?x?xf16>, tensor<?x32x?x128xf16>)
      outs(%init : tensor<?x32x?x128xf32>) {
    ^bb0(%prob: f16, %value: f16, %acc: f32):
      %prob_f32 = arith.extf %prob : f16 to f32
      %value_f32 = arith.extf %value : f16 to f32
      %mul = arith.mulf %prob_f32, %value_f32 : f32
      %add = arith.addf %acc, %mul : f32
      linalg.yield %add : f32
    } -> tensor<?x32x?x128xf32>
  return %context : tensor<?x32x?x128xf32>
}
