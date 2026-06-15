// Bounded dynamic version of DeepSeek-like attention context:
//   probs: [?B*H, ?S, ?S]
//   v: [?B*H, ?S, 128]
//   out: [?B*H, ?S, 128]
//
// The function ABI is still dynamic. The util.assume.int ops describe a
// compile-time range contract for the dynamic dimensions, so GPU schedule
// selection can use finite upper bounds without fixing the runtime shape.
func.func @main(%probs: tensor<?x?x?xf16>,
                %v: tensor<?x?x128xf16>) -> tensor<?x?x128xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32

  %bh = tensor.dim %v, %c0 : tensor<?x?x128xf16>
  %s = tensor.dim %v, %c1 : tensor<?x?x128xf16>
  %bh_bounded = util.assume.int %bh<umin = 1, umax = 128> : index
  %s_bounded = util.assume.int %s<umin = 1, umax = 128> : index

  %probs_bounded = tensor.extract_slice %probs[0, 0, 0]
      [%bh_bounded, %s_bounded, %s_bounded] [1, 1, 1]
      : tensor<?x?x?xf16> to tensor<?x?x?xf16>
  %v_bounded = tensor.extract_slice %v[0, 0, 0]
      [%bh_bounded, %s_bounded, 128] [1, 1, 1]
      : tensor<?x?x128xf16> to tensor<?x?x128xf16>

  %empty = tensor.empty(%bh_bounded, %s_bounded) : tensor<?x?x128xf32>
  %init = linalg.fill ins(%cst : f32)
      outs(%empty : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
  %context = linalg.batch_matmul
      ins(%probs_bounded, %v_bounded
          : tensor<?x?x?xf16>, tensor<?x?x128xf16>)
      outs(%init : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
  return %context : tensor<?x?x128xf32>
}
