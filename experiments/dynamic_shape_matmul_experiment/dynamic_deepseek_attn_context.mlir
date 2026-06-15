// Dynamic version of DeepSeek-like attention context:
//   probs: [?B*H, ?S, ?S]
//   v: [?B*H, ?S, 128]
//   out: [?B*H, ?S, 128]
func.func @main(%probs: tensor<?x?x?xf16>,
                %v: tensor<?x?x128xf16>) -> tensor<?x?x128xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %b = tensor.dim %v, %c0 : tensor<?x?x128xf16>
  %s = tensor.dim %v, %c1 : tensor<?x?x128xf16>
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty(%b, %s) : tensor<?x?x128xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
  %context = linalg.batch_matmul ins(%probs, %v : tensor<?x?x?xf16>, tensor<?x?x128xf16>)
    outs(%init : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
  return %context : tensor<?x?x128xf32>
}
