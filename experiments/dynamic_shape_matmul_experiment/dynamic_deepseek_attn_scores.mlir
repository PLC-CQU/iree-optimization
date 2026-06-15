// Dynamic version of DeepSeek-like attention scores:
//   q: [?B*H, ?S, 128]
//   k: [?B*H, 128, ?S]
//   out: [?B*H, ?S, ?S]
func.func @main(%q: tensor<?x?x128xf16>,
                %k: tensor<?x128x?xf16>) -> tensor<?x?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %b = tensor.dim %q, %c0 : tensor<?x?x128xf16>
  %s = tensor.dim %q, %c1 : tensor<?x?x128xf16>
  %cst = arith.constant 0.0 : f32
  %empty_f32 = tensor.empty(%b, %s, %s) : tensor<?x?x?xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty_f32 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %scores_f32 = linalg.batch_matmul ins(%q, %k : tensor<?x?x128xf16>, tensor<?x128x?xf16>)
    outs(%init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %empty_f16 = tensor.empty(%b, %s, %s) : tensor<?x?x?xf16>
  %scores_f16 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    ],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%scores_f32 : tensor<?x?x?xf32>)
    outs(%empty_f16 : tensor<?x?x?xf16>) {
  ^bb0(%in: f32, %out: f16):
    %trunc = arith.truncf %in : f32 to f16
    linalg.yield %trunc : f16
  } -> tensor<?x?x?xf16>
  return %scores_f16 : tensor<?x?x?xf16>
}
