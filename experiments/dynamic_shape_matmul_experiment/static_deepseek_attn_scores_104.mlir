// DeepSeek-like attention scores:
//   q: [B*H, S, D] = [128, 104, 128]
//   k: [B*H, D, S] = [128, 128, 104]
//   out: [B*H, S, S] = [128, 104, 104]
func.func @main(%q: tensor<128x104x128xf16>,
                %k: tensor<128x128x104xf16>) -> tensor<128x104x104xf16> {
  %cst = arith.constant 0.0 : f32
  %empty_f32 = tensor.empty() : tensor<128x104x104xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty_f32 : tensor<128x104x104xf32>) -> tensor<128x104x104xf32>
  %scores_f32 = linalg.batch_matmul ins(%q, %k : tensor<128x104x128xf16>, tensor<128x128x104xf16>)
    outs(%init : tensor<128x104x104xf32>) -> tensor<128x104x104xf32>
  %empty_f16 = tensor.empty() : tensor<128x104x104xf16>
  %scores_f16 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    ],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%scores_f32 : tensor<128x104x104xf32>)
    outs(%empty_f16 : tensor<128x104x104xf16>) {
  ^bb0(%in: f32, %out: f16):
    %trunc = arith.truncf %in : f32 to f16
    linalg.yield %trunc : f16
  } -> tensor<128x104x104xf16>
  return %scores_f16 : tensor<128x104x104xf16>
}
