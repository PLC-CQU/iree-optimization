// DeepSeek-like attention context:
//   probs: [B*H, S, S] = [128, 104, 104]
//   v: [B*H, S, D] = [128, 104, 128]
//   out: [B*H, S, D] = [128, 104, 128]
func.func @main(%probs: tensor<128x104x104xf16>,
                %v: tensor<128x104x128xf16>) -> tensor<128x104x128xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x104x128xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x104x128xf32>) -> tensor<128x104x128xf32>
  %context = linalg.batch_matmul ins(%probs, %v : tensor<128x104x104xf16>, tensor<128x104x128xf16>)
    outs(%init : tensor<128x104x128xf32>) -> tensor<128x104x128xf32>
  return %context : tensor<128x104x128xf32>
}
