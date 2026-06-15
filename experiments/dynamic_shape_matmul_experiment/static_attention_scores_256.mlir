module {
  func.func @main(%q: tensor<4x32x128x8xf32>,
                  %k_t: tensor<4x32x8x128xf32>) -> tensor<4x32x128x128xf32>
      attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %q_flat = tensor.collapse_shape %q [[0, 1], [2], [3]] : tensor<4x32x128x8xf32> into tensor<128x128x8xf32>
    %k_flat = tensor.collapse_shape %k_t [[0, 1], [2], [3]] : tensor<4x32x8x128xf32> into tensor<128x8x128xf32>
    %empty = tensor.empty() : tensor<128x128x128xf32>
    %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x128x128xf32>) -> tensor<128x128x128xf32>
    %scores_flat = linalg.batch_matmul ins(%q_flat, %k_flat : tensor<128x128x8xf32>, tensor<128x8x128xf32>)
                                       outs(%init : tensor<128x128x128xf32>) -> tensor<128x128x128xf32>
    %scores = tensor.expand_shape %scores_flat [[0, 1], [2], [3]] output_shape [4, 32, 128, 128] : tensor<128x128x128xf32> into tensor<4x32x128x128xf32>
    return %scores : tensor<4x32x128x128xf32>
  }
}
