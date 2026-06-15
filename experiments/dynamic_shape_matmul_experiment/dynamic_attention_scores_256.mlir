module {
  func.func @main(%q: tensor<?x32x?x8xf32>,
                  %k_t: tensor<?x32x8x?xf32>) -> tensor<?x32x?x?xf32>
      attributes {iree.abi.stub} {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %b = tensor.dim %q, %c0 : tensor<?x32x?x8xf32>
    %s = tensor.dim %q, %c2 : tensor<?x32x?x8xf32>
    %bh = arith.muli %b, %c32 : index
    %cst = arith.constant 0.000000e+00 : f32
    %q_flat = tensor.collapse_shape %q [[0, 1], [2], [3]] : tensor<?x32x?x8xf32> into tensor<?x?x8xf32>
    %k_flat = tensor.collapse_shape %k_t [[0, 1], [2], [3]] : tensor<?x32x8x?xf32> into tensor<?x8x?xf32>
    %empty = tensor.empty(%bh, %s, %s) : tensor<?x?x?xf32>
    %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %scores_flat = linalg.batch_matmul ins(%q_flat, %k_flat : tensor<?x?x8xf32>, tensor<?x8x?xf32>)
                                       outs(%init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %scores = tensor.expand_shape %scores_flat [[0, 1], [2], [3]] output_shape [%b, 32, %s, %s] : tensor<?x?x?xf32> into tensor<?x32x?x?xf32>
    return %scores : tensor<?x32x?x?xf32>
  }
}
