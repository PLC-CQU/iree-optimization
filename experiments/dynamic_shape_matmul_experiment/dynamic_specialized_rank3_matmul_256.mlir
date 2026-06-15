module {
  func.func @main(%lhs: tensor<?x?x256xf32>,
                  %rhs: tensor<256x256xf32>) -> tensor<?x?x256xf32> {
    %lhs_static = tensor.cast %lhs : tensor<?x?x256xf32> to tensor<4x128x256xf32>
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<4x128x256xf32>
    %filled = linalg.fill ins(%zero : f32)
      outs(%init : tensor<4x128x256xf32>) -> tensor<4x128x256xf32>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(b, s, n, k) -> (b, s, k)>,
        affine_map<(b, s, n, k) -> (k, n)>,
        affine_map<(b, s, n, k) -> (b, s, n)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%lhs_static, %rhs : tensor<4x128x256xf32>, tensor<256x256xf32>)
      outs(%filled : tensor<4x128x256xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %mul = arith.mulf %a, %b : f32
      %add = arith.addf %c, %mul : f32
      linalg.yield %add : f32
    } -> tensor<4x128x256xf32>
    %out_dynamic = tensor.cast %out : tensor<4x128x256xf32> to tensor<?x?x256xf32>
    return %out_dynamic : tensor<?x?x256xf32>
  }
}
