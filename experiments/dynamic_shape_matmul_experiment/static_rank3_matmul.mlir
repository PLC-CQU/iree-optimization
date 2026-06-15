module {
  func.func @main(%lhs: tensor<4x128x4096xf32>,
                  %rhs: tensor<4096x4096xf32>) -> tensor<4x128x4096xf32>
      attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x128x4096xf32>
    %init = linalg.fill ins(%cst : f32)
                        outs(%empty : tensor<4x128x4096xf32>)
                        -> tensor<4x128x4096xf32>
    %0 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
      } ins(%lhs, %rhs : tensor<4x128x4096xf32>, tensor<4096x4096xf32>)
        outs(%init : tensor<4x128x4096xf32>) {
      ^bb0(%l: f32, %r: f32, %acc: f32):
        %mul = arith.mulf %l, %r : f32
        %add = arith.addf %mul, %acc : f32
        linalg.yield %add : f32
      } -> tensor<4x128x4096xf32>
    return %0 : tensor<4x128x4096xf32>
  }
}
