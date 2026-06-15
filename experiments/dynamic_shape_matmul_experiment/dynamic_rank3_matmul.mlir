module {
  func.func @main(%lhs: tensor<?x?x4096xf32>,
                  %rhs: tensor<4096x4096xf32>) -> tensor<?x?x4096xf32>
      attributes {iree.abi.stub} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %b = tensor.dim %lhs, %c0 : tensor<?x?x4096xf32>
    %s = tensor.dim %lhs, %c1 : tensor<?x?x4096xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty(%b, %s) : tensor<?x?x4096xf32>
    %init = linalg.fill ins(%cst : f32)
                        outs(%empty : tensor<?x?x4096xf32>)
                        -> tensor<?x?x4096xf32>
    %0 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
      } ins(%lhs, %rhs : tensor<?x?x4096xf32>, tensor<4096x4096xf32>)
        outs(%init : tensor<?x?x4096xf32>) {
      ^bb0(%l: f32, %r: f32, %acc: f32):
        %mul = arith.mulf %l, %r : f32
        %add = arith.addf %mul, %acc : f32
        linalg.yield %add : f32
      } -> tensor<?x?x4096xf32>
    return %0 : tensor<?x?x4096xf32>
  }
}
