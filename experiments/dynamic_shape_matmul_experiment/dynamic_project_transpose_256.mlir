module {
  func.func @main(%lhs: tensor<?x?x256xf32>,
                  %wq: tensor<256x256xf32>) -> tensor<?x32x?x8xf32>
      attributes {iree.abi.stub} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %b = tensor.dim %lhs, %c0 : tensor<?x?x256xf32>
    %s = tensor.dim %lhs, %c1 : tensor<?x?x256xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty(%b, %s) : tensor<?x?x256xf32>
    %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?x256xf32>) -> tensor<?x?x256xf32>
    %q = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
      } ins(%lhs, %wq : tensor<?x?x256xf32>, tensor<256x256xf32>)
        outs(%init : tensor<?x?x256xf32>) {
      ^bb0(%l: f32, %r: f32, %acc: f32):
        %mul = arith.mulf %l, %r : f32
        %add = arith.addf %mul, %acc : f32
        linalg.yield %add : f32
      } -> tensor<?x?x256xf32>
    %q4 = tensor.expand_shape %q [[0], [1], [2, 3]] output_shape [%b, %s, 32, 8] : tensor<?x?x256xf32> into tensor<?x?x32x8xf32>
    %out_empty = tensor.empty(%b, %s) : tensor<?x32x?x8xf32>
    %out = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%q4 : tensor<?x?x32x8xf32>) outs(%out_empty : tensor<?x32x?x8xf32>) {
      ^bb0(%in: f32, %out_arg: f32):
        linalg.yield %in : f32
      } -> tensor<?x32x?x8xf32>
    return %out : tensor<?x32x?x8xf32>
  }
}
