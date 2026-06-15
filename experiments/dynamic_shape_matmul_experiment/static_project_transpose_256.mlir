module {
  func.func @main(%lhs: tensor<4x128x256xf32>,
                  %wq: tensor<256x256xf32>) -> tensor<4x32x128x8xf32>
      attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x128x256xf32>
    %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x128x256xf32>) -> tensor<4x128x256xf32>
    %q = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
      } ins(%lhs, %wq : tensor<4x128x256xf32>, tensor<256x256xf32>)
        outs(%init : tensor<4x128x256xf32>) {
      ^bb0(%l: f32, %r: f32, %acc: f32):
        %mul = arith.mulf %l, %r : f32
        %add = arith.addf %mul, %acc : f32
        linalg.yield %add : f32
      } -> tensor<4x128x256xf32>
    %q4 = tensor.expand_shape %q [[0], [1], [2, 3]] output_shape [4, 128, 32, 8] : tensor<4x128x256xf32> into tensor<4x128x32x8xf32>
    %out_empty = tensor.empty() : tensor<4x32x128x8xf32>
    %out = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%q4 : tensor<4x128x32x8xf32>) outs(%out_empty : tensor<4x32x128x8xf32>) {
      ^bb0(%in: f32, %out_arg: f32):
        linalg.yield %in : f32
      } -> tensor<4x32x128x8xf32>
    return %out : tensor<4x32x128x8xf32>
  }
}
