module {
  func.func @main(%lhs: tensor<4x128x256xf32>,
                  %wq: tensor<256x256xf32>,
                  %wk: tensor<256x256xf32>,
                  %wv: tensor<256x256xf32>)
      -> (tensor<4x128x256xf32>, tensor<4x128x256xf32>, tensor<4x128x256xf32>)
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
    %k = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
      } ins(%lhs, %wk : tensor<4x128x256xf32>, tensor<256x256xf32>)
        outs(%init : tensor<4x128x256xf32>) {
      ^bb0(%l: f32, %r: f32, %acc: f32):
        %mul = arith.mulf %l, %r : f32
        %add = arith.addf %mul, %acc : f32
        linalg.yield %add : f32
      } -> tensor<4x128x256xf32>
    %v = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
      } ins(%lhs, %wv : tensor<4x128x256xf32>, tensor<256x256xf32>)
        outs(%init : tensor<4x128x256xf32>) {
      ^bb0(%l: f32, %r: f32, %acc: f32):
        %mul = arith.mulf %l, %r : f32
        %add = arith.addf %mul, %acc : f32
        linalg.yield %add : f32
      } -> tensor<4x128x256xf32>
    return %q, %k, %v : tensor<4x128x256xf32>, tensor<4x128x256xf32>, tensor<4x128x256xf32>
  }
}
