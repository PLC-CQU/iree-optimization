module {
  func.func @main(%lhs: tensor<?x?x256xf32>,
                  %w_gate: tensor<256x512xf32>,
                  %w_up: tensor<256x512xf32>,
                  %w_down: tensor<512x256xf32>) -> tensor<?x?x256xf32>
      attributes {iree.abi.stub} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %b = tensor.dim %lhs, %c0 : tensor<?x?x256xf32>
    %s = tensor.dim %lhs, %c1 : tensor<?x?x256xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %empty_i = tensor.empty(%b, %s) : tensor<?x?x512xf32>
    %init_i = linalg.fill ins(%cst : f32) outs(%empty_i : tensor<?x?x512xf32>) -> tensor<?x?x512xf32>
    %gate = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
      } ins(%lhs, %w_gate : tensor<?x?x256xf32>, tensor<256x512xf32>)
        outs(%init_i : tensor<?x?x512xf32>) {
      ^bb0(%l: f32, %r: f32, %acc: f32):
        %mul = arith.mulf %l, %r : f32
        %add = arith.addf %mul, %acc : f32
        linalg.yield %add : f32
      } -> tensor<?x?x512xf32>
    %up = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
      } ins(%lhs, %w_up : tensor<?x?x256xf32>, tensor<256x512xf32>)
        outs(%init_i : tensor<?x?x512xf32>) {
      ^bb0(%l: f32, %r: f32, %acc: f32):
        %mul = arith.mulf %l, %r : f32
        %add = arith.addf %mul, %acc : f32
        linalg.yield %add : f32
      } -> tensor<?x?x512xf32>
    %prod_empty = tensor.empty(%b, %s) : tensor<?x?x512xf32>
    %prod = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%gate, %up : tensor<?x?x512xf32>, tensor<?x?x512xf32>)
        outs(%prod_empty : tensor<?x?x512xf32>) {
      ^bb0(%g: f32, %u: f32, %out: f32):
        %mul = arith.mulf %g, %u : f32
        linalg.yield %mul : f32
      } -> tensor<?x?x512xf32>
    %empty_o = tensor.empty(%b, %s) : tensor<?x?x256xf32>
    %init_o = linalg.fill ins(%cst : f32) outs(%empty_o : tensor<?x?x256xf32>) -> tensor<?x?x256xf32>
    %down = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
      } ins(%prod, %w_down : tensor<?x?x512xf32>, tensor<512x256xf32>)
        outs(%init_o : tensor<?x?x256xf32>) {
      ^bb0(%l: f32, %r: f32, %acc: f32):
        %mul = arith.mulf %l, %r : f32
        %add = arith.addf %mul, %acc : f32
        linalg.yield %add : f32
      } -> tensor<?x?x256xf32>
    return %down : tensor<?x?x256xf32>
  }
}
