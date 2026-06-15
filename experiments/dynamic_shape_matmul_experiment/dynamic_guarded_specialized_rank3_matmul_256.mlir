module {
  func.func @main(%lhs: tensor<?x?x256xf32>,
                  %rhs: tensor<256x256xf32>) -> tensor<?x?x256xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %zero = arith.constant 0.0 : f32
    %b = tensor.dim %lhs, %c0 : tensor<?x?x256xf32>
    %s = tensor.dim %lhs, %c1 : tensor<?x?x256xf32>
    %is_b4 = arith.cmpi eq, %b, %c4 : index
    %is_s128 = arith.cmpi eq, %s, %c128 : index
    %is_target_shape = arith.andi %is_b4, %is_s128 : i1
    %out = scf.if %is_target_shape -> tensor<?x?x256xf32> {
      %lhs_static = tensor.cast %lhs
        : tensor<?x?x256xf32> to tensor<4x128x256xf32>
      %init = tensor.empty() : tensor<4x128x256xf32>
      %filled = linalg.fill ins(%zero : f32)
        outs(%init : tensor<4x128x256xf32>) -> tensor<4x128x256xf32>
      %static = linalg.generic {
        indexing_maps = [
          affine_map<(b, s, n, k) -> (b, s, k)>,
          affine_map<(b, s, n, k) -> (k, n)>,
          affine_map<(b, s, n, k) -> (b, s, n)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
      } ins(%lhs_static, %rhs : tensor<4x128x256xf32>, tensor<256x256xf32>)
        outs(%filled : tensor<4x128x256xf32>) {
      ^bb0(%a: f32, %b_val: f32, %c: f32):
        %mul = arith.mulf %a, %b_val : f32
        %add = arith.addf %c, %mul : f32
        linalg.yield %add : f32
      } -> tensor<4x128x256xf32>
      %static_dynamic = tensor.cast %static
        : tensor<4x128x256xf32> to tensor<?x?x256xf32>
      scf.yield %static_dynamic : tensor<?x?x256xf32>
    } else {
      %init = tensor.empty(%b, %s) : tensor<?x?x256xf32>
      %filled = linalg.fill ins(%zero : f32)
        outs(%init : tensor<?x?x256xf32>) -> tensor<?x?x256xf32>
      %dynamic = linalg.generic {
        indexing_maps = [
          affine_map<(b, s, n, k) -> (b, s, k)>,
          affine_map<(b, s, n, k) -> (k, n)>,
          affine_map<(b, s, n, k) -> (b, s, n)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
      } ins(%lhs, %rhs : tensor<?x?x256xf32>, tensor<256x256xf32>)
        outs(%filled : tensor<?x?x256xf32>) {
      ^bb0(%a: f32, %b_val: f32, %c: f32):
        %mul = arith.mulf %a, %b_val : f32
        %add = arith.addf %c, %mul : f32
        linalg.yield %add : f32
      } -> tensor<?x?x256xf32>
      scf.yield %dynamic : tensor<?x?x256xf32>
    }
    return %out : tensor<?x?x256xf32>
  }
}
