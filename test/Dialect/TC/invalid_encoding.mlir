// RUN: not einsumcc-opt %s -o /dev/null 2>&1 | FileCheck %s

#layout = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

module {
  func.func @encoded_lhs(
      %lhs: tensor<3x4xf32, #layout>, %rhs: tensor<4x5xf32>)
      -> tensor<3x5xf32> {
    %result = tc.contract %lhs, %rhs {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } : (tensor<3x4xf32, #layout>, tensor<4x5xf32>) -> tensor<3x5xf32>
    return %result : tensor<3x5xf32>
  }
}

// CHECK: does not support encoded tensor types in Nano v1 lowering
