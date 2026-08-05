// RUN: not einsumcc-opt %s -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @bad_broadcast(%lhs: tensor<3x4xf32>, %rhs: tensor<4x5xf32>)
      -> tensor<3x5xf32> {
    %result = tc.contract %lhs, %rhs {
      indexing_maps = [
        affine_map<(m, n, k, q) -> (m, k)>,
        affine_map<(m, n, k, q) -> (k, n)>,
        affine_map<(m, n, k, q) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction", "parallel"]
    } : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
    return %result : tensor<3x5xf32>
  }
}

// CHECK: contains a loop dimension unused by every indexing map
