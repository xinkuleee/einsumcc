// RUN: not einsumcc-opt %s -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @implicit_equation(
      %lhs: tensor<3x4xf32>, %rhs: tensor<4x5xf32>) -> tensor<3x5xf32> {
    %result = tc.contract %lhs, %rhs {
      equation = "mk,kn",
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
    return %result : tensor<3x5xf32>
  }
}

// CHECK: equation must contain exactly one explicit '->'
