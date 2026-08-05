// RUN: einsumcc-opt %s | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: einsumcc-opt %s --tc-contract-to-linalg | FileCheck %s --check-prefix=LOWER

module {
  func.func @matmul(%lhs: tensor<3x4xf32>, %rhs: tensor<4x5xf32>)
      -> tensor<3x5xf32> {
    %result = tc.contract %lhs, %rhs {
      equation = "mk,kn->mn",
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

// ROUNDTRIP: tc.contract
// LOWER-NOT: tc.contract
// LOWER: tensor.empty
// LOWER: linalg.fill
// LOWER: linalg.generic
// LOWER: arith.mulf
// LOWER: arith.addf
