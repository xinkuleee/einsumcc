#ifndef EINSUMCC_DIALECT_TC_IR_TCOPS_TD
#define EINSUMCC_DIALECT_TC_IR_TCOPS_TD

include "einsumcc/Dialect/TC/IR/TCDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TC_Op<string mnemonic, list<Trait> traits = []>
    : Op<TC_Dialect, mnemonic, traits>;

def TC_ContractOp : TC_Op<"contract", [Pure]> {
  let summary = "A statically-shaped binary tensor contraction";
  let description = [{
    `tc.contract` represents the Nano v1 binary Einstein contraction. The
    indexing maps map a shared loop domain to the left input, right input, and
    result dimensions respectively. Iterator types are strings containing one
    `parallel` or `reduction` entry per loop dimension.

    The operation owns zero-initialization semantically, so it has no output
    accumulator operand. Lowering materializes that initialization explicitly.
    The optional `equation` attribute is checked against the indexing maps when
    present; it is not unchecked display metadata. Nano v1 accepts only dense,
    unencoded tensor types.

    Example:

    ```mlir
    %result = tc.contract %lhs, %rhs {
      equation = "mk,kn->mn",
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    AffineMapArrayAttr:$indexing_maps,
    StrArrayAttr:$iterator_types,
    OptionalAttr<StrAttr>:$equation
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

#endif // EINSUMCC_DIALECT_TC_IR_TCOPS_TD
