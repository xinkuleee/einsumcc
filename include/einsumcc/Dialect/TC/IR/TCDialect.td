#ifndef EINSUMCC_DIALECT_TC_IR_TCDIALECT_TD
#define EINSUMCC_DIALECT_TC_IR_TCDIALECT_TD

include "mlir/IR/OpBase.td"

def TC_Dialect : Dialect {
  let name = "tc";
  let cppNamespace = "::mlir::einsumcc::tc";
  let summary = "Target-independent tensor contraction dialect";
  let description = [{
    The TC dialect is the semantic boundary between EinsumCC's Einstein
    notation frontend and target-specific scheduling. Its contract operation
    records the loop/index relationship without committing to CPU or GPU code.
  }];
}

#endif // EINSUMCC_DIALECT_TC_IR_TCDIALECT_TD
