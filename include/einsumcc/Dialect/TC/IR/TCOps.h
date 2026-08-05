#ifndef EINSUMCC_DIALECT_TC_IR_TCOPS_H
#define EINSUMCC_DIALECT_TC_IR_TCOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "einsumcc/Dialect/TC/IR/TCOps.h.inc"

#endif // EINSUMCC_DIALECT_TC_IR_TCOPS_H
