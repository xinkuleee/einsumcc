#include "einsumcc/Dialect/TC/IR/TCDialect.h"
#include "einsumcc/Dialect/TC/IR/TCOps.h"

using namespace mlir;
using namespace mlir::einsumcc::tc;

#include "einsumcc/Dialect/TC/IR/TCOpsDialect.cpp.inc"

void TCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "einsumcc/Dialect/TC/IR/TCOps.cpp.inc"
      >();
}
