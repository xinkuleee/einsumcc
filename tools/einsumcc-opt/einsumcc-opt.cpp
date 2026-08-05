#include "einsumcc/Conversion/TCToLinalg/TCToLinalg.h"
#include "einsumcc/Dialect/TC/IR/TCDialect.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::einsumcc::tc::TCDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect>();
  mlir::bufferization::registerBufferizationPasses();
  mlir::registerLinalgPasses();
  mlir::registerArithToLLVMConversionPass();
  mlir::registerConvertControlFlowToLLVMPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerConvertIndexToLLVMPass();
  mlir::registerFinalizeMemRefToLLVMConversionPass();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerSCFToControlFlowPass();
  mlir::einsumcc::registerTCContractToLinalgPass();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "EinsumCC optimizer", registry));
}
