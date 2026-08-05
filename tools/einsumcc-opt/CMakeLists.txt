add_llvm_executable(einsumcc-opt
  einsumcc-opt.cpp
)

llvm_update_compile_flags(einsumcc-opt)
target_link_libraries(einsumcc-opt PRIVATE
  EinsumCCTCDialect
  EinsumCCTCToLinalg
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMlirOptMain
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTensorDialect
)
mlir_check_all_link_libraries(einsumcc-opt)
