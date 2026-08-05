add_mlir_conversion_library(EinsumCCTCToLinalg
  TCToLinalg.cpp

  LINK_LIBS PUBLIC
  EinsumCCTCDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
)
