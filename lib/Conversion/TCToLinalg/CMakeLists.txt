add_mlir_conversion_library(EinsumCCTCToLinalg
  ScheduleDirect.cpp
  TCToLinalg.cpp

  LINK_LIBS PUBLIC
  EinsumCCTCDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
)
