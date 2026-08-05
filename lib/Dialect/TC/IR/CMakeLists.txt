add_mlir_dialect_library(EinsumCCTCDialect
  TCDialect.cpp
  TCOps.cpp

  DEPENDS
  MLIRTCOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
