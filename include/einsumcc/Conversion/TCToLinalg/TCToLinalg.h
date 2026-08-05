#ifndef EINSUMCC_CONVERSION_TCTOLINALG_TCTOLINALG_H
#define EINSUMCC_CONVERSION_TCTOLINALG_TCTOLINALG_H

#include <memory>

namespace mlir {
class Pass;

namespace einsumcc {

std::unique_ptr<Pass> createTCContractToLinalgPass();
void registerTCContractToLinalgPass();

} // namespace einsumcc
} // namespace mlir

#endif // EINSUMCC_CONVERSION_TCTOLINALG_TCTOLINALG_H
