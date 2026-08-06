#ifndef EINSUMCC_CONVERSION_TCTOLINALG_TCTOLINALG_H
#define EINSUMCC_CONVERSION_TCTOLINALG_TCTOLINALG_H

#include <cstdint>
#include <memory>

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class Pass;

namespace einsumcc {

std::unique_ptr<Pass> createTCContractToLinalgPass();
void registerTCContractToLinalgPass();

std::unique_ptr<Pass> createScheduleDirectPass(
    llvm::ArrayRef<int64_t> tileSizes = {});
void registerScheduleDirectPass();

} // namespace einsumcc
} // namespace mlir

#endif // EINSUMCC_CONVERSION_TCTOLINALG_TCTOLINALG_H
