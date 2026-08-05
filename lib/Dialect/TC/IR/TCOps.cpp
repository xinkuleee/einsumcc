#include "einsumcc/Dialect/TC/IR/TCOps.h"

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;
using namespace mlir::einsumcc::tc;

namespace {

LogicalResult verifyMap(ContractOp op, AffineMap map, RankedTensorType type,
                        StringRef role, unsigned domainRank,
                        llvm::SmallBitVector &usedDims) {
  if (map.getNumSymbols() != 0)
    return op.emitOpError() << role << " indexing map must not use symbols";
  if (map.getNumDims() != domainRank)
    return op.emitOpError()
           << role << " indexing map has " << map.getNumDims()
           << " domain dimensions, expected " << domainRank;
  if (map.getNumResults() != static_cast<unsigned>(type.getRank()))
    return op.emitOpError()
           << role << " indexing map has " << map.getNumResults()
           << " results, but its tensor has rank " << type.getRank();
  if (!map.isProjectedPermutation(/*allowZeroInResults=*/false))
    return op.emitOpError()
           << role << " indexing map must be a projected permutation";

  for (AffineExpr expression : map.getResults()) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension)
      return op.emitOpError()
             << role << " indexing map results must be loop dimensions";
    usedDims.set(dimension.getPosition());
  }
  return success();
}

} // namespace

LogicalResult ContractOp::verify() {
  auto lhsType = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(getRhs().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!lhsType || !rhsType || !resultType)
    return emitOpError("requires ranked tensor operands and result");

  if (lhsType.getRank() < 1 || lhsType.getRank() > 6 || rhsType.getRank() < 1 ||
      rhsType.getRank() > 6)
    return emitOpError("requires each input rank to be between 1 and 6");

  Type elementType = lhsType.getElementType();
  if (!elementType.isF32() || rhsType.getElementType() != elementType ||
      resultType.getElementType() != elementType)
    return emitOpError("requires f32 operands and result");
  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !resultType.hasStaticShape())
    return emitOpError("requires static tensor shapes");
  auto hasNonPositiveExtent = [](RankedTensorType type) {
    return llvm::any_of(type.getShape(),
                        [](int64_t extent) { return extent <= 0; });
  };
  if (hasNonPositiveExtent(lhsType) || hasNonPositiveExtent(rhsType) ||
      hasNonPositiveExtent(resultType))
    return emitOpError("requires positive tensor dimensions");

  ArrayAttr maps = getIndexingMaps();
  if (maps.size() != 3)
    return emitOpError("requires exactly three indexing maps (lhs, rhs, result)");
  auto lhsMap = cast<AffineMapAttr>(maps[0]).getValue();
  auto rhsMap = cast<AffineMapAttr>(maps[1]).getValue();
  auto resultMap = cast<AffineMapAttr>(maps[2]).getValue();
  const unsigned domainRank = lhsMap.getNumDims();

  ArrayAttr iteratorTypes = getIteratorTypes();
  if (iteratorTypes.size() != domainRank)
    return emitOpError() << "has " << iteratorTypes.size()
                         << " iterator types for " << domainRank
                         << " loop dimensions";

  llvm::SmallBitVector parallelDims(domainRank);
  llvm::SmallBitVector reductionDims(domainRank);
  for (auto [index, attribute] : llvm::enumerate(iteratorTypes)) {
    StringRef iterator = cast<StringAttr>(attribute).getValue();
    if (iterator == "parallel")
      parallelDims.set(index);
    else if (iterator == "reduction")
      reductionDims.set(index);
    else
      return emitOpError() << "iterator type #" << index
                           << " must be 'parallel' or 'reduction', got '"
                           << iterator << "'";
  }
  if (reductionDims.none())
    return emitOpError("requires at least one reduction iterator");

  llvm::SmallBitVector usedDims(domainRank);
  if (failed(verifyMap(*this, lhsMap, lhsType, "lhs", domainRank, usedDims)) ||
      failed(verifyMap(*this, rhsMap, rhsType, "rhs", domainRank, usedDims)) ||
      failed(verifyMap(*this, resultMap, resultType, "result", domainRank,
                       usedDims)))
    return failure();
  if (usedDims.count() != domainRank)
    return emitOpError("contains a loop dimension unused by every indexing map");

  llvm::SmallBitVector lhsDims(domainRank), rhsDims(domainRank), outputDims(domainRank);
  auto collectDims = [](AffineMap map, llvm::SmallBitVector &bits) {
    for (AffineExpr expression : map.getResults())
      bits.set(cast<AffineDimExpr>(expression).getPosition());
  };
  collectDims(lhsMap, lhsDims);
  collectDims(rhsMap, rhsDims);
  collectDims(resultMap, outputDims);

  if (outputDims != parallelDims)
    return emitOpError("result map dimensions must be exactly the parallel iterators");
  if ((reductionDims & lhsDims) != reductionDims ||
      (reductionDims & rhsDims) != reductionDims)
    return emitOpError("every reduction iterator must index both inputs");
  if ((parallelDims & (lhsDims | rhsDims)) != parallelDims)
    return emitOpError("every parallel iterator must index at least one input");

  // Nano v1 has four and only four index memberships. A parallel loop is
  // either B (both inputs and output), M (lhs and output), or N (rhs and
  // output); a reduction loop is K (both inputs and no output). Verifying the
  // exact membership here prevents accidental broadcasting semantics from
  // entering target lowering.
  for (unsigned loop = 0; loop < domainRank; ++loop) {
    bool inLhs = lhsDims.test(loop);
    bool inRhs = rhsDims.test(loop);
    bool inOutput = outputDims.test(loop);
    if (reductionDims.test(loop)) {
      if (!inLhs || !inRhs || inOutput)
        return emitOpError()
               << "reduction iterator #" << loop
               << " must index both inputs and not the result";
      continue;
    }
    if (!inOutput || (!inLhs && !inRhs))
      return emitOpError()
             << "parallel iterator #" << loop
             << " must index the result and at least one input";
  }

  SmallVector<int64_t> loopExtents(domainRank, ShapedType::kDynamic);
  auto mergeExtents = [&](AffineMap map, RankedTensorType type, StringRef role) {
    for (auto [dimension, expression] : llvm::enumerate(map.getResults())) {
      unsigned loop = cast<AffineDimExpr>(expression).getPosition();
      int64_t extent = type.getDimSize(dimension);
      if (loopExtents[loop] == ShapedType::kDynamic)
        loopExtents[loop] = extent;
      else if (loopExtents[loop] != extent) {
        emitOpError() << "loop dimension #" << loop << " has extent "
                      << loopExtents[loop] << " but " << role << " dimension #"
                      << dimension << " has extent " << extent;
        return failure();
      }
    }
    return success();
  };
  if (failed(mergeExtents(lhsMap, lhsType, "lhs")) ||
      failed(mergeExtents(rhsMap, rhsType, "rhs")) ||
      failed(mergeExtents(resultMap, resultType, "result")))
    return failure();

  return success();
}

#define GET_OP_CLASSES
#include "einsumcc/Dialect/TC/IR/TCOps.cpp.inc"
