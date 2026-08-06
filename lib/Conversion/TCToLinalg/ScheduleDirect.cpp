#include "einsumcc/Conversion/TCToLinalg/TCToLinalg.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::einsumcc {
namespace {

constexpr StringLiteral kScheduleRoot = "einsumcc.schedule_root";

class ScheduleDirectPass final
    : public PassWrapper<ScheduleDirectPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleDirectPass)
  ScheduleDirectPass() = default;
  ScheduleDirectPass(const ScheduleDirectPass &other)
      : PassWrapper(other) {
    tileSizes = ArrayRef<int64_t>(other.tileSizes);
  }
  explicit ScheduleDirectPass(ArrayRef<int64_t> sizes) : ScheduleDirectPass() {
    tileSizes = sizes;
  }

  StringRef getArgument() const final { return "einsumcc-schedule-direct"; }
  StringRef getDescription() const final {
    return "Tile the marked Direct linalg contraction with an explicit schedule";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    SmallVector<int64_t> sizes(tileSizes.begin(), tileSizes.end());
    if (sizes.empty()) {
      getOperation().emitError("Direct scheduling requires tile-sizes");
      return signalPassFailure();
    }
    if (llvm::any_of(sizes, [](int64_t value) { return value < 0; })) {
      getOperation().emitError("Direct tile sizes must be non-negative");
      return signalPassFailure();
    }
    if (llvm::all_of(sizes, [](int64_t value) { return value == 0; })) {
      getOperation().emitError("Direct schedule must tile at least one loop");
      return signalPassFailure();
    }

    SmallVector<linalg::GenericOp> roots;
    getOperation().walk([&](linalg::GenericOp op) {
      if (op->hasAttr(kScheduleRoot))
        roots.push_back(op);
    });
    if (roots.empty()) {
      getOperation().emitError(
          "Direct scheduling found no marked contraction");
      return signalPassFailure();
    }

    IRRewriter rewriter(&getContext());
    for (linalg::GenericOp root : roots) {
      if (sizes.size() != root.getNumLoops()) {
        root.emitError() << "expected " << root.getNumLoops()
                         << " tile sizes, got " << sizes.size();
        signalPassFailure();
        return;
      }

      rewriter.setInsertionPoint(root);
      linalg::LinalgTilingOptions options;
      options.setTileSizes(sizes).setLoopType(
          linalg::LinalgTilingLoopType::Loops);
      FailureOr<linalg::TiledLinalgOp> tiled =
          linalg::tileLinalgOp(rewriter, root, options);
      if (failed(tiled)) {
        root.emitError("failed to apply Direct tile schedule");
        signalPassFailure();
        return;
      }
      tiled->op->removeAttr(kScheduleRoot);
      if (tiled->tensorResults.empty())
        rewriter.eraseOp(root);
      else
        rewriter.replaceOp(root, tiled->tensorResults);
    }
  }

private:
  ListOption<int64_t> tileSizes{
      *this, "tile-sizes",
      llvm::cl::desc(
          "Comma-separated tile sizes in output-plus-reduction loop order; "
          "zero leaves a loop untiled"),
      llvm::cl::ZeroOrMore};
};

} // namespace

std::unique_ptr<Pass> createScheduleDirectPass(ArrayRef<int64_t> tileSizes) {
  return std::make_unique<ScheduleDirectPass>(tileSizes);
}

void registerScheduleDirectPass() { PassRegistration<ScheduleDirectPass>(); }

} // namespace mlir::einsumcc
