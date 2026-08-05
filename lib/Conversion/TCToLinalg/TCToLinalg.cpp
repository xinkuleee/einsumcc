#include "einsumcc/Conversion/TCToLinalg/TCToLinalg.h"

#include "einsumcc/Dialect/TC/IR/TCDialect.h"
#include "einsumcc/Dialect/TC/IR/TCOps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::einsumcc {
namespace {

class ContractLowering final : public OpRewritePattern<tc::ContractOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tc::ContractOp op,
                                PatternRewriter &rewriter) const override {
    Location location = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    auto empty = rewriter.create<tensor::EmptyOp>(
        location, resultType.getShape(), resultType.getElementType());
    auto zero = rewriter.create<arith::ConstantFloatOp>(
        location, rewriter.getF32Type(), APFloat(0.0f));
    auto init = rewriter.create<linalg::FillOp>(
        location, TypeRange{resultType}, ValueRange{zero}, ValueRange{empty});

    SmallVector<AffineMap> maps;
    for (Attribute attribute : op.getIndexingMaps())
      maps.push_back(cast<AffineMapAttr>(attribute).getValue());

    SmallVector<utils::IteratorType> iteratorTypes;
    for (Attribute attribute : op.getIteratorTypes()) {
      StringRef value = cast<StringAttr>(attribute).getValue();
      iteratorTypes.push_back(value == "parallel"
                                  ? utils::IteratorType::parallel
                                  : utils::IteratorType::reduction);
    }

    auto generic = rewriter.create<linalg::GenericOp>(
        location, TypeRange{resultType}, ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{init.getResult(0)}, maps, iteratorTypes,
        [&](OpBuilder &builder, Location nestedLocation, ValueRange arguments) {
          Value product = builder.create<arith::MulFOp>(
              nestedLocation, arguments[0], arguments[1]);
          Value sum = builder.create<arith::AddFOp>(
              nestedLocation, arguments[2], product);
          builder.create<linalg::YieldOp>(nestedLocation, sum);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

class TCContractToLinalgPass final
    : public PassWrapper<TCContractToLinalgPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TCContractToLinalgPass)

  StringRef getArgument() const final { return "tc-contract-to-linalg"; }
  StringRef getDescription() const final {
    return "Lower tc.contract to initialized linalg.generic";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addIllegalOp<tc::ContractOp>();
    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect,
                           tensor::TensorDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(context);
    patterns.add<ContractLowering>(context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTCContractToLinalgPass() {
  return std::make_unique<TCContractToLinalgPass>();
}

void registerTCContractToLinalgPass() {
  PassRegistration<TCContractToLinalgPass>();
}

} // namespace mlir::einsumcc
