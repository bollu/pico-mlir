#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace cl = llvm::cl;

class ConstantOp : public mlir::Op<ConstantOp,
                     /// The ConstantOp takes zero inputs.
                     mlir::OpTrait::ZeroOperands,
                     /// The ConstantOp returns a single result.
                     mlir::OpTrait::OneResult,
                     /// The ConstantOp is pure and has no visible side-effects.
                     mlir::OpTrait::HasNoSideEffect> {

 public:
  /// Inherit the constructors from the base Op class.
  using Op::Op;

  /// Provide the unique name for this operation. MLIR will use this to register
  /// the operation and uniquely identify it throughout the system.
  static llvm::StringRef getOperationName() { return "toy.constant"; }

  /// Return the value of the constant by fetching it from the attribute.
  mlir::DenseElementsAttr getValue();

  /// Operations can provide additional verification beyond the traits they
  /// define. Here we will ensure that the specific invariants of the constant
  /// operation are upheld, for example the result type must be of TensorType.
  LogicalResult verify() {  return success(); }

  /// Provide an interface to build this operation from a set of input values.
  /// This interface is used by the builder to allow for easily generating
  /// instances of this operation:
  ///   mlir::OpBuilder::create<ConstantOp>(...)
  /// This method populates the given `state` that MLIR uses to create
  /// operations. This state is a collection of all of the discrete elements
  /// that an operation may contain.
  /// Build a constant with the given return type and `value` attribute.
  static void build(mlir::Builder *builder, mlir::OperationState &state,
                    mlir::Type result, mlir::DenseElementsAttr value);
  /// Build a constant and reuse the type from the given 'value'.
  static void build(mlir::Builder *builder, mlir::OperationState &state,
                    mlir::DenseElementsAttr value);
  /// Build a constant by broadcasting the given 'value'.
  static void build(mlir::Builder *builder, mlir::OperationState &state,
                    double value);
};



/// This is the definition of the Toy dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types (in its
/// constructor). It can also override some general behavior exposed via virtual
/// methods, which will be demonstrated in later chapters of the tutorial.
class ToyDialect : public mlir::Dialect {
 public:
	 explicit ToyDialect(mlir::MLIRContext *ctx)
		 : mlir::Dialect(getDialectNamespace(), ctx) {
			 // addOperations<ConstantOp>();
		 }

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities.
  static llvm::StringRef getDialectNamespace() { return "toy"; }
};


int main() { 
    mlir::registerDialect<ToyDialect>();
    return 0;
}
