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
