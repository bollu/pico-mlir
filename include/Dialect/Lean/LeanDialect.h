#ifndef MLIR_DIALECT_LEAN_LEANDIALECT_H
#define MLIR_DIALECT_LEAN_LEANDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"

namespace mlir {
namespace lean {

// Constant used to mark unused dimensions of lower dimensional fields
constexpr static int64_t kIgnoreDimension = std::numeric_limits<int64_t>::min();

// Constant dimension identifiers
constexpr static int kIDimension = 0;
constexpr static int kJDimension = 1;
constexpr static int kKDimension = 2;

class LeanDialect : public Dialect {
public:
  explicit LeanDialect(MLIRContext *context);

  /// Returns the prefix used in the textual IR to refer to lean operations
  static StringRef getDialectNamespace() { return "lean"; }

  static StringRef getLeanFunctionAttrName() { return "lean.function"; }
  static StringRef getLeanProgramAttrName() { return "lean.program"; }

  static StringRef getFieldTypeName() { return "field"; }
  static StringRef getViewTypeName() { return "view"; }

  static bool isLeanFunction(FuncOp funcOp) {
    return !!funcOp.getAttr(getLeanFunctionAttrName());
  }
  static bool isLeanProgram(FuncOp funcOp) {
    return !!funcOp.getAttr(getLeanProgramAttrName());
  }

  /// Parses a type registered to this dialect
  Type parseType(DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect
  void printType(Type type, DialectAsmPrinter &os) const override;
};

} // namespace lean
} // namespace mlir

#endif // MLIR_DIALECT_LEAN_LEANDIALECT_H
