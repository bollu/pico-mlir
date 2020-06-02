#include "Dialect/Lean/LeanDialect.h"
#include "Dialect/Lean/LeanOps.h"
#include "Dialect/Lean/LeanTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
// #include <bits/stdint-intn.h>
#include <cstddef>
#include <string>

using namespace mlir;
using namespace mlir::lean;

//===----------------------------------------------------------------------===//
// Lean Dialect
//===----------------------------------------------------------------------===//

LeanDialect::LeanDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {

  addOperations<
#define GET_OP_LIST
#include "Dialect/Lean/LeanOps.cpp.inc"
      >();

  // Allow Lean operations to exist in their generic form
  allowUnknownOperations();
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

Type LeanDialect::parseType(DialectAsmParser &parser) const {
  parser.emitError(parser.getNameLoc(), "unknown lean type: ")
      << parser.getFullSymbolSpec();
  return Type();
}

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

void LeanDialect::printType(Type type, DialectAsmPrinter &printer) const {
  switch (type.getKind()) {
  default:
    llvm_unreachable("unhandled lean type");
  }
}
