#include "Dialect/Lean/LeanOps.h"
#include "Dialect/Lean/LeanDialect.h"
#include "Dialect/Lean/LeanTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
// #include <bits/stdint-intn.h>
#include <cstddef>
#include <functional>
#include <iterator>

using namespace mlir;

namespace mlir {
namespace lean {
#define GET_OP_CLASSES
#include "Dialect/Lean/LeanOps.cpp.inc"
} // namespace lean
} // namespace mlir
