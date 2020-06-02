#ifndef MLIR_DIALECT_LEAN_PASSES_H
#define MLIR_DIALECT_LEAN_PASSES_H

#include "Dialect/Lean/LeanOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace lean {

std::unique_ptr<OpPassBase<ModuleOp>> createCallInliningPass();

std::unique_ptr<OpPassBase<FuncOp>> createLeanInliningPass();

std::unique_ptr<OpPassBase<FuncOp>> createShapeShiftPass();

std::unique_ptr<OpPassBase<FuncOp>> createShapeInferencePass();

std::unique_ptr<OpPassBase<FuncOp>> createLeanUnrollingPass();

} // namespace lean
} // namespace mlir

#endif // MLIR_DIALECT_LEAN_PASSES_H
