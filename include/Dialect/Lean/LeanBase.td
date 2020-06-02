#ifndef MLIR_DIALECT_LEAN_LEANBASE
#define MLIR_DIALECT_LEAN_LEANBASE

include "mlir/Interfaces/SideEffects.td"

//===----------------------------------------------------------------------===//
// Lean dialect definition
//===----------------------------------------------------------------------===//

def Lean_Dialect : Dialect {
    let name = "Lean";

    let description = [{
        A lean dialect in MLIR.
    }];

    let cppNamespace = "lean";
}

//===----------------------------------------------------------------------===//
// Lean type definitions
//===----------------------------------------------------------------------===//

def Lean_IsViewType : CPred<"$_self.isa<::mlir::lean::ViewType>()">;
def Lean_IsFieldType : CPred<"$_self.isa<::mlir::lean::FieldType>()">;

def Lean_View : Type<Lean_IsViewType, "a view on a field">;
def Lean_Field : Type<Lean_IsFieldType, "a field">;
def Lean_ElementType : AnyTypeOf<[F32, F64]>;

//===----------------------------------------------------------------------===//
// Lean op definition
//===----------------------------------------------------------------------===//

// Base class for all Lean ops.
class Lean_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Lean_Dialect, mnemonic, traits> {

    // For each Lean op, the following static functions need to be defined in
    // LeanOps.cpp:
    //
    // * static ParseResult parse<op-c++-class-name>(OpAsmParser &parser,
    //                                               OperationState &state);
    // * static void print(OpAsmPrinter &p, <op-c++-class-name> op)
    // * static LogicalResult verify(<op-c++-class-name> op)
    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ ::print(*this, p); }];
    let verifier = [{ return ::verify(*this); }];
}

#endif //  MLIR_DIALECT_LEAN_LEANBASE
