#ifndef LEAN_OPS
#define LEAN_OPS

include "mlir/Interfaces/SideEffects.td"
include "mlir/Interfaces/CallInterfaces.td"

//===----------------------------------------------------------------------===//
// Lean dialect definition
//===----------------------------------------------------------------------===//

def Lean_Dialect : Dialect {
    let name = "lean";

    let description = [{
        A lean dialect in MLIR.
    }];

    let cppNamespace = "lean";
}

//===----------------------------------------------------------------------===//
// Lean type definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Lean op definition
//===----------------------------------------------------------------------===//

// Base class for all Lean ops.
// class Lean_Op<string mnemonic, list<OpTrait> traits = []> :
//     Op<Lean_Dialect, mnemonic, traits> {

//     // For each Lean op, the following static functions need to be defined in
//     // LeanOps.cpp:
//     //
//     // * static ParseResult parse<op-c++-class-name>(OpAsmParser &parser,
//     //                                               OperationState &state);
//     // * static void print(OpAsmPrinter &p, <op-c++-class-name> op)
//     // * static LogicalResult verify(<op-c++-class-name> op)
//     let parser = [{ return ::parse$cppClass(parser, result); }];
//     let printer = [{ ::print(*this, p); }];
//     let verifier = [{ return ::verify(*this); }];
// }

class Lean_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Lean_Dialect, mnemonic, traits>;

// Provide a definition for the Toy StructType for use in ODS. This allows for
// using StructType in a similar way to Tensor or MemRef.
def Lean_StructType :
    Type<CPred<"$_self.isa<StructType>()">, "Toy struct type">;

// Provide a definition of the types that are used within the Toy dialect.
def Lean_Type : AnyTypeOf<[F64Tensor, Lean_StructType]>;



#endif // LEAN_OPS
