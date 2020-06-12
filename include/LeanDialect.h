#ifndef MLIR_DIALECT_LEAN_LEANDIALECT_H
#define MLIR_DIALECT_LEAN_LEANDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace lean {

class LeanDialect : public Dialect {
public:
  explicit LeanDialect(MLIRContext *context);


  // static StringRef getLeanFunctionAttrName() { return "lean.function"; }
  // static StringRef getLeanProgramAttrName() { return "lean.program"; }

  // static StringRef getFieldTypeName() { return "field"; }
  // static StringRef getViewTypeName() { return "view"; }

  // funky action-at-a-distance at play here!
  // static bool isLeanFunction(FuncOp funcOp) {
  //   return !!funcOp.getAttr(getLeanFunctionAttrName());
  // }
  // static bool isLeanProgram(FuncOp funcOp) {
  //   return !!funcOp.getAttr(getLeanProgramAttrName());
  // }

  /// Parses a type registered to this dialect
  Type parseType(DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect
  void printType(Type type, DialectAsmPrinter &os) const override;
  
  /// Returns the prefix used in the textual IR to refer to lean operations
  static StringRef getDialectNamespace() { return "lean"; }


};

namespace detail {
struct StructTypeStorage;
} // end namespace detail

/// Create a local enumeration with all of the types that are defined by Toy.
namespace LeanTypes {
  enum Types {
    Struct = mlir::Type::FIRST_TOY_TYPE,
    Simple,
  };
} // end namespace ToyTypes


/// This class defines the Toy struct type. It represents a collection of
/// element types. All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructType), the base class to use (Type), and the storage class
/// (StructTypeStorage).
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               detail::StructTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static bool kindof(unsigned kind) { return kind == LeanTypes::Struct; }

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be atleast one element type.
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes);

  /// Returns the element types of this struct type.
  llvm::ArrayRef<mlir::Type> getElementTypes();

  /// Returns the number of element type held by this struct.
  size_t getNumElementTypes() { return getElementTypes().size(); }
};

/// This class defines a simple parameterless type. All derived types must
/// inherit from the CRTP class 'Type::TypeBase'. It takes as template
/// parameters the concrete type (SimpleType), and the base class to use (Type).
/// 'Type::TypeBase' also provides several utility methods to simplify type
/// construction.
class SimpleType : public Type::TypeBase<SimpleType, Type> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static bool kindof(unsigned kind) { return kind == LeanTypes::Simple; }

  /// This method is used to get an instance of the 'SimpleType'. Given that
  /// this is a parameterless type, it just needs to take the context for
  /// uniquing purposes.
  static SimpleType get(MLIRContext *context) {
    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type.
    return Base::get(context, LeanTypes::Simple);
  }
};




} // end namespace lean
} // end namespace mlir

#endif // MLIR_DIALECT_LEAN_LEANDIALECT_H
