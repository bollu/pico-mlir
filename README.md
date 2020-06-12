# A compiler for a small lambda calculus written using MLIR.
- ![mu-squared](https://wikimedia.org/api/rest_v1/media/math/render/svg/68b31ff8082976980ee460521d36b8a30f9603a2)
- [Lioville function](https://en.wikipedia.org/wiki/Liouville_function)

### Good resources:

- [SPIR-V dialect spec](https://github.com/bollu/mlir/blob/master/orig_docs/Dialects/SPIR-V.md)
- [SPIR-V dialect base tablegen](https://github.com/bollu/mlir/blob/master/include/mlir/Dialect/SPIRV/SPIRVBase.td)
- [SPIR-V dialect header](https://github.com/bollu/mlir/blob/master/include/mlir/Dialect/SPIRV/SPIRVDialect.h)
- [SPIR-V dialect cpp](https://github.com/bollu/mlir/blob/master/lib/Dialect/SPIRV/SPIRVDialect.cpp)

### Round tripping

- What stops someone from defining a printer that is completely different
  from what the parser wants?

### TODO

- [ ] Parse `case x of { ... }` style input.
- [ ] Parse `let var = val in ...` input.
- [ ] Add `PrimInt` type.
- [ ] Get factorial up and running.
- [ ] Get a notion of explicit GC. 