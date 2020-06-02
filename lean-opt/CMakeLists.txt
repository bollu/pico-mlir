set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

set(LIBS
  MLIRAffineOps
  MLIRFxpMathOps
  MLIRGPU
  MLIRLinalgAnalysis
  MLIRLinalgEDSC
  MLIRLinalgOps
  MLIRLinalgTransforms
  MLIRLinalgUtils
  # MLIRLLVMIRTransforms
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRROCDLIR
  MLIRLoopOps
  MLIRLoopOpsTransforms
  MLIROpenMP
  MLIRQuantOps
  MLIRSDBM
  # MLIRShape
  MLIRSPIRV
  MLIRSPIRVSerialization
  MLIRSPIRVTransforms
  MLIRStandardOps
  # MLIRVector

  MLIRAffineToStandard
  MLIRGPUtoCUDATransforms
  MLIRGPUtoNVVMTransforms
  MLIRGPUtoROCDLTransforms
  MLIRGPUtoSPIRVTransforms
  MLIRGPUtoVulkanTransforms
  MLIRLinalgToLLVM
  MLIRLinalgToSPIRVTransforms
  MLIRLoopsToGPU
  MLIRLoopToStandard
  MLIRStandardToLLVM
  MLIRStandardToSPIRVTransforms
  MLIRVectorToLLVM
  MLIRVectorToLoops

  MLIRLoopOpsTransforms
  MLIRLoopAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIREDSC
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRQuantizerFxpMathConfig
  MLIRQuantizerSupport
  MLIRQuantizerTransforms
  MLIRSPIRV
  MLIRSPIRVTestPasses
  MLIRSPIRVTransforms
  MLIRTransforms
  MLIRTransformUtils
  MLIRTestDialect
  MLIRTestIR
  MLIRTestPass
  MLIRTestTransforms
  MLIRSupport
  MLIRIR
  MLIROptLib
  LLVMSupport
  LLVMCore
  LLVMAsmParser

  MLIRLean
)
add_llvm_executable(lean-opt
 lean-opt.cpp
)

# Manually expand the target library, since our MLIR libraries
# aren't plugged into the LLVM dependency tracking. If we don't
# do this then we can't insert the CodeGen library after ourselves
llvm_expand_pseudo_components(TARGET_LIBS AllTargetsCodeGens)
# Prepend LLVM in front of every target, this is how the library
# are named with CMake
SET(targets_to_link)
FOREACH(t ${TARGET_LIBS})
  LIST(APPEND targets_to_link "LLVM${t}")
ENDFOREACH(t)

llvm_update_compile_flags(lean-opt)
target_link_libraries(lean-opt PRIVATE ${LIBS} ${targets_to_link})
