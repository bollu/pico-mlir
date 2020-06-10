//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "Dialect/Lean/LeanDialect.h"
#include "Dialect/Lean/Passes.h"
#include "mlir/Analysis/Passes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("XXXXOutput filenameXXXXX"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

// static cl::opt<bool>
//     splitInputFile("split-input-file",
//                    cl::desc("Split the input file into pieces and process each "
//                             "chunk independently"),
//                    cl::init(false));

// static cl::opt<bool>
//     verifyDiagnostics("verify-diagnostics",
//                       cl::desc("Check that emitted diagnostics match "
//                                "expected-* lines on the corresponding line"),
//                       cl::init(false));

// static cl::opt<bool>
//     verifyPasses("verify-each",
//                  cl::desc("Run the verifier after each transformation pass"),
//                  cl::init(true));

int loadMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return -1;
  }

  outs() << "input file contents:\nvvvvv\n" << (*fileOrErr)->getBuffer() << "^^^^\n";

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file |" << inputFilename << "| \n";
    return 3;
  }
  return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningModuleRef &module) {
  if (int error = loadMLIR(context, module))
    return error;

  mlir::PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  // Check to see what granularity of MLIR we are compiling to.
  bool isLoweringToAffine = false; //emitAction >= Action::DumpMLIRAffine;
  bool isLoweringToLLVM = false; // emitAction >= Action::DumpMLIRLLVM;

  if (isLoweringToAffine) {
    // Inline all functions into main and then delete them.
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createSymbolDCEPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations.
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    // optPM.addPass(mlir::toy::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (isLoweringToAffine) {
    // Partially lower the toy dialect with a few cleanups afterwards.
    // pm.addPass(mlir::toy::createLowerToAffinePass());

    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    optPM.addPass(mlir::createLoopFusionPass());
    optPM.addPass(mlir::createMemRefDataFlowOptPass());

  }


  if (mlir::failed(pm.run(*module)))
    return 4;
  return 0;
}

int main(int argc, char **argv) {
  // Regster all MLIR dialects and passes
  registerAllDialects();
  registerAllPasses();

  // Register the lean dialect
  registerDialect<lean::LeanDialect>();

  // Register the lean passes

  // Initialize LLVM
  InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  registerPassManagerCLOptions();
  PassPipelineCLParser passPipeline("", "Compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR modular optimizer driver\n");

  
  errs() << "parsed input!\n";

  // mlir::registerDialect<mlir::lean::LeanDialect>();

  mlir::MLIRContext context;
  mlir::OwningModuleRef module;
  if (int error = loadAndProcessMLIR(context, module)) {
    return error;
  }

  outs() << "dumping module:\n";
  module->dump();

  /*
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  errs() << "writing output to: |" << outputFilename << "|\n";

  errs () << "file...:vvvvv\n" << file->getBuffer() << "^^^^^\n";

  // assert(false && "wrote output file");

  errs() << "running MLIR...\n";


  mlir::MLIRContext context;
  mlir::OwningModuleRef module;
  if (int error = loadAndProcessMLIR(context, module))
    return error;

  // If we aren't exporting to non-mlir, then we are done.
  bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
  if (isOutputingMLIR) {
    module->dump();
    return 0;
  }
  */

  // return failed(MlirOptMain(errs(), std::move(file), passPipeline,
  //                           splitInputFile, verifyDiagnostics, verifyPasses));

  // return failed(MlirOptMain(output->os(), std::move(file), passPipeline,
  //                           splitInputFile, verifyDiagnostics, verifyPasses));
}
