cmake_minimum_required(VERSION 3.4.3)
project(musq C CXX)

# Colors when compiling with clang
# set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always -fno-rtti -g -Wall -Werror")
set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always -fno-rtti -g -Wall")

# LLVM
find_package(LLVM REQUIRED)
set(LLVM_LINK_COMPONENTS Support)


# INCLUDES AND SOURCES
include_directories(include ${LLVM_INCLUDE_DIRS})

add_executable(musq src/main.cpp)

target_link_libraries(musq ${LLVM_LIBS} -lstdc++ MLIRAnalysis MLIRIR MLIRParser MLIRTransforms)
set_target_properties(musq PROPERTIES CXX_STANDARD 14)
