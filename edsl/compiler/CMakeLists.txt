find_package(LLVM CONFIG REQUIRED)
find_package(MLIR CONFIG REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${LLVM_CMAKE_DIR} ${MLIR_CMAKE_DIR})
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(PROJ_GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(PROJ_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${PROJ_INCLUDE_DIR} ${PROJ_GENERATED_INCLUDE_DIR})

add_subdirectory(lib)

find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(edsl_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/binding/binding.cpp
)

target_link_libraries(edsl_cpp PUBLIC
  LLVMObject
  LLVMCore
  LLVMSupport
  LLVMOption
  LLVMCodeGen
  LLVMOrcJIT
)

target_link_libraries(edsl_cpp PUBLIC
  ${dialect_libs}
  ${extension_libs}
  MLIREdslEntry
)
