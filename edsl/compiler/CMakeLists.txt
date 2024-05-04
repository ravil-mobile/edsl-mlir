find_package(LLVM CONFIG REQUIRED)
find_package(MLIR CONFIG REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${LLVM_CMAKE_DIR} ${MLIR_CMAKE_DIR})
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddMLIRPython)
include(HandleLLVMOptions)

set(PROJ_GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(PROJ_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${PROJ_INCLUDE_DIR} ${PROJ_GENERATED_INCLUDE_DIR})

add_subdirectory(lib)

include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=edsl.")
set(EDSL_PYTHON_INSTALL_DIR "python_packages")
set(EDSL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(EDSL_PYTHON_PACKAGES_DIR "${EDSL_BINARY_DIR}/${EDSL_PYTHON_INSTALL_DIR}")

declare_mlir_python_sources(EDSLPythonSources ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
declare_mlir_python_sources(EDSLPythonExtensions)

get_property(MLIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(MLIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(MLIR_EXTENSION_LIBS GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(MLIR_TRANSLATION_LIBS GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(MLIR_UPSTREAM_CAPI_LIBS GLOBAL PROPERTY MLIR_UPSTREAM_CAPI_LIBS)

# Dig out all of the upstream cpp source files associated with the C extension modules.
_flatten_mlir_python_targets(mlir_python_sources MLIRPythonSources)
set(_mlir_python_cpp_sources)
foreach(s ${mlir_python_sources})
  get_target_property(_sources ${s} INTERFACE_SOURCES)
  list(FILTER _sources INCLUDE REGEX "(\.cpp|\.h)$")
  if(_sources)
    list(APPEND _mlir_python_cpp_sources ${_sources})
  endif()
endforeach()

# Missed one...
get_target_property(RegisterEverythingSources MLIRPythonExtension.RegisterEverything INTERFACE_SOURCES)

set(PYBINDINGS_SRC "${MLIR_INSTALL_PREFIX}/src/python/MLIRPythonExtension.Core")

set(_py_srcs
  ${CMAKE_CURRENT_SOURCE_DIR}/binding/binding.cpp
  ${_mlir_python_cpp_sources}
  ${RegisterEverythingSources}
)

set(_py_libs
  LLVMSupport
  ${MLIR_CONVERSION_LIBS}
  ${MLIR_DIALECT_LIBS}
  ${MLIR_EXTENSION_LIBS}
  ${MLIR_TRANSLATION_LIBS}
  ${MLIR_UPSTREAM_CAPI_LIBS}
)

declare_mlir_python_extension(EDSLPythonExtensions.MLIR
  MODULE_NAME _edsl
  ADD_TO_PARENT EDSLPythonExtensions
  ROOT_DIR "/"

  SOURCES
    ${_py_srcs}

  PRIVATE_LINK_LIBS
    ${_py_libs}
)
target_include_directories(
  EDSLPythonExtensions.MLIR
  INTERFACE $<BUILD_INTERFACE:${PYBINDINGS_SRC}>
)

_flatten_mlir_python_targets(mlir_python_sources_deps MLIRPythonSources)
list(FILTER mlir_python_sources_deps INCLUDE REGEX "^MLIRPythonSources.Dialects")
foreach(t ${mlir_python_sources_deps})
  set_property(TARGET ${t} PROPERTY mlir_python_DEPENDS "")
endforeach()
set_property(TARGET MLIRPythonSources.ExecutionEngine PROPERTY mlir_python_DEPENDS "")

add_mlir_python_modules(EDSLPythonModules
  ROOT_PREFIX "${EDSL_PYTHON_PACKAGES_DIR}/edsl"
  INSTALL_PREFIX "${EDSL_PYTHON_INSTALL_DIR}/edsl"
  DECLARED_SOURCES
    ${mlir_python_sources_deps}
    MLIRPythonSources.Core.Python
    MLIRPythonSources.ExecutionEngine
    EDSLPythonExtensions.MLIR
    EDSLPythonSources
)

set(_other_extensions
  _mlir
  _mlirAsyncPasses
  _mlirDialectsLinalg
  _mlirDialectsPDL
  _mlirDialectsQuant
  _mlirDialectsSparseTensor
  _mlirDialectsTransform
  _mlirExecutionEngine
  _mlirGPUPasses
  _mlirLinalgPasses
  _mlirPythonTest
  _mlirRegisterEverything
  _mlirSparseTensorPasses
)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${EDSL_PYTHON_INSTALL_DIR}/edsl/_mlir_libs")

foreach(ext ${_other_extensions})
  # Build time/directory creation.
  add_custom_target("symlink_${ext}"
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      "_edsl${PYTHON_MODULE_EXTENSION}"
      "${ext}${PYTHON_MODULE_EXTENSION}"
    DEPENDS EDSLPythonExtensions
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${EDSL_PYTHON_INSTALL_DIR}/edsl/_mlir_libs"
  )
  add_dependencies(EDSLPythonModules "symlink_${ext}")
  # Install time/directory creation.
  install(CODE "execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            _edsl${PYTHON_MODULE_EXTENSION}
            ${ext}${PYTHON_MODULE_EXTENSION}
            WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/${EDSL_PYTHON_INSTALL_DIR}/edsl/_mlir_libs
    )"
  )
endforeach()




# normal stuff below....

#declare_mlir_python_extension(EDSLPythonExtensions.MLIR
#  MODULE_NAME _edsl
#  ADD_TO_PARENT EDSLPythonExtensions
#  ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/binding
#
#  PARTIAL_SOURCES_INTENDED
#  SOURCES
#    binding.cpp
#  PRIVATE_LINK_LIBS
#    # upstream
#    LLVMObject
#    LLVMCore
#    LLVMSupport
#    LLVMOption
#    LLVMCodeGen
#    LLVMOrcJIT
#    # sharp edge
##    MLIRIR
#    # ours
#    MLIREdslEntry
#)
#
#add_mlir_python_common_capi_library(EDSLAggregateCAPI
#  INSTALL_COMPONENT EDSLPythonModules
#  INSTALL_DESTINATION ${EDSL_PYTHON_INSTALL_DIR}/edsl/_mlir_libs
#  OUTPUT_DIRECTORY "${EDSL_PYTHON_PACKAGES_DIR}/edsl/_mlir_libs"
#  # see https://github.com/llvm/llvm-project/blob/dcf376aae738252fb52a73bcf7f58fd030e15ee2/mlir/cmake/modules/AddMLIRPython.cmake#L405
#  RELATIVE_INSTALL_ROOT "../../../.."
#  DECLARED_SOURCES
#    MLIRPythonSources
#    MLIRPythonExtension.Core
#    MLIRPythonExtension.RegisterEverything
#    MLIRPythonExtension.ExecutionEngine
#    EDSLPythonSources
#    EDSLPythonExtensions
#)
#
#add_mlir_python_modules(EDSLPythonModules
#  ROOT_PREFIX "${EDSL_PYTHON_PACKAGES_DIR}/edsl"
#  INSTALL_PREFIX "${EDSL_PYTHON_INSTALL_DIR}/edsl"
#  DECLARED_SOURCES
#    EDSLPythonSources
#    MLIRPythonSources
#    MLIRPythonExtension.Core
#    MLIRPythonExtension.RegisterEverything
#    MLIRPythonExtension.ExecutionEngine
#    EDSLPythonExtensions
#  COMMON_CAPI_LINK_LIBS
#    EDSLAggregateCAPI
#)
