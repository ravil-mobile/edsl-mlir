#include "edsl.h"
#include <pybind11/pybind11.h>
#include "mlir/CAPI/IR.h"
#include "mlir-c/IR.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinOps.h"
#include <stdexcept>


PYBIND11_MODULE(_edsl, m) {
  m.doc() = "edsl python bindings";
  m.def("entry", [](pybind11::object capsule){
    MlirModule mlirModule = mlirPythonCapsuleToModule(capsule.ptr());
    if (mlirModuleIsNull(mlirModule)) {
      throw std::runtime_error("empty module");
    }
    MlirContext context = mlirModuleGetContext(mlirModule);
    if (mlirContextIsNull(context)) {
      throw std::runtime_error("empty context");
    }
    auto module = unwrap(mlirModule);
    module->dump();
  });
}
