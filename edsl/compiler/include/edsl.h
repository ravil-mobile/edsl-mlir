#ifndef EDSL_H_
#define EDSL_H_

#include "mlir/IR/BuiltinOps.h"

namespace edsl {
mlir::LogicalResult compile(mlir::ModuleOp module);
} // namespace edsl


#endif // EDSL_H_