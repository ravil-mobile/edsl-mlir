from mlir.ir import Context, Module, get_dialect_registry
from mlir.ir import InsertionPoint, Location, F32Type
from mlir.dialects import func, arith
from mlir.dialects import arith
from edsl_cpp import entry


def main():
  registry = get_dialect_registry()
  context = Context()
  context.append_dialect_registry(registry)
  module = None
  with context:
    module = Module.create(Location.unknown())
    with InsertionPoint(module.body), Location.unknown():
      fp_type = F32Type.get()
      function = func.FuncOp("test", ([fp_type, fp_type, fp_type], [fp_type]))

      with InsertionPoint(function.add_entry_block()) as block:
        one = function.arguments[0]
        two = function.arguments[1]
        mult_res = arith.MulFOp(one, two)

        three = function.arguments[2]
        res = arith.AddFOp(mult_res, three)
        func.ReturnOp([res])

    entry(module._CAPIPtr)


if __name__ == "__main__":
  main()
