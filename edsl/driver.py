from edsl.ir import Context, Module, get_dialect_registry
from edsl.ir import InsertionPoint, Location, F32Type
from edsl.dialects import func, arith
from edsl.dialects import arith
from edsl._mlir_libs._edsl import entry


def main():
  registry = get_dialect_registry()
  context = Context()
  context.append_dialect_registry(registry)
  with context, Location.unknown():
    module = Module.create()
    fp_type = F32Type.get()
    with InsertionPoint(module.body):
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
