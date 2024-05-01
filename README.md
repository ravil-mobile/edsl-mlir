# Dependencies

Firstly, you need to install LLVM

```bash
git clone --depth=1 https://github.com/llvm/llvm-project.git
mkdir -p llvm-project/build && cd llvm-project/build

python -m pip install -r ../mlir/python/requirements.txt

cmake ../llvm \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_CCACHE_BUILD=ON \
-DLLD_BUILD_TOOLS=ON \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DLLVM_ENABLE_PROJECTS="llvm;mlir" \
-DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
-DPython3_EXECUTABLE=$(which python3) \
-DCMAKE_INSTALL_PREFIX=$(realpath ~/.llvm)

make -j <num_proc>
make install
```

# Installation

```bash
export LLVM_DIR=$(realpath ~/.llvm)
pip3 install -e .
```

# Executing

```bash
export PYTHONPATH=${LLVM_DIR}/python_packages/mlir_core
python3 ./edsl/driver.py
```
