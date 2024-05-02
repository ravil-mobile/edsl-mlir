import os
import ninja
import shutil
import sys
import sysconfig
import subprocess
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from distutils.command.clean import clean
from pathlib import Path


def get_base_dir():
  dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
  return Path(dir) / "edsl"


def get_cmake_dir():
  cmake_dir = Path(get_base_dir()) / "compiler" / "build"
  cmake_dir.mkdir(parents=True, exist_ok=True)
  return cmake_dir


class CMakeBuild(build_ext):
  def initialize_options(self):
    build_ext.initialize_options(self)
    self.base_dir = get_base_dir()

  def finalize_options(self):
    build_ext.finalize_options(self)

  def build_extension(self, ext):
    extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))
    make_dir = shutil.which('make')
    python_include_dir = sysconfig.get_path("platinclude")
    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
    cmake_args = [
      f"{get_base_dir()}",
      # f"-DCMAKE_MAKE_PROGRAM={make_dir}",
      f"-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
      # f"-DLLVM_ENABLE_WERROR=ON",
      f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
      f"-DPython3_EXECUTABLE:FILEPATH={sys.executable}",
      f"-DCMAKE_VERBOSE_MAKEFILE=ON",
      f"-DPYTHON_INCLUDE_DIRS={python_include_dir}",
      f"-DCMAKE_INSTALL_PREFIX={Path(__file__).parent.absolute() / 'install'}",
      "-DLLVM_USE_LINKER=lld",
      "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
      "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
      "-GNinja",
      f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
    ]

    llvm_dir = os.getenv("LLVM_DIR") or None
    cmake_prefix = []
    if llvm_dir:
      cmake_prefix.append(llvm_dir)

    pybind11_config = shutil.which('pybind11-config')
    proc = subprocess.run(['python3', pybind11_config, '--cmakedir'],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,
                          text=True)
    if proc.returncode == 0:
      cmake_prefix.append(proc.stdout.strip())

    if cmake_prefix:
      cmake_args.append(
        f'-DCMAKE_PREFIX_PATH={";".join(cmake_prefix)}'
      )

    build_type = os.getenv("BUILD_TYPE") or None
    if build_type is None:
      build_type = "Debug"

    if build_type:
      assert build_type in ["Release", "Debug", "RelWithDebInfo"]
      cmake_args.append(
        f"-DCMAKE_BUILD_TYPE={build_type}",
      )

    env = os.environ.copy()
    cmake_dir = get_cmake_dir()
    subprocess.run(
      ["cmake", ext.sourcedir, *cmake_args],
      cwd=cmake_dir,
      check=True,
      env=env,
    )

    build_args = ["--config", build_type]
    subprocess.run(
      ["cmake", "--build", ".", "--target", "install"] + build_args,
      cwd=cmake_dir,
      check=True
    )

class CMakeClean(clean):
  def initialize_options(self):
    clean.initialize_options(self)
    self.build_temp = get_cmake_dir()


class CMakeExtension(Extension):
  def __init__(self, name, path, sourcedir=""):
    Extension.__init__(self, name, sources=[])
    self.sourcedir = os.path.abspath(sourcedir)
    self.path = path


setup(
  name="edsl",
  version="0.0.1",
  author="Ravil Dorozhinskii",
  author_email="ravil.aviva.com@gmail.com",
  description="Python EDSL Example",
  long_description="",
  packages=[],
  entry_points={
    'console_scripts': [
      'my-edsl=edsl.driver:main',
    ]
  },
  install_requires=[],
  package_data={},
  include_package_data=True,
  ext_modules=[CMakeExtension("_edsl", "edsl")],
  cmdclass={
    "build_ext": CMakeBuild,
    # "clean": CMakeClean,
  },
  zip_safe=False,
  keywords=["EDSL", "Python", "MLIR"],
  url="",
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  test_suite="tests",
  extras_require={
    "build": [
      "cmake>=3.20",
      "pybind11"
      "lit",
    ],
    "tests": [],
    "tutorials": [],
  },
)

