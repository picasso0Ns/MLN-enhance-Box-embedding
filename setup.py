"""
cpp_ext/setup.py
================
Build script for the C++ / pybind11 rule_miner extension.

Usage:
    cd cpp_ext
    python setup.py build_ext --inplace

Or from the project root:
    python cpp_ext/setup.py build_ext --inplace --build-lib cpp_ext/

Requirements:
    pip install pybind11
    A C++14 compiler (g++, clang++, or MSVC)

Optional:
    OpenMP is enabled automatically when the compiler supports it.
    On macOS with Apple Clang you may need:
        brew install libomp
        CXXFLAGS="-Xpreprocessor -fopenmp" python setup.py build_ext --inplace
"""

import sys
import os
from setuptools import setup, Extension

try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    raise RuntimeError(
        "pybind11 is required to build the C++ extension.\n"
        "Install it with:  pip install pybind11"
    )

# ---- Compiler flags ----
extra_compile_args = ['-std=c++14', '-O3', '-ffast-math']
extra_link_args    = []

# Try to enable OpenMP
if sys.platform == 'linux':
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')
elif sys.platform == 'darwin':
    # macOS with Homebrew libomp
    omp_prefix = '/usr/local/opt/libomp'
    if os.path.isdir(omp_prefix):
        extra_compile_args += ['-Xpreprocessor', '-fopenmp',
                               f'-I{omp_prefix}/include']
        extra_link_args    += [f'-L{omp_prefix}/lib', '-lomp']
elif sys.platform == 'win32':
    extra_compile_args = ['/O2', '/std:c++14', '/openmp']
    extra_link_args    = []

ext = Extension(
    name='rule_miner',
    sources=['rule_miner.cpp'],
    include_dirs=[pybind11_include],
    language='c++',
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name='rule_miner',
    version='1.0.0',
    author='MLN-BoxEmbedding Project',
    description='C++ accelerated Horn rule miner for knowledge graphs',
    ext_modules=[ext],
    zip_safe=False,
)
