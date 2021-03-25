from setuptools import setup
from distutils.core import setup, Extension

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

import os, sys

__version__ = "0.0.1"

linkargs = ["-Ofast", "-ffast-math", "-lpthread", "-lgomp", "-fopenmp-simd", "-g", "-w", "-fPIC", "-DNDEBUG", "-DEIGEN_USE_BLAS", "-lopenblas"]
compargs = ["-Ofast", "-ffast-math", "-lpthread", "-lgomp", "-fopenmp-simd", '-std=c++17', "-DNDEBUG", "-DEIGEN_USE_BLAS", "-lopenblas"]

ext_modules = [
    Extension("solver_cpp",
        ['solver_cpp.cpp'],
        include_dirs=[os.environ.get("EIGEN_INCLUDE_DIR", "/usr/include/eigen3/"), ".", pybind11.get_include()],
        extra_link_args = linkargs,
        extra_compile_args = compargs,
        define_macros = [('VERSION_INFO', __version__)],
        cxx_std=17,
        ),
]

setup(
    name='elasticpath',
    version=__version__,
    author='Bryn Noel Ubald',
    author_email='bubald@turing.ac.uk',
    description='Coordinate descent',
    long_description="",
    ext_modules=ext_modules,
    install_requires=[ 'numpy>=1.7' ],
    setup_requires=['pybind11>=2.2', 'numpy>=1.7'],
    zip_safe=False,
)
