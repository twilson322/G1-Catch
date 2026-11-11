"""Builds the C++ flight core. Everything else is declared in pyproject.toml."""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    ext_modules=[Pybind11Extension("g1catch._flight_core", ["cpp/flight_core.cpp"], cxx_std=17,
                                   extra_compile_args=["-O2", "-Wall", "-Wextra"])],
    cmdclass={"build_ext": build_ext},
)
