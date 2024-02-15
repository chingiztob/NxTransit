from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("calculate_delay.pyx", annotate=True)
)
