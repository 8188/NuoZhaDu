
# from distutils.core import setup
# from distutils.extension import Extension
# setuptools 效率高于 distutils
from setuptools import setup # c -> .pyd/.so
from setuptools.extension import Extension
from Cython.Build import cythonize # cython -> c
import numpy as np

ext_modules = Extension(
    name="utils",
    sources=["utils.pyx"],
    include_dirs=[".", np.get_include()], 
#     extra_compile_args=["/openmp"],
#     extra_link_args=["/openmp"],
#     libraries=["m"]  # Unix-like specific
)


setup(
    name="utils",
    version = '1.0',
    description='',
    ext_modules=cythonize(
        [ext_modules], annotate=True, language_level = "3", force=True
    )
)

# 链接不上 MSVC = /openmp 不管用，也许不需要 https://stackoverflow.com/questions/33613019/parallelism-in-cython-does-not-work
