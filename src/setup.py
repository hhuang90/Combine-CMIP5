from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions1 = Extension(name="estimate_gamma_m",
                sources=["estimate_gamma_m.pyx"],
                include_dirs=[numpy.get_include()],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'])

extensions2 = Extension(name="estimate_X",
                sources=["estimate_X.pyx"],
                include_dirs=[numpy.get_include()],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'])

setup(
    ext_modules=cythonize(extensions1),
)

setup(
    ext_modules=cythonize(extensions2),
)