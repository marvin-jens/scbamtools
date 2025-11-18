from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy

# These are optional
Options.docstrings = True
Options.annotate = False

# Modules to be compiled and include_dirs when necessary
extensions = [
    Extension(
        "scbamtools.cython.kmers",
        ["scbamtools/cython/kmers.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "scbamtools.cython.bctree",
        ["scbamtools/cython/bctree.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "scbamtools.cython.fastquery",
        ["scbamtools/cython/fastquery.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

if __name__ == "__main__":
    from setuptools import setup

    setup(
        name="scbamtools",  # Required
        packages=[
            "scbamtools",
            "scbamtools.cython",
            "scbamtools.bin",
            "scbamtools.config",
        ],
        package_data={"scbamtools/cython": ["kmers.pyx", "bctree.pyx", "types.pxd"]},
        # A list of compiler Directives is available at
        # https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives
        # external to be compiled
        ext_modules=cythonize(
            extensions, compiler_directives={"language_level": 3, "profile": False}
        ),
    )
