from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(name="reax_lt", ext_modules=[CUDAExtension("reax_lt", ["reax_lt.cpp"], libraries=["cublasLt"])],
      cmdclass={"build_ext": BuildExtension})
