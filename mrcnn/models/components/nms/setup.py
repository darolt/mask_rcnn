from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nms_wrapper',
    ext_modules=[
        CUDAExtension('nms_wrapper', [
            'nms_wrapper.cpp',
            'nms_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
