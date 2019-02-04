from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='crop_and_resize',
    ext_modules=[
        CUDAExtension('crop_and_resize', [
            'crop_and_resize_gpu.cpp',
            'crop_and_resize_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
