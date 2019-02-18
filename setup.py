from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mrcnn',
    author='Jean Da Rolt',
    url='github.com/darolt/mask_rcnn',
    packages=['mrcnn'],
    ext_modules=[
        CUDAExtension('nms_wrapper', [
            './mrcnn/models/components/nms/nms_wrapper.cpp',
            './mrcnn/models/components/nms/nms_cuda.cu',
        ]),
        CUDAExtension('crop_and_resize', [
            './mrcnn/models/components/roialign/crop_and_resize_gpu.cpp',
            './mrcnn/models/components/roialign/crop_and_resize_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
