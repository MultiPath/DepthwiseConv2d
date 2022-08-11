#!/usr/bin/env python3
import os

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUTLASS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cutlass'))

setup(
    name='depthwise_conv2d_implicit_gemm',
    py_modules=['depthwise_conv2d_implicit_gemm'],
    ext_modules=[
        CUDAExtension(
            name='_depthwise_conv2d_implicit_gemm_C',
            sources=[
                "src/frontend.cpp",
                "src/forward_fp32.cu",
                "src/backward_data_fp32.cu",
                "src/backward_filter_fp32.cu",
                "src/forward_fp16.cu",
                "src/backward_data_fp16.cu",
                "src/backward_filter_fp16.cu",
            ],
            include_dirs=[
                ".",
                os.path.join(CUTLASS_ROOT, "include"),
                os.path.join(CUTLASS_ROOT, "tools", "library", "include"),
                os.path.join(CUTLASS_ROOT, "tools", "util", "include"),
                os.path.join(CUTLASS_ROOT, "examples", "common"),
            ],
            extra_compile_args=['-g']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
