from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="dmesh2_renderer",
    packages=['dmesh2_renderer'],
    ext_modules=[
        CUDAExtension(
            name="dmesh2_renderer._C",
            sources=[
                # cuda impl
                "cuda_impl/backward.cu",
                "cuda_impl/forward.cu",
                "cuda_impl/renderer.cu",

                "render.cu",
                "ext.cpp"
            ],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)