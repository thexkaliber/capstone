from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(name='swin_window_process',
    ext_modules=[
        CUDAExtension('swin_window_process', [
            'swin_window_process.cpp',
            'swin_window_process_kernel.cu',
        ], extra_compile_args={'cxx':['-D_GLIBCXX_USE_CXX11-ABI=1'], 'nvcc':['-D_GLIBCXX_USE_CX11_ABI=1']})
    ],
    cmdclass={'build_ext': BuildExtension})
