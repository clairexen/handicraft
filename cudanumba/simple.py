#!/usr/bin/env python3

# install notes:
# sudo LLVM_CONFIG=`which llvm-config-3.7` pip3 install numba

# usage:
# export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
# export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice/
# nvprof python3 simple.py

from numba import cuda
import numpy as np

@cuda.jit
def matmul(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)
C = np.zeros([1000, 1000])
matmul(A, B, C)

