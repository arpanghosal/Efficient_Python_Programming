#!/usr/bin/env -S submit -M 2000 -f python

# Vector addition in pyCUDA
# author: Alessandro Scarabotto (alessandro.scarabotto@cern.ch)
# date: 08/2024

import numpy as np
from pycuda import compiler, gpuarray
import pycuda.driver as cuda

# Initialize PyCUDA
cuda.init()

# Create a CUDA context
device = cuda.Device(0)
context = device.make_context()

# Define the CUDA kernel
kernel_code = """
// define global function performing a + b = c addition
// using a strided for loop
"""

# Compile the CUDA kernel
module = compiler.SourceModule(kernel_code)

#define array size and arrays
# decide your array size
# N = 3
a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
c = np.zeros(N).astype(np.float32)

# Allocate memory on the GPU
# a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)
# do the same for b and c


# Copy from host to device
# cuda.memcpy_htod(a_gpu, a)
# same for b and c


# Launch the CUDA kernel
# decide blocks and grid size
# call kernel
# function_name = module.get_function("function_name")
# function_name(a_gpu, b_gpu, c_gpu, np.int32(N), grid=(n_blocks,1,1), block=(n_threads,1,1))


# Copy the result back to the CPU
# cuda.memcpy_dtoh(c, c_gpu)


# Clean up
context.pop()

print("a ",a)
print(" + ")
print("b ",b)
print(" = ")
print("c ",c)