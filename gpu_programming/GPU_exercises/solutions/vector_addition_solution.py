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
__global__ void vector_addition_gpu(float *a, float *b, float *c, int N) {
  const int start = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = start; i < N; i += stride) {
    c[i] = a[i] + b[i];
  }
}
"""

# Compile the CUDA kernel
module = compiler.SourceModule(kernel_code)

#define array size
N = 3
# Allocate memory on the GPU
a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
c = np.zeros(N).astype(np.float32)

# Allocate on device
a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)
b_gpu = cuda.mem_alloc(b.size * b.dtype.itemsize)
c_gpu = cuda.mem_alloc(c.size * c.dtype.itemsize)

# Copy from host to device
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)
cuda.memcpy_htod(c_gpu, c)

# Launch the CUDA kernel
n_blocks = 4
n_threads = 3
vector_addition_gpu = module.get_function("vector_addition_gpu")
vector_addition_gpu(a_gpu, b_gpu, c_gpu, np.int32(N), grid=(n_blocks,1,1), block=(n_threads,1,1))

# Copy the result back to the CPU
cuda.memcpy_dtoh(c, c_gpu)


# Clean up
context.pop()

print("a ",a)
print(" + ")
print("b ",b)
print(" = ")
print("c ",c)