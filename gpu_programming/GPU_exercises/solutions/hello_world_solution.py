#!/usr/bin/env -S submit -M 2000 -f python

# Hello World printing in pyCUDA
# author: Alessandro Scarabotto (alessandro.scarabotto@cern.ch)
# date: 08/2024

import numpy as np
from pycuda import driver, compiler

# Initialize PyCUDA
driver.init()

# Create a CUDA context
device = driver.Device(0)
context = device.make_context()

# Define the CUDA kernel
kernel_code = """
#include <stdio.h>
__global__ void hello_world_gpu() {
   if ( blockIdx.x < 100 && threadIdx.x < 100 ) 
    printf("Hello World from the GPU at block %u, thread %u \\n", blockIdx.x, threadIdx.x);
}
"""

# Compile the CUDA kernel
module = compiler.SourceModule(kernel_code)

# Launch the CUDA kernel
hello_world_gpu = module.get_function("hello_world_gpu")

# Define grid size in x,y,z (number of blocks) 
# and block size x,y,z (number of threads)
hello_world_gpu(grid=(3,1,1), block=(4,1,1))

# Clean up
context.pop()
