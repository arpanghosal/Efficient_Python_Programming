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
// add here a global function call printing also block and thread index
__global__ void hello_world_gpu() {
 printf("Hello World from the GPU at block %u, thread %u \\n", blockIdx.x, threadIdx.x);
}
"""

# Compile the CUDA kernel
module = compiler.SourceModule(kernel_code)

# Launch the CUDA kernel
# get here the function
# change name_of_function
# name_of_function = module.get_function("name_of_function")
hello_world_gpu = module.get_function("hello_world_gpu")

# Define grid size in x,y,z (number of blocks) 
# and block size x,y,z (number of threads)
# add here the function call
# name_of_function(grid=(x,y,z), block=(x,y,z))
hello_world_gpu(grid=(3,2,1), block=(1,2,2))

# Clean up
context.pop()