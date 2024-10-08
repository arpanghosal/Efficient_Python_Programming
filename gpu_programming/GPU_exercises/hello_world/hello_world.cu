/**
   Most simple CUDA Hello World program

   author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
        Alessandro Scarabotto (alessandro.scarabotto@cern.ch)
   date: 05/2019
   updated 08/2024
 */

#include <stdio.h>
#include <iostream>

using namespace std;

/* to do: Add the __global__ keyword in front of the function declaration to indicate that this function is executed on the GPU */
__global__ void hello_world_gpu() {
    
    /* to do: uncomment this line once the hello_world_gpu function has been marked with the __global__ keyword */
    printf("Hello World from the GPU at block %u, thread %u \n", blockIdx.x, threadIdx.x);
}

void hello_world_cpu() {
    printf("Hello World from the CPU \n");
}

int main( int argc, char *argv[] ) {

  if ( argc != 3 ) {
    cout << "Need two arguments: number of blocks and number of threads" << endl;
    return -1;
  }

  /* Call CPU function */
  hello_world_cpu();
    
  /* Call GPU function */
  /* taking 2 inputs from command line*/
  const int n_blocks  = atoi(argv[argc-2]);
  const int n_threads = atoi(argv[argc-1]);
  
   
  /* Refactor and code below to call the function on the GPU */  
    
    
    
  /* to do: variables of type dim3 to declare the size of the grid (n_blocks) and the size of the blocks (n_threads) 
      example: dim3 grid_dim(n_blocks);
      define the same for block dimension, also as dim3, using n_threads
  */
   dim3 grid_dim(n_blocks);
   dim3 block_dim(n_threads);
   
   // cout << grid_dim; // doesn't work, look into this later
  /* to do: launch the kernel
     Reminder: Syntax to launch a kernel: 
     kernel_name<<<grid_dim, block_dim>>>();
     grid_dim and block_dim are the variables of type dim3 that you declared above
     paramters can be passed to the function in the brackets (), 
     the kernel in this exercise does not take any parameters
  */
    hello_world_gpu<<<grid_dim, block_dim>>>();


  /* to do: call the pre-defined function cudaDeviceSynchronize();
     It blocks until all requested tasks on device were completed
  */
    cudaDeviceSynchronize();
    cout << " Done \n";
  return 0;
}
