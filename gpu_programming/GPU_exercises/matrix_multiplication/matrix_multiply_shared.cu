/**
   Square matrix multiplication example

   author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
           Daniel Campora (dcampora@cern.ch)
           Alessandro Scarabotto (alessandro.scarabotto@cern.ch)
   date: 05/2019, 08/2024
 */

#include "matrix_utils.h"
#include <chrono>
#include <cstdio>
#include <iostream>

// Define the tile size
constexpr int TILE_SIZE = 32;

/**
 * @brief Multiplies matrices using shared memory.
 * @details This version of the square matrix multiplication uses
 *          shared memory and a predefined TILE_SIZE to preload data and
 *          speed up memory accesses.
 *
 *          Shared memory is populated in a coalesced manner, which more
 *          efficiently utilizes memory throughput.
 */
__global__ void multiply_square_matrices(const int size, const float *A,
                                         const float *B, float *C) {
    // define shared memory arrays using TILE_SIZE
    // make sub_multiplication on tile-sized matrices
    // sum sub_matrices together and fill global memory (C array)
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      float element = 0;
      for (int k = 0; k < size; k++) {
        element += A[i * size + k] * B[k * size + j];
      }
      C[i * size + j] = element;
    }
  }
}

int main(int argc, char *argv[]) {

  if (argc != 2) {
    std::cout << "Needs an argument: number of rows (= number of columns) of "
                 "square matrices\n";
    return -1;
  }

  const int matrix_size = atoi(argv[argc - 1]);

  // Allocate host and device memory for three matrices
  float *host_matrix[3]; // matrix[0] and matrix[1] are the source for the
                         // multiplication, result stored in matrix[2]
  float *device_matrix[3];

  for (int i = 0; i < 3; i++) {
    host_matrix[i] = new float[matrix_size * matrix_size];
    cudaMalloc((void **)&device_matrix[i],
               matrix_size * matrix_size * sizeof(float));
  }

  // Initialize matrices
  for (int i = 0; i < matrix_size; i++) {
    for (int j = 0; j < matrix_size; j++) {
      host_matrix[0][i * matrix_size + j] = 0.1 * (((i + 1) * (j + 1)) % 10);
      host_matrix[1][i * matrix_size + j] = 0.1 * ((2 * i + j) % 10);
      host_matrix[2][i * matrix_size + j] = 0;
    }
  }

  // Copy matrices to device
  for (int i = 0; i < 3; i++) {
    cudaMemcpy(device_matrix[i], host_matrix[i],
               matrix_size * matrix_size * sizeof(float),
               cudaMemcpyHostToDevice);
  }

  // Launch kernel
  int size = matrix_size;
  dim3 grid(1);
  dim3 block(1);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  multiply_square_matrices<<<grid, block>>>(size, device_matrix[0],
                                            device_matrix[1], device_matrix[2]);

  cudaDeviceSynchronize();

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;

  // Copy back result
  cudaMemcpy(host_matrix[2], device_matrix[2],
             matrix_size * matrix_size * sizeof(float), cudaMemcpyDeviceToHost);

  // Check and print result
  check_result(host_matrix[0], host_matrix[1], host_matrix[2], matrix_size,
               matrix_size, matrix_size);

  std::cout << "Kernel duration: " << elapsed_seconds.count() << " s\n";

  // Free memory
  for (int i = 0; i < 3; i++) {
    delete[] host_matrix[i];
    cudaFree(device_matrix[i]);
  }
}
