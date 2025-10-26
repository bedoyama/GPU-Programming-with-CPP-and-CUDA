#include <iostream>
#include <stdio.h>        // Required for printf
#include <cuda_runtime.h> // <-- THIS IS THE FIX

__global__ void helloWorld()
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  printf("Hello, World! Thread %d\n", tid);
}

int main()
{
  // Launch the kernel with 1 block and 10 threads
  helloWorld<<<1, 10>>>();

  // Wait for the GPU to finish before letting the CPU exit
  cudaDeviceSynchronize();

  return 0;
}