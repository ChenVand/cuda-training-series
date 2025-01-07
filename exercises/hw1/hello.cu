#include <stdio.h>

// #include "cuda_runtime.h"
// #include "thrust/host_vector.h"
// #include "thrust/device_vector.h"

__global__ void hello(){

  auto block = blockIdx.x;
  auto thread = threadIdx.x;
  printf("Hello from block: %u, thread: %u\n", block, thread);
}

int main(){

  hello<<<2, 3>>>();
  cudaDeviceSynchronize();
}

