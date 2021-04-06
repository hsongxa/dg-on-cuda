#include <cuda_runtime.h>

#include <stdio.h>

__global__ void hello_world_k(void)
{
  printf("Hello World from GPU!\n");
}

void launch_hello_world()
{
  hello_world_k <<<1, 10>>> ();
}

