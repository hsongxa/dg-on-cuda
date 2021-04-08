#include <cuda_runtime_api.h>

//#include "cuda_launch.h"
#include "hello_world_template.cuh"

int test_hello_world_template_kernel()
{
  hello_world_template_k<double><<<1, 10>>>();
  cudaDeviceReset();
  return 0;
}
