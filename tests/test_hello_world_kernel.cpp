#include <cuda_runtime_api.h>

#include "kernel_wrappers.h"

int test_hello_world_kernel()
{
  launch_hello_world();
  cudaDeviceReset();
  return 0;
}
