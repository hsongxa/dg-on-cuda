#ifndef HELLO_WORLD_TEMPLATE_H
#define HELLO_WORLD_TEMPLATE_H

#include <cuda_runtime.h>
#include <stdio.h>

template<typename T>
__global__ void hello_world_template_k(void)
{
  printf("Hello World <template> from GPU!\n");
}

#endif
