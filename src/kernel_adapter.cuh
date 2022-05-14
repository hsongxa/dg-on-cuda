/**
 * MIT License
 * 
 * Copyright (c) 2021 hsongxa
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 **/

#ifndef KERNEL_ADAPTER_CUH
#define KERNEL_ADAPTER_CUH

#include <cstddef>
#include <stdexcept>

#include "config.h"

BEGIN_NAMESPACE

 // rvlaue references do not work for kernels
template<typename DeviceFunctor, typename... Args>
__global__ void device_func_kernel(DeviceFunctor* obj, std::size_t num_execs, Args... args)
{
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint i = tid; i < num_execs; i += blockDim.x * gridDim.x)
    obj->operator()(i, args...);
}

template<typename DeviceFunctor, typename Phase, typename... Args>
__global__ void device_func_phase_kernel(DeviceFunctor* obj, std::size_t num_execs, Phase phase, Args... args)
{
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint i = tid; i < num_execs; i += blockDim.x * gridDim.x)
    obj->operator()(phase, i, args...);
}

// wrap a device object and launch its operator() as a kernel by the "execute" function
// or the "execute_phase" function if the operation needs to be executed in multiple
// phases, i.e., by multiple kernels
//
// unfortunately, device pointer to non-static member function cannot be used as a
// parameter to kernels, otherwise this adapter could be made even more general such
// that any member function, not just the operator(), can be launched by the kernel
template<typename DeviceFunctor>
struct kernel_adapter
{
  DeviceFunctor* Dobj;
  int GridSize;
  int BlockSize;
  std::size_t NumExecs; 

  kernel_adapter(DeviceFunctor* obj, int grid_size, int block_size, std::size_t num_execs)
    : Dobj(obj), GridSize(grid_size), BlockSize(block_size), NumExecs(num_execs) {}

  template<typename... Args>
  void execute(Args&&... args)
  {
    device_func_kernel<<<GridSize, BlockSize>>>(Dobj, NumExecs, args...);
    if (cudaGetLastError()) throw std::runtime_error("failed to launch kernel of device function!");
  }

  template<typename Phase, typename... Args>
  void execute_phase(Phase phase, Args&&... args)
  {
    device_func_phase_kernel<<<GridSize, BlockSize>>>(Dobj, NumExecs, phase, args...);
    if (cudaGetLastError()) throw std::runtime_error("failed to launch kernel of device function phase!");
  }
};

END_NAMESPACE

#endif
