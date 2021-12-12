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

#ifndef K_AXPY_CUH
#define K_AXPY_CUH

#include <cstddef>

#include <cuda_runtime.h>

#include "config.h"

BEGIN_NAMESPACE

// single pointer version

template<typename T>
__global__ void axpy(T a, const T* x, std::size_t n, const T* y, T* out)
{
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint i = tid; i < n; i += blockDim.x * gridDim.x)
    out[i] = a * x[i] + y[i];
}

template<typename T>
void k_axpy_auto(T a, const T* x, std::size_t n, const T* y, T* out)
{
  int blockSize;
  int minGridSize;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axpy<T>, 0, 0);

  int gridSize = (n + blockSize - 1) / blockSize;
  axpy<<<gridSize, blockSize>>>(a, x, n, y, out);
}

template<typename T>
void k_axpy(int grid_size, int block_size, T a, const T* x, std::size_t n, const T* y, T* out)
{ axpy<<<grid_size, block_size>>>(a, x, n, y, out); }

// zip_iterator version: directly use axpy_n instead, which calls thrust::tranform(),
// which will generate device code if the zipped iterators are from device_vectors

END_NAMESPACE

#endif
