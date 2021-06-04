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

#include <cstddef>
#include <cuda_runtime.h>

#include "config.h"

BEGIN_NAMESPACE

template<typename T>
__global__ void axpy(T a, const T* x, std::size_t n, T* y)
{
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint i = tid; i < n; i += blockDim.x * gridDim.x)
    y[i] = a * x[i] + y[i];
}

void d_axpy(double a, const double* x, std::size_t n, double* y)
{
  int blockSize;
  int minGridSize;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axpy<double>, 0, 0);

  int gridSize = (n + blockSize - 1) / blockSize;
  axpy<<<gridSize, blockSize>>>(a, x, n, y);
}

END_NAMESPACE

