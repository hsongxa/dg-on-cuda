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

#ifndef K_OPX_CUH
#define K_OPX_CUH

#include <cstddef>
#include <stdexcept>

#include "config.h"

BEGIN_NAMESPACE

template<typename T, typename CellOp>
__global__ void opx(const T* x, std::size_t size, T t, T* out, CellOp* c_op)
{
  // figure out the cell id (= thread id) and forward to the device code
  // for processing this cell
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint i = tid; i < c_op->num_cells(); i += blockDim.x * gridDim.x)
    c_op->operator()(i, x, size, t, out);
}

template<typename T, typename CellOp>
void k_opx(int grid_size, int block_size, const T* x, std::size_t size, T t, T* out, CellOp* c_op)
{
  opx<<<grid_size, block_size>>>(x, size, t, out, c_op);
  if (cudaGetLastError()) throw std::runtime_error("failed to launch kernel!");
}

END_NAMESPACE

#endif
