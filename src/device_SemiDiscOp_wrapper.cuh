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

#ifndef DEVICE_SEMIDISCOP_WRAPPER_CUH
#define DEVICE_SEMIDISCOP_WRAPPER_CUH

#include <cstddef>
#include <stdexcept>

#include "config.h"

BEGIN_NAMESPACE

template<typename T, typename ConstItr, typename Itr, typename CellOp>
__global__ void opx(ConstItr x, std::size_t size, T t, Itr out, CellOp* c_op)
{
  // figure out the cell id (= thread id) and forward to the device code
  // for processing this cell
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint i = tid; i < c_op->num_cells(); i += blockDim.x * gridDim.x)
    c_op->operator()(i, x, size, t, out);
}

//template<typename Itr, typename CellOp>
//__global__ void opx(Itr x, std::size_t size, CellOp* c_op)
//{
//  // figure out the cell id (= thread id) and forward to the device code
//  // for processing this cell
//  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
//  for (uint i = tid; i < c_op->num_cells(); i += blockDim.x * gridDim.x)
//    c_op->operator()(i, x, size);
//}

template<typename DeviceSemiDiscOp>
struct device_SemiDiscOp_wrapper
{
  DeviceSemiDiscOp* m_Dop;
  int m_GridSize;
  int m_BlockSize;

  template<typename T, typename ConstItr, typename Itr>
  void operator()(ConstItr in_cbegin, std::size_t size, T t, Itr out_begin) const
  {
    opx<<<m_GridSize, m_BlockSize>>>(in_cbegin, size, t, out_begin, m_Dop);
    if (cudaGetLastError()) throw std::runtime_error("failed to launch kernel/device code!");
  }

  //template<typename Itr>
  //void operator()(Itr inout_begin, std::size_t size) const
  //{
  //  opx<<<m_GridSize, m_BlockSize>>>(inout_begin, size, m_Dop);
  //  if (cudaGetLastError()) throw std::runtime_error("failed to launch kernel/device code!");
  //}
};

END_NAMESPACE

#endif
