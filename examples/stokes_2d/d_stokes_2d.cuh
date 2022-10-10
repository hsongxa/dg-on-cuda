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

#ifndef D_STOKES_2D_CUH
#define D_STOKES_2D_CUH

#include <cstddef>
#include <cmath>
#include <math.h>

#include "d_simple_discretization_2d.cuh"
#include "gemv.cuh"


// a simplified version of the stokes_2d class on device
template<typename T, typename I>
struct d_stokes_2d : public dgc::d_simple_discretization_2d<T, I>
{
  const T* Dr;
  const T* Ds;
  const T* L;

  // "D" stands for device
  using DDblIterator = thrust::device_vector<double>::iterator;
  using DIteratorTuple = thrust::tuple<DDblIterator, DDblIterator, DDblIterator>;
  using DZipIterator = thrust::zip_iterator<DIteratorTuple>;

  // process the specified cell 
  __device__ void operator()(std::size_t cid, DZipIterator in, std::size_t size, T t, DZipIterator out) const
  {
    // TODO: implement kernels
  }

  // in addition, also need to tell kernel how many cells in total
  __device__ I num_cells() const { return this->NumCells; }
};

#endif
