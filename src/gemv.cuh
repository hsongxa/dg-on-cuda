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

#ifndef GEMV_CUH
#define GEMV_CUH

#include "config.h"

BEGIN_NAMESPACE

// assumption: the matrix m is row major
template<typename T>
__device__ void gemv(const T* m, bool transpose, std::size_t row, std::size_t col, T alpha, const T* x, T beta, T* y)
{
  if (transpose)
  {
    for (std::size_t i = 0; i < col; ++i)
    {
      T val = beta * y[i];
      for (std::size_t j = 0; j < row; ++j)
        val += alpha * x[j] * m[i + j * col];
      y[i] = val;
    }
  }
  else
  {
    for (std::size_t i = 0; i < row; ++i)
    {
      T val = beta * y[i];
      for (std::size_t j = 0; j < col; ++j)
        val += alpha * x[j] * m[j + i * col];
      y[i] = val;
    }
  }
}

END_NAMESPACE

#endif
