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

#ifndef ORTHONORMAL_BASIS_SEGMENT_H
#define ORTHONORMAL_BASIS_SEGMENT_H

#include "const_val.h"
#include "jacobi_polynomial.h"

BEGIN_NAMESPACE

// 1D orthonormal basis defined on [-1, 1]
template<typename T>
struct orthonormal_basis_segment
{
  static T value(std::size_t order, T x)
  { return jacobi_polynomial_value(const_val<T, 0>, const_val<T, 0>, order, x); }

  static T derivative(std::size_t order, T x)
  { return jacobi_polynomial_derivative(const_val<T, 0>, const_val<T, 0>, order, x); }
};

END_NAMESPACE

#endif
