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

#ifndef ORTHONORMAL_BASIS_TRIANGLE_H
#define ORTHONORMAL_BASIS_TRIANGLE_H

#include <utility>
#include <tuple>

#include "const_val.h"
#include "jacobi_polynomial.h"
#include "orthonormal_basis_segment.h"


BEGIN_NAMESPACE

// 2D orthonormal basis defined on reference triangle with
// r >= -1, s >= -1, and (r + s) <= 0.
template<typename T>
struct orthonormal_basis_triangle
{
  using basis_on_face = orthonormal_basis_segment<T>;

  static T value(std::size_t order_r, T r, std::size_t order_s, T s)
  {
    T a, b;
    std::tie(a, b) = rs_to_ab(r, s);
    return sqrt(const_val<T, 2>) * jacobi_polynomial_value(const_val<T, 0>, const_val<T, 0>, order_r, a) * jacobi_polynomial_value(const_val<T, 2> * order_r + const_val<T, 1>, const_val<T, 0>, order_s, b) * pow(const_val<T, 1> - b, static_cast<T>(order_r));
  }

  static std::pair<T, T> derivative(std::size_t order_r, T r, std::size_t order_s, T s)
  {
    T a, b;
    std::tie(a, b) = rs_to_ab(r, s);
    T val_a = jacobi_polynomial_value(const_val<T, 0>, const_val<T, 0>, order_r, a);
    T val_b = jacobi_polynomial_value(const_val<T, 2> * order_r + const_val<T, 1>, const_val<T, 0>, order_s, b);
    T deri_a = jacobi_polynomial_derivative(const_val<T, 0>, const_val<T, 0>, order_r, a);
    T deri_b = jacobi_polynomial_derivative(const_val<T, 2> * order_r + const_val<T, 1>, const_val<T, 0>, order_s, b);

    T dr = sqrt(const_val<T, 2>) * deri_a * val_b;
    if (order_r > 0) dr *= (const_val<T, 2> * pow(const_val<T, 1> - b, static_cast<T>(order_r - 1)));

    T ds = order_r == 0 ? sqrt(const_val<T, 2>) * (deri_a * val_b * (const_val<T, 1> / const_val<T, 2>) * (const_val<T, 1> + a) + val_a * deri_b) :
                          sqrt(const_val<T, 2>) * pow(const_val<T, 1> - b, static_cast<T>(order_r - 1)) * (deri_a * val_b * (const_val<T, 1> + a) + val_a * deri_b * (const_val<T, 1> - b) - static_cast<T>(order_r) * val_a * val_b);

    return std::make_pair(dr, ds);
  }

private:
  static std::pair<T, T> rs_to_ab(T r, T s)
  {
    T a = s == const_val<T, 1> ? - const_val<T, 1> : const_val<T, 2> * (const_val<T, 1> + r) / (const_val<T, 1> - s) - const_val<T, 1>;
    return std::make_pair(a, s);
  }
};

END_NAMESPACE

#endif
