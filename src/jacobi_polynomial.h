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

#ifndef JABOBI_POLYNOMIAL_H
#define JABOBI_POLYNOMIAL_H

#include <cmath>
#include <cstddef>

#include "config.h"

BEGIN_NAMESPACE

template<typename T>
T gamma(T arg) { return std::tgamma(arg); }

template<typename T>
T sqrt(T arg) { return std::sqrt(arg); }

template<typename T>
T pow(T base, T exp) { return std::pow(base, exp); }

template<typename T>
constexpr T one = T(1.0L);

template<typename T>
constexpr T two = T(2.0L);

template<typename T>
constexpr T half = T(0.5L);

template<typename T>
T jacobi_polynomial_value(T alpha, T beta, std::size_t n, T x)
{
  T a1 = alpha + one<T>;
  T b1 = beta + one<T>;
  T ab2 = alpha + beta + two<T>;

  T prev_prev_val = sqrt(gamma(ab2) / gamma(a1) / gamma(b1) / pow(two<T>, a1 + beta));
  if (n == 0) return prev_prev_val;

  T prev_val = half<T> * prev_prev_val * sqrt((ab2 + one<T>) / a1 / b1) * (ab2 * x + alpha - beta);
  if (n == 1) return prev_val;

  T val;
  T a_i = two<T> / ab2 * sqrt((a1 + beta) * a1 * b1 / (ab2 - one<T>) / (ab2 + one<T>));
  for (std::size_t i = 1; i < n; ++i)
  {
    T iab = two<T> * i + alpha + beta;
    T b_i = (alpha + beta) * (beta - alpha) / iab / (iab + two<T>);
    T a_n = two<T> / (iab + two<T>) * sqrt((i + one<T>) * (i + a1 + beta) * (i + a1) * (i + b1) / (iab + one<T>) / (iab + one<T> + two<T>));
    val = ((x - b_i) * prev_val - a_i * prev_prev_val) / a_n;

    a_i = a_n;
    prev_prev_val = prev_val;
    prev_val = val;
  }

  return val;
}

template<typename T, typename OutputIterator>
void jacobi_polynomial_values(T alpha, T beta, std::size_t n, T x, OutputIterator it)
{
  T a1 = alpha + one<T>;
  T b1 = beta + one<T>;
  T ab2 = alpha + beta + two<T>;

  T prev_prev_val = sqrt(gamma(ab2) / gamma(a1) / gamma(b1) / pow(two<T>, a1 + beta));
  it = prev_prev_val;
  if (n == 0) return;

  T prev_val = half<T> * prev_prev_val * sqrt((ab2 + one<T>) / a1 / b1) * (ab2 * x + alpha - beta);
  it = prev_val;
  if (n == 1) return;

  T a_i = two<T> / ab2 * sqrt((a1 + beta) * a1 * b1 / (ab2 - one<T>) / (ab2 + one<T>));
  for (std::size_t i = 1; i < n; ++i)
  {
    T iab = two<T> * i + alpha + beta;
    T b_i = (alpha + beta) * (beta - alpha) / iab / (iab + two<T>);
    T a_n = two<T> / (iab + two<T>) * sqrt((i + one<T>) * (i + a1 + beta) * (i + a1) * (i + b1) / (iab + one<T>) / (iab + one<T> + two<T>));
    T val = ((x - b_i) * prev_val - a_i * prev_prev_val) / a_n;
    it = val;

    a_i = a_n;
    prev_prev_val = prev_val;
    prev_val = val;
  }
}

template<typename T>
T jacobi_polynomial_derivative(T alpha, T beta, std::size_t n, T x)
{
  if (n == 0) return T{}; // zero

  T val = jacobi_polynomial_value(alpha + one<T>, beta + one<T>, n - 1, x);
  return sqrt(n * (n + alpha + beta + one<T>)) * val;
}

template<typename T, typename OutputIterator>
void jacobi_polynomial_derivatives(T alpha, T beta, std::size_t n, T x, OutputIterator it)
{
  it = T{}; // zero
  if (n == 0) return;
  
  T input_alpha = alpha;
  T input_beta = beta;

  alpha = alpha + one<T>;
  beta = beta + one<T>;
  n = n - 1;

  // repeat the code for evaluating values at the new (alpha, beta, n)
  T a1 = alpha + one<T>;
  T b1 = beta + one<T>;
  T ab2 = alpha + beta + two<T>;

  T prev_prev_val = sqrt(gamma(ab2) / gamma(a1) / gamma(b1) / pow(two<T>, a1 + beta));
  it = sqrt(input_alpha + input_beta + two<T>) * prev_prev_val;
  if (n == 0) return;

  T prev_val = half<T> * prev_prev_val * sqrt((ab2 + one<T>) / a1 / b1) * (ab2 * x + alpha - beta);
  it = sqrt(two<T> * (two<T> + input_alpha + input_beta + one<T>)) * prev_val;
  if (n == 1) return;

  T a_i = two<T> / ab2 * sqrt((a1 + beta) * a1 * b1 / (ab2 - one<T>) / (ab2 + one<T>));
  for (std::size_t i = 1; i < n; ++i)
  {
    T iab = two<T> * i + alpha + beta;
    T b_i = (alpha + beta) * (beta - alpha) / iab / (iab + two<T>);
    T a_n = two<T> / (iab + two<T>) * sqrt((i + one<T>) * (i + a1 + beta) * (i + a1) * (i + b1) / (iab + one<T>) / (iab + one<T> + two<T>));
    T val = ((x - b_i) * prev_val - a_i * prev_prev_val) / a_n;
    it = sqrt((i + 2) * ((i + 2) + input_alpha + input_beta + one<T>)) * val;

    a_i = a_n;
    prev_prev_val = prev_val;
    prev_val = val;
  }
}

END_NAMESPACE

#endif
