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

#ifndef AXPY_H
#define AXPY_H

#include <cstddef>
#include <cassert>

#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

BEGIN_NAMESPACE

// define the axpy operation on thrust::tuples --
// these are needed for axpy to deal with thrust::zip_iterator

template <typename T, int N>
struct tuple_axpy;

template <typename T>
struct tuple_axpy<T, 2> : public thrust::binary_function<thrust::tuple<T, T>, thrust::tuple<T, T>, thrust::tuple<T, T>>
{ 
  tuple_axpy(T a) : m_a(a) {}

  __host__ __device__ thrust::tuple<T, T> operator()(const thrust::tuple<T, T>& x, const thrust::tuple<T, T>& y) const
  {
    return thrust::make_tuple(m_a * thrust::get<0>(x) + thrust::get<0>(y),
                              m_a * thrust::get<1>(x) + thrust::get<1>(y));
  }

private:
  T m_a;
};

template <typename T>
struct tuple_axpy<T, 3> : public thrust::binary_function<thrust::tuple<T, T, T>, thrust::tuple<T, T, T>, thrust::tuple<T, T, T>>
{ 
  tuple_axpy(T a) : m_a(a) {}

  __host__ __device__ thrust::tuple<T, T, T> operator()(const thrust::tuple<T, T, T>& x, const thrust::tuple<T, T, T>& y) const
  {
    return thrust::make_tuple(m_a * thrust::get<0>(x) + thrust::get<0>(y),
                              m_a * thrust::get<1>(x) + thrust::get<1>(y),
                              m_a * thrust::get<2>(x) + thrust::get<2>(y));
  }

private:
  T m_a;
};

template <typename T>
struct tuple_axpy<T, 4> : public thrust::binary_function<thrust::tuple<T, T, T, T>, thrust::tuple<T, T, T, T>, thrust::tuple<T, T, T, T>>
{ 
  tuple_axpy(T a) : m_a(a) {}

  __host__ __device__ thrust::tuple<T, T, T, T> operator()(const thrust::tuple<T, T, T, T>& x, const thrust::tuple<T, T, T, T>& y) const
  {
    return thrust::make_tuple(m_a * thrust::get<0>(x) + thrust::get<0>(y),
                              m_a * thrust::get<1>(x) + thrust::get<1>(y),
                              m_a * thrust::get<2>(x) + thrust::get<2>(y),
                              m_a * thrust::get<3>(x) + thrust::get<3>(y));
  }

private:
  T m_a;
};

// ... ...
//
// this can go on and on, up to tuple of 9 elements which is the maximum size thrust::zip_iterator can zip

// traits class to help dispatch axpy based on the iterator types
template <typename ItrType>
struct zip_iterator_traits
{
  static constexpr bool is_zip_iterator = false;
  static constexpr int  zip_size = 0;
};

template <typename Tuple>
struct zip_iterator_traits<thrust::zip_iterator<Tuple>>
{
  static constexpr bool is_zip_iterator = true;
  static constexpr int  zip_size = thrust::tuple_size<Tuple>::value;
};

// axpy

template <typename T, typename ConstItr, typename Itr>
void axpy_n(T a, ConstItr x_cbegin, std::size_t x_size, ConstItr y_cbegin, Itr out_begin)
{
  assert(x_cbegin != y_cbegin);
  assert(out_begin != x_cbegin && out_begin != y_cbegin);

  if constexpr (zip_iterator_traits<ConstItr>::is_zip_iterator)
    thrust::transform(x_cbegin, x_cbegin + x_size, y_cbegin, out_begin, tuple_axpy<T, zip_iterator_traits<ConstItr>::zip_size>(a));
  else
    for (std::size_t i = 0; i < x_size; ++i) *out_begin++ = a * (*x_cbegin++) + (*y_cbegin++);
}

END_NAMESPACE

#endif
