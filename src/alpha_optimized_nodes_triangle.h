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

#ifndef ALPHA_OPTIMIZED_NODES_TRIANGLE_H
#define ALPHA_OPTIMIZED_NODES_TRIANGLE_H 

#include <vector>
#include <limits>

#include "jacobi_polynomial.h"
#include "dense_matrix.h"
#include "quadrature_rules.h"

BEGIN_NAMESPACE

// 2D node policy for reference triangle
template<typename T>
struct alpha_optimized_nodes_triangle
{
  static std::size_t num_nodes(std::size_t order) { return (order + 1) * (order + 2) / 2; }

  template<typename OutputIterator>
  static void node_positions(std::size_t order, OutputIterator it);

  static std::size_t num_face_nodes(std::size_t order) { return order + 1; }

  template<typename OutputIterator>
  static void face_nodes(std::size_t order, std::size_t face_id, OutputIterator it);

private:
  static T wrap_factor(std::size_t order, T r);

  static std::pair<T, T> rs_to_ab(T r, T s)
  {
    T a = s == one<T> ? - one<T> : two<T> * (one<T> + r) / (one<T> - s) - one<T>;
    return std::make_pair(a, s);
  }

  inline static const T s_alpha[] = {T(0.0L), T(0.0L), T(1.4152L), T(0.1001L), T(0.2751L), T(0.9808L), T(1.0999L), T(1.2832L),
                                     T(1.3648L), T(1.4773L), T(1.4959L), T(1.5743L), T(1.5770L), T(1.6223L), T(1.6258L)};

  inline static dense_matrix<T, false> s_wrap_factor_matrix;

  inline static std::vector<T> s_wrap_factor_dist;
};

template<typename T> template<typename OutputIterator>
void alpha_optimized_nodes_triangle<T>::face_nodes(std::size_t order, std::size_t face_id, OutputIterator it)
{
  assert(order > 0);
  assert(face_id >= 0 && face_id <= 2);

  std::size_t delta = face_id == 0 ? 1 : (face_id == 1 ? order : 2);

  // indices of face nodes are subset of indices to node_positions()
  // and follow the counter-clockwise direction
  std::size_t prev = face_id == 0 ? 0 : (face_id == 1 ? order : num_nodes(order) - 1);
  it = prev;
  if (face_id == 0)
    for (std::size_t i = 1; i <= order; ++i)
    {
      std::size_t node = prev + delta;
      it = node;
      prev = node;
    }
  else if (face_id == 1)
    for (std::size_t i = 1; i <= order; ++i)
    {
      std::size_t node = prev + delta--;
      it = node;
      prev = node;
    }
  else
    for (std::size_t i = 1; i <= order; ++i)
    {
      std::size_t node = prev - delta++;
      it = node;
      prev = node;
    }
}

template<typename T> template<typename OutputIterator>
void alpha_optimized_nodes_triangle<T>::node_positions(std::size_t order, OutputIterator it)
{ 
  assert(order > 0);

  T alpha = order < 16 ? s_alpha[order - 1] : T(5) / T(3);
  T sqrt3 = sqrt(one<T> + two<T>);

  // this looping determines the ordering of nodes (consistent with the ordering of vertices)
  for (std::size_t i = 0; i <= order; ++i)
  {
    for (std::size_t j = 0; j <= (order - i); ++j)
    {
      // generate equal distance node in a symmetrical equilateral triangle
      T l1 = T(i) / T(order);
      T l3 = T(j) / T(order);
      T l2 = one<T> - l1 - l3;

      T x = l3 - l2;
      T y = (two<T> * l1 - l2 - l3) / sqrt3;

      // move the node based on wraping and blending
      T wrap1 = two<T> * two<T> * l2 * l3 * (one<T> + alpha * l1 * alpha * l1) * wrap_factor(order, l3 - l2);
      T wrap2 = two<T> * two<T> * l3 * l1 * (one<T> + alpha * l2 * alpha * l2) * wrap_factor(order, l1 - l3);
      T wrap3 = two<T> * two<T> * l1 * l2 * (one<T> + alpha * l3 * alpha * l3) * wrap_factor(order, l2 - l1);

      x += (wrap1 - half<T> * wrap2 - half<T> * wrap3);
      y += half<T> * sqrt3 * (wrap2 - wrap3);

      // transfer the position in the equilateral triangle to the reference triangle
      l1 = (one<T> + sqrt3 * y) / T(3);
      l2 = (two<T> - T(3) * x - sqrt3 * y) / T(6);
      l3 = (two<T> + T(3) * x - sqrt3 * y) / T(6);

      it = std::make_pair(l3 - l2 - l1, l1 - l2 - l3);
    }
  }
}

template<typename T>
T alpha_optimized_nodes_triangle<T>::wrap_factor(std::size_t order, T r)
{
  assert(order > 0);

  if (s_wrap_factor_matrix.size_row() != (order + 1) || s_wrap_factor_dist.size() != (order + 1))
  {
    // equal distance node positions
    std::vector<T> equal_dist_pos(order + 1);
    equal_dist_pos[0] = - one<T>;
    equal_dist_pos[order] = one<T>;
    for (std::size_t i = 1; i < order; ++i)
      equal_dist_pos[i] = two<T> * T(i) / T(order) - one<T>;

    // transpose of the vandermonde matrix based on the equal distance node positions
    dense_matrix<T, false> v(order + 1, order + 1);
    for (std::size_t col = 0; col < v.size_col(); ++col)
    {
      std::vector<T> vals;
      jacobi_polynomial_values(T{}, T{}, order, equal_dist_pos[col], std::back_inserter(vals));
      for(std::size_t row = 0; row < v.size_row(); ++row)
        v(row, col) = vals[row];
    }

    s_wrap_factor_matrix = v.inverse();

    // distances between LGL nodes and equal distance nodes
    std::vector<T> dist(order + 1);
    std::vector<T> pos, ws;
    gauss_lobatto_quadrature(order + 1, std::back_inserter(pos), std::back_inserter(ws));
    for (std::size_t i = 0; i <= order; ++i)
      dist[i] = pos[i] - equal_dist_pos[i];

    s_wrap_factor_dist = dist;
  }

  std::vector<T> p_vals;
  jacobi_polynomial_values(T{}, T{}, order, r, std::back_inserter(p_vals));
  std::vector<T> l_vals(order + 1, T{});
  s_wrap_factor_matrix.gemv(one<T>, p_vals.begin(), T{}, l_vals.begin());

  T w = 0;
  for (std::size_t i = 0; i <= order; ++i)
    w += l_vals[i] * s_wrap_factor_dist[i];

  return std::abs(r) < (one<T> - std::numeric_limits<T>::epsilon() * 10) ? w / (one<T> - r * r) : T{}; // hard-coded tolerance to avoid singularity!
}

END_NAMESPACE

#endif
