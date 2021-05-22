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

#ifndef REFERENCE_TRIANGLE_H 
#define REFERENCE_TRIANGLE_H

#include <utility>
#include <cassert>

#include "dense_matrix.h"
#include "orthonormal_basis_triangle.h"
#include "alpha_optimized_nodes_triangle.h"

BEGIN_NAMESPACE

// 1D reference element
template<typename T,
         template<typename> class BasisPolicy = orthonormal_basis_triangle,
         template<typename> class NodePolicy = alpha_optimized_nodes_triangle>
struct reference_triangle : public BasisPolicy<T>, NodePolicy<T>
{
  using matrix_type = dense_matrix<T, false>; // hard code the matrix type - can be extracted out if needed

  static matrix_type vandermonde_matrix(std::size_t order)
  {
    assert(order > 0);

    std::vector<std::pair<T, T>> pos;
    node_policy::node_positions(order, std::back_inserter(pos));

    std::size_t size = pos.size();
    matrix_type v(size, size); 

    // this looping determines the ordering of the 2D basis
    std::size_t col = 0;
    for (std::size_t i = 0; i <= order; ++i)
      for (std::size_t j = 0; j <= (order - i); ++j)
      {
        for (std::size_t row = 0; row < size; ++row)
          v(row, col) = basis_policy::value(i, pos[row].first, j, pos[row].second);
        col++;
      }

    return v;
  }

  static std::pair<matrix_type, matrix_type> grad_vandermonde_matrix(std::size_t order)
  {
    assert(order > 0);

    std::vector<std::pair<T, T>> pos;
    node_policy::node_positions(order, std::back_inserter(pos));
    
    std::size_t size = pos.size();
    matrix_type v_r(size, size);
    matrix_type v_s(size, size); 

    // this looping determines the ordering of the 2D basis
    std::size_t col = 0;
    for (std::size_t i = 0; i <= order; ++i)
      for (std::size_t j = 0; j <= (order - i); ++j)
      {
        for (std::size_t row = 0; row < size; ++row)
        {
          auto deri = basis_policy::derivative(i, pos[row].first, j, pos[row].second);
          v_r(row, col) = deri.first;
          v_s(row, col) = deri.second;
        }
        col++;
      }

    return std::make_pair(v_r, v_s);
  }

  static matrix_type face_vandermonde_matrix(std::size_t order, std::size_t face_id)
  {
    assert(face_id >= 0 && face_id <=2);
    using basis_on_face = typename basis_policy::basis_on_face;

    std::vector<std::pair<T, T>> pos;
    node_policy::node_positions(order, std::back_inserter(pos));
    std::vector<std::size_t> face_ids;
    node_policy::face_nodes(order, face_id, std::back_inserter(face_ids));

    matrix_type fv(order + 1, order + 1);
    for (std::size_t row = 0; row < fv.size_row(); ++row)
      for (std::size_t col = 0; col < fv.size_col(); ++col)
        fv(row, col) = face_id == 0 ?
                       basis_on_face::value(col, pos[face_ids[row]].first) :
                       basis_on_face::value(col, pos[face_ids[row]].second);

    return fv;
  }

private:
  using node_policy = NodePolicy<T>;
  using basis_policy = BasisPolicy<T>;
};

END_NAMESPACE

#endif
