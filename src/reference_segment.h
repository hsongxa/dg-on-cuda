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

#ifndef REFERENCE_SEGMENT_H
#define REFERENCE_SEGMENT_H

#include <vector>
#include <cassert>

#include "dense_matrix.h"
#include "orthonormal_basis_segment.h"
#include "lgl_nodes_segment.h"

BEGIN_NAMESPACE

// 1D reference element
template<typename T,
         template<typename> class BasisPolicy = orthonormal_basis_segment,
         template<typename> class NodePolicy = lgl_nodes_segment>
struct reference_segment : public BasisPolicy<T>, NodePolicy<T>
{
  using matrix_type = dense_matrix<T, false>; // hard code the matrix type - can be extracted out if needed

  static matrix_type vandermonde_matrix(std::size_t order)
  {
    assert(order > 0);

    std::vector<T> pos;
    node_policy::node_positions(order, std::back_inserter(pos));

    matrix_type v(pos.size(), order + 1); 

    for (std::size_t row = 0; row < v.size_row(); ++row)
      for (std::size_t col = 0; col < v.size_col(); ++col)
        v(row, col) = basis_policy::value(col, pos[row]);
    return v;
  }

  static matrix_type grad_vandermonde_matrix(std::size_t order)
  {
    assert(order > 0);

    std::vector<T> pos;
    node_policy::node_positions(order, std::back_inserter(pos));

    matrix_type v(pos.size(), order + 1); 

    for (std::size_t row = 0; row < v.size_row(); ++row)
      for (std::size_t col = 0; col < v.size_col(); ++col)
        v(row, col) = basis_policy::derivative(col, pos[row]);
    return v;
  }

private:
  using node_policy = NodePolicy<T>;
  using basis_policy = BasisPolicy<T>;
};

END_NAMESPACE

#endif
