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

#ifndef LGL_NODES_SEGMENT_H
#define LGL_NODES_SEGMENT_H

#include <vector>

#include "quadrature_rules.h"

BEGIN_NAMESPACE

// 1D node policy using Legendre-Gauss-Lobatto points
template<typename T>
struct lgl_nodes_segment
{
  static std::size_t num_nodes(std::size_t order) { return order + 1; }

  template<typename OutputIterator>
  static void node_positions(std::size_t order, OutputIterator it)
  { 
    std::vector<T> tmp;
    return gauss_lobatto_quadrature(num_nodes(order), it, std::back_inserter(tmp));
  }
};

END_NAMESPACE

#endif
