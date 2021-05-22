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

#include <vector>
#include <iterator>
#include <iostream>

#include "reference_segment.h"

int test_reference_segment()
{
  using namespace dgc;

  std::vector<double> nodes;

  reference_segment<double> refSeg;
  for(std::size_t order = 1; order < 7; ++order)
  {
    nodes.clear();

    std::cout << "reference segment order = " << order << std::endl;
    std::cout << "number of nodes: " << refSeg.num_nodes(order) << std::endl;
    refSeg.node_positions(order, std::back_inserter(nodes));
    for (std::size_t j = 0; j < nodes.size(); ++j)
      std::cout << "p = " << nodes[j] << std::endl;    

    auto v = refSeg.vandermonde_matrix(order);
    std::cout << "vandermonde matrix: " << std::endl << v;
    auto vr = refSeg.grad_vandermonde_matrix(order);
    std::cout << "gradient of vandermonde matrix: " << std::endl << vr;
    std::cout << "the Dr matrix: " << std::endl << vr * v.inverse() << std::endl;
  }
  std::cout << std::endl;

  // TODO: could create custom basis and node policies and test them...

  return 0;
}
