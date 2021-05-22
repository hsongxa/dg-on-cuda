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

#include "quadrature_rules.h"

int test_quadrature_rules()
{
  using namespace dgc;

  std::vector<double> points;
  std::vector<double> weights;

  // Gauss-Lobatto quadrature
  for(int np = 2; np < 8; ++np)
  {
    points.clear();
    weights.clear();
    gauss_lobatto_quadrature(np, std::back_inserter(points), std::back_inserter(weights));

    std::cout << "Gauss-Lobatto quadrature of " << np << " points:" << std::endl;
    for (std::size_t j = 0; j < points.size(); ++j)
      std::cout << "p = " << points[j] << ", w = " << weights[j] << std::endl;    
  }
  std::cout << std::endl;

  return 0;
}
