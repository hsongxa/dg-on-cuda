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
#include <fstream>
#include <limits>

#include "jacobi_polynomial.h"

int test_jacobi_polynomial()
{
  using namespace dgc;

  // zero order value and derivative
  std::cout << "zero-order jacobi polynomial (alpha = 0.0, beta = 0.0):" << std::endl;
  std::cout << " x = " << -1.0 << ", value = " << jacobi_polynomial_value(0.0, 0.0, 0, -1.0) << ", derivative = " << jacobi_polynomial_derivative(0.0, 0.0, 0, -1.0) << std::endl;
  std::cout << " x = " << 0.0 << ", value = " << jacobi_polynomial_value(0.0, 0.0, 0, 0.0) << ", derivative = " << jacobi_polynomial_derivative(0.0, 0.0, 0, 0.0) << std::endl;
  std::cout << " x = " << 1.0 << ", value = " << jacobi_polynomial_value(0.0, 0.0, 0, 1.0) << ", derivative = " << jacobi_polynomial_derivative(0.0, 0.0, 0, 1.0) << std::endl;

  // two ways of computing derivatives should match
  const int order = 7;
  const int np = 100;
  double delta = 2. / np;

  std::vector<double> data[order + 1];
  std::vector<double> tmp;
  for(int i = 0; i <= np; ++i)
  {
    double x = i * delta - 1.;
    if (x > 1.) x = 1.;

    tmp.clear();
    jacobi_polynomial_derivatives(0., 0., order, x, std::back_inserter(tmp));
    for (int j = 0; j <= order; ++j) data[j].push_back(tmp[j]);
  }

  std::ofstream file;
  file.open("PolynomialDataFile.txt");
  file.precision(std::numeric_limits<double>::digits10 + 1);
  file << "#		x	y" << std::endl;
  for(int j = 0; j <= order; ++j)
  {
    for(int i = 0; i <= np; ++i)
      file << i * delta - 1. << "  " << data[j][i] << std::endl;
    file << std::endl;
  }
  file.close();

  std::vector<double> data1[order + 1];
  for (int j = 0; j <= order; ++j)
    for(int i = 0; i <= np; ++i)
    {
      double x = i * delta - 1.;
      if (x > 1.) x = 1.;
      data1[j].push_back(jacobi_polynomial_derivative(0., 0., j, x));
    }

  file.open("PolynomialDataFile1.txt");
  file.precision(std::numeric_limits<double>::digits10 + 1);
  file << "#		x	y" << std::endl;
  for(int j = 0; j <= order; ++j)
  {
          for(int i = 0; i <= np; ++i)
          {
                  file << i * delta - 1. << "  " << data1[j][i] << std::endl;
          }
          file << std::endl;
  }
  file.close();

  return 0;
}
