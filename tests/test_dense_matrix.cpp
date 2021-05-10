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

#include <iostream>
#include <vector>
#include <iterator>

#include "dense_matrix.h"

template<typename T>
void print_vector(const std::vector<T>& v)
{
	std::cout << '{';
	for (std::size_t i = 0; i < v.size(); ++i)
        {
          std::cout << v[i];
          if (i < v.size() - 1) std::cout << ", ";
        }
        std::cout << '}' << std::endl;
}

int test_dense_matrix()
{
  using namespace dgc;

  // compare size with std::vector
  std::cout << "sizeof dense_matrix<float, true/false>: " << sizeof(dense_matrix<float, true>) << std::endl;
  std::cout << "sizeof std::vector<float>: " << sizeof(std::vector<float>) << std::endl;
  std::cout << std::endl;

  dense_matrix<double, false> m;
  std::cout << "default dense matrix:" << std::endl;
  std::cout << "# rows: " << m.size_row() << std::endl;
  std::cout << "# cols: " << m.size_col() << std::endl;
  
  m.resize(2, 3);
  dense_matrix<double, false> mm(m); // copy construction
  std::cout << "resize-then-copy-construct dense matrix:" << std::endl;
  std::cout << "# rows: " << mm.size_row() << std::endl;
  std::cout << "# cols: " << mm.size_col() << std::endl;

  dense_matrix<double, false> m0(0, 9);
  std::cout << "zero-row-or-col dense matrix:" << std::endl;
  std::cout << "# rows: " << m0.size_row() << std::endl;
  std::cout << "# cols: " << m0.size_col() << std::endl;

  dense_matrix<double, false> m1r(10, 6);
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 6; ++j)
      m1r(i, j) = i * 6 + j;
  std::cout << std::endl << "row major 10x6 matrix: " << std::endl << m1r << std::endl;

  dense_matrix<double, true> m1c(10, 6);
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 6; ++j)
      m1c(i, j) = i * 6 + j;
  std::cout << "column major 10x6 matrix: " << std::endl << m1c << std::endl;

  // initializer list: construction and assignment 
  dense_matrix<double, false> m2r{{}};
  std::cout << "empty-initialized dense matrix:" << std::endl;
  std::cout << "# rows: " << m0.size_row() << std::endl;
  std::cout << "# cols: " << m0.size_col() << std::endl;

  m2r = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}};
  std::cout << "initializer-assigned dense matrix:" << std::endl << m2r << std::endl;

  // move assignment
  dense_matrix<double, false> m3r = std::move(m2r);
  std::cout << "move-assigned dense matrix:" << std::endl << m3r << std::endl;
  std::cout << "dense matrix after moved:" << m2r << std::endl;

  // initializer list in column major
  dense_matrix<double, true> m2c = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}};
  std::cout << "column-major-initialized dense matrix:" << std::endl << m2c;
  std::cout << "# rows: " << m2c.size_row() << std::endl;
  std::cout << "# cols: " << m2c.size_col() << std::endl;

  // move a vector of dense matrices
  std::vector<dense_matrix<double, true>> v1{m1c, m2c};
  std::cout << std::endl << "# of dense matrices in v1: " << v1.size() << std::endl;
  std::cout << "move elements of v1 to v2 one by one..." << std::endl;
  std::vector<dense_matrix<double, true>> v2(std::make_move_iterator(v1.begin()), std::make_move_iterator(v1.end()));
  std::cout << "# of dense matrices in v1: " << v1.size() << std::endl;
  std::cout << v1[0];
  std::cout << v1[1];
  std::cout << "# of dense matrices in v2: " << v2.size() << std::endl;
  std::cout << v2[0];
  std::cout << v2[1];
  std::cout << "copy back the second matrix from v2 to v1..." << std::endl;
  v1[1] = v2[1];
  std::cout << "# of dense matrices in v1: " << v1.size() << std::endl;
  std::cout << v1[0];
  std::cout << v1[1];
  std::cout << "# of dense matrices in v2: " << v2.size() << std::endl;
  std::cout << v2[0];
  std::cout << v2[1];

  std::cout << std::endl << "product of scalar and matrix:" << std::endl << 0.5 * m1r << std::endl;
  std::cout << "product of matrix and scalar:" << std::endl << m1c * 1.5 << std::endl;
  std::cout << "sum of matrix and self:" << std::endl << m1r + m1r << std::endl;

  m(0, 0) = 1.0;
  m(0, 1) = 2.0;
  m(0, 2) = 3.0;
  m(1, 0) = 4.0;
  m(1, 1) = 5.0;
  m(1, 2) = 6.0;
  dense_matrix<double, false> mt = m.transpose();
  std::cout << "transpose of row major [2x3] matrix:" << std::endl << mt;

  dense_matrix<double, false> prod = m * mt;
  std::cout << "product of row major [2x3] and [3x2] matrices:" << std::endl << prod;

  prod = mt * m;
  std::cout << "product of row major [3x2] and [2x3] matrices:" << std::endl << prod << std::endl;

  dense_matrix<double, true> mc = {{1, 4}, {2, 5}, {3, 6}};
  dense_matrix<double, true> mct = mc.transpose();
  std::cout << "transpose of column major [2x3] matrix:" << std::endl << mct;

  dense_matrix<double, true> prodc = mc * mct;
  std::cout << "product of column major [2x3] and [3x2] matrices:" << std::endl << prodc;

  prodc = mct * mc;
  std::cout << "product of column major [3x2] and [2x3] matrices:" << std::endl << prodc << std::endl;
  
  dense_matrix<double, true> m2inv(3, 3);
  m2inv(0, 0) = 1.;
  m2inv(0, 1) = 0.6;
  m2inv(0, 2) = 0.;
  m2inv(1, 0) = 0.;
  m2inv(1, 1) = 1.5;
  m2inv(1, 2) = 1.;
  m2inv(2, 0) = 0.;
  m2inv(2, 1) = 1.;
  m2inv(2, 2) = 1.;
  std::cout << "matrix to invert:" << std::endl << m2inv;
  auto inv2 = m2inv.inverse();
  std::cout << "determinant of the matrix: " << m2inv.determinant() << std::endl;
  std::cout << "inverse of the matrix:" << std::endl << inv2;
  std::cout << "product of the matrix and its inverse:" << std::endl << inv2 * m2inv << std::endl;

  dense_matrix<float, false> m3inv{{5, -2, 2, 7}, {1, 0, 0, 3}, {-3, 1, 5, 0}, {3, -1, -9, 4}};
  std::cout << "matrix to invert:" << std::endl << m3inv;
  auto inv3 = m3inv.inverse();
  std::cout << "determinant of the matrix: " << m3inv.determinant() << std::endl;
  std::cout << "inverse of the matrix:" << std::endl << inv3;
  std::cout << "product of the matrix and its inverse:" << std::endl << inv3 * m3inv << std::endl;

  std::cout << "gemv:" << std::endl;
  std::vector<double> vc{1.0, 2.0, 3.0};
  std::vector<double> vcc{4.0, 5.0};
  std::cout << "vector x:" << std::endl;
  print_vector(vc);
  std::cout << "vector y:" << std::endl;
  print_vector(vcc);
  std::cout << "row major matrix:" << std::endl << m;
  m.gemv(2.0, vc.begin(), 3.0, vcc.begin());
  std::cout << "alpha = 2.0, beta = 3.0" << std::endl;
  std::cout << "result:" << std::endl;
  print_vector(vc);
  print_vector(vcc);
  std::cout << "column major matrix:" << std::endl << mc;
  mc.gemv(2.0, vc.begin(), 3.0, vcc.begin());
  std::cout << "alpha = 2.0, beta = 3.0" << std::endl;
  std::cout << "result:" << std::endl;
  print_vector(vc);
  print_vector(vcc);

  // empty matrix
  dense_matrix<double, false> empty, empty2;
  std::vector<double> emptyv, emptyvv{1.0};
  std::cout << std::endl << "empty matrix: size_row = " << empty.size_row() << ", size_col = " << empty.size_col();
  std::cout << std::endl << empty;
  std::cout << "transpose: " << empty.transpose();
  std::cout << "inverse: " << empty.inverse();
  std::cout << "scalar product: " << 3.3 * empty;
  std::cout << "matrix addition: " << empty2 + empty;
  std::cout << "matrix product: " << empty * empty2;
  empty.gemv(1.0, emptyv.begin(), 2.0, emptyvv.begin());
  std::cout << "gemv result: ";
  print_vector(emptyvv);

  // matrix with just one entry - scalar
  dense_matrix<double, true> scalar{{3.14}}, scalar2{{6.28}};
  std::vector<double> scalarv{2.5}, scalarvv{1.0};
  std::cout << std::endl << "scalar matrix: size_row = " << scalar.size_row() << ", size_col = " << scalar.size_col();
  std::cout << std::endl << scalar;
  std::cout << "transpose: " << scalar.transpose();
  std::cout << "inverse: " << scalar.inverse();
  std::cout << "scalar product: " << scalar * 3.3;
  std::cout << "matrix addition: " << scalar + scalar2;
  std::cout << "matrix product: " << scalar2 * scalar;
  scalar.gemv(1.0, scalarv.begin(), 2.0, scalarvv.begin());
  std::cout << "gemv result: "; 
  print_vector(scalarvv);

  return 0;
}
