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

#include "reference_triangle.h"

int test_reference_triangle()
{
  using namespace dgc;

  reference_triangle<double> refTri;

  // node positions
  std::vector<std::pair<double, double>> nodes;
  std::vector<std::size_t> face_nodes;
  nodes.clear();
  std::cout << "reference triangle of order 1:" << std::endl;
  std::cout << "number of nodes: " << refTri.num_nodes(1) << std::endl;
  refTri.node_positions(1, std::back_inserter(nodes));
  for (std::size_t i = 0; i < nodes.size(); ++i)
    std::cout << "(" << nodes[i].first << ", " << nodes[i].second << ")" << std::endl;

  std::cout << "face nodes: " << std::endl;
  face_nodes.clear();
  refTri.face_nodes(1, 0, std::back_inserter(face_nodes));
  std::cout << "face 0:"; 
  for(std::size_t i = 0; i < face_nodes.size(); ++i)  std::cout << " " << face_nodes[i];
  std::cout << std::endl;

  face_nodes.clear();
  refTri.face_nodes(1, 1, std::back_inserter(face_nodes));
  std::cout << "face 1:"; 
  for(std::size_t i = 0; i < face_nodes.size(); ++i)  std::cout << " " << face_nodes[i];
  std::cout << std::endl;

  face_nodes.clear();
  refTri.face_nodes(1, 2, std::back_inserter(face_nodes));
  std::cout << "face 2:"; 
  for(std::size_t i = 0; i < face_nodes.size(); ++i)  std::cout << " " << face_nodes[i];
  std::cout << std::endl;

  nodes.clear();
  std::cout << std::endl << "reference triangle of order 4:" << std::endl;
  std::cout << "number of nodes: " << refTri.num_nodes(4) << std::endl;
  refTri.node_positions(4, std::back_inserter(nodes));
  for (std::size_t i = 0; i < nodes.size(); ++i)
    std::cout << "(" << nodes[i].first << ", " << nodes[i].second << ")" << std::endl;

  std::cout << "face nodes: " << std::endl;
  face_nodes.clear();
  refTri.face_nodes(4, 0, std::back_inserter(face_nodes));
  std::cout << "face 0:"; 
  for(std::size_t i = 0; i < face_nodes.size(); ++i)  std::cout << " " << face_nodes[i];
  std::cout << std::endl;

  face_nodes.clear();
  refTri.face_nodes(4, 1, std::back_inserter(face_nodes));
  std::cout << "face 1:"; 
  for(std::size_t i = 0; i < face_nodes.size(); ++i)  std::cout << " " << face_nodes[i];
  std::cout << std::endl;

  face_nodes.clear();
  refTri.face_nodes(4, 2, std::back_inserter(face_nodes));
  std::cout << "face 2:"; 
  for(std::size_t i = 0; i < face_nodes.size(); ++i)  std::cout << " " << face_nodes[i];
  std::cout << std::endl;

  nodes.clear();
  std::cout << std::endl << "reference triangle of order 6:" << std::endl;
  std::cout << "number of nodes: " << refTri.num_nodes(6) << std::endl;
  refTri.node_positions(6, std::back_inserter(nodes));
  for (std::size_t i = 0; i < nodes.size(); ++i)
    std::cout << "(" << nodes[i].first << ", " << nodes[i].second << ")" << std::endl;

  std::cout << "face nodes: " << std::endl;
  face_nodes.clear();
  refTri.face_nodes(6, 0, std::back_inserter(face_nodes));
  std::cout << "face 0:"; 
  for(std::size_t i = 0; i < face_nodes.size(); ++i)  std::cout << " " << face_nodes[i];
  std::cout << std::endl;

  face_nodes.clear();
  refTri.face_nodes(6, 1, std::back_inserter(face_nodes));
  std::cout << "face 1:"; 
  for(std::size_t i = 0; i < face_nodes.size(); ++i)  std::cout << " " << face_nodes[i];
  std::cout << std::endl;

  face_nodes.clear();
  refTri.face_nodes(6, 2, std::back_inserter(face_nodes));
  std::cout << "face 2:"; 
  for(std::size_t i = 0; i < face_nodes.size(); ++i)  std::cout << " " << face_nodes[i];
  std::cout << std::endl;

  // face vandermonde matrix
  std::cout << std::endl << "face vandermonde matrix of order 1:" << std::endl;
  std::cout << "face 0:" << std::endl << refTri.face_vandermonde_matrix(1, 0);
  std::cout << "face 1:" << std::endl << refTri.face_vandermonde_matrix(1, 1);
  std::cout << "face 2:" << std::endl << refTri.face_vandermonde_matrix(1, 2);

  std::cout << std::endl << "face vandermonde matrix of order 4:" << std::endl;
  std::cout << "face 0:" << std::endl << refTri.face_vandermonde_matrix(4, 0);
  std::cout << "face 1:" << std::endl << refTri.face_vandermonde_matrix(4, 1);
  std::cout << "face 2:" << std::endl << refTri.face_vandermonde_matrix(4, 2);

  std::cout << std::endl << "face vandermonde matrix of order 6:" << std::endl;
  std::cout << "face 0:" << std::endl << refTri.face_vandermonde_matrix(6, 0);
  std::cout << "face 1:" << std::endl << refTri.face_vandermonde_matrix(6, 1);
  std::cout << "face 2:" << std::endl << refTri.face_vandermonde_matrix(6, 2);

  // vandermonde matrix
  std::cout << std::endl << "vandermonde matrix of order 1:" << std::endl;
  std::cout << refTri.vandermonde_matrix(1);

  std::cout << "vandermonde matrix of order 4:" << std::endl;
  std::cout << refTri.vandermonde_matrix(4);

  std::cout << "vandermonde matrix of order 6:" << std::endl;
  std::cout << refTri.vandermonde_matrix(6);

  // gradient vandermonde matrix
  std::cout << std::endl << "gradient vandermonde matrix of order 1:" << std::endl;
  std::cout << "v_r:" << std::endl << refTri.grad_vandermonde_matrix(1).first;
  std::cout << "v_s:" << std::endl << refTri.grad_vandermonde_matrix(1).second;

  std::cout << "gradient vandermonde matrix of order 4:" << std::endl;
  std::cout << "v_r:" << std::endl << refTri.grad_vandermonde_matrix(4).first;
  std::cout << "v_s:" << std::endl << refTri.grad_vandermonde_matrix(4).second;

  std::cout << "gradient vandermonde matrix of order 6:" << std::endl;
  std::cout << "v_r:" << std::endl << refTri.grad_vandermonde_matrix(6).first;
  std::cout << "v_s:" << std::endl << refTri.grad_vandermonde_matrix(6).second;

  // TODO: could create custom basis and node policies and test them...

  return 0;
}
