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
#include <iterator>

#include "encoded_integer.h"
#include "basic_geom_2d.h"
#include "simple_triangular_mesh_2d.h"
#include "gmsh_importer_exporter.h"

int test_simple_triangular_mesh_2d()
{
  using namespace dgc;

  // unsigned integer with encoding
  encoded_integer<unsigned int, 3> en_0_0(0, 0);
  std::cout << "integer: " << en_0_0.integer() << ", code: " << en_0_0.code() << std::endl;

  encoded_integer<unsigned int, 3> en_0_7(0, 7);
  std::cout << "integer: " << en_0_7.integer() << ", code: " << en_0_7.code() << std::endl;

  encoded_integer<unsigned int, 3> en_1_0(1, 0);
  std::cout << "integer: " << en_1_0.integer() << ", code: " << en_1_0.code() << std::endl;

  encoded_integer<unsigned int, 3> en_1_7(1, 7);
  std::cout << "integer: " << en_1_7.integer() << ", code: " << en_1_7.code() << std::endl;

  encoded_integer<unsigned int, 3> en_536870911_0(536870911, 0);
  std::cout << "integer: " << en_536870911_0.integer() << ", code: " << en_536870911_0.code() << std::endl;

  encoded_integer<unsigned int, 3> en_536870911_7(536870911, 7);
  std::cout << "integer: " << en_536870911_7.integer() << ", code: " << en_536870911_7.code() << std::endl;

  // signed integer with encoding
  encoded_integer<int, 3> sen_0_0(0, 0);
  std::cout << "integer: " << sen_0_0.integer() << ", code: " << sen_0_0.code() << std::endl;

  encoded_integer<int, 3> sen_0_7(0, 7);
  std::cout << "integer: " << sen_0_7.integer() << ", code: " << sen_0_7.code() << std::endl;

  encoded_integer<int, 3> sen_1_0(1, 0);
  std::cout << "integer: " << sen_1_0.integer() << ", code: " << sen_1_0.code() << std::endl;

  encoded_integer<int, 3> sen_1_7(1, 7);
  std::cout << "integer: " << sen_1_7.integer() << ", code: " << sen_1_7.code() << std::endl;

  encoded_integer<int, 3> sen_268435455_0(268435455, 0);
  std::cout << "integer: " << sen_268435455_0.integer() << ", code: " << sen_268435455_0.code() << std::endl;

  encoded_integer<int, 3> sen_268435455_7(268435455, 7);
  std::cout << "integer: " << sen_268435455_7.integer() << ", code: " << sen_268435455_7.code() << std::endl;

  encoded_integer<int, 3> sen_n268435456_0(-268435456, 0);
  std::cout << "integer: " << sen_n268435456_0.integer() << ", code: " << sen_n268435456_0.code() << std::endl;

  encoded_integer<int, 3> sen_n268435456_7(-268435456, 7);
  std::cout << "integer: " << sen_n268435456_7.integer() << ", code: " << sen_n268435456_7.code() << std::endl;

  // 2D geometries
  point_2d<double> p(0.1, 0.2);
  std::cout << "point_2d (" << p.x() << ", " << p.y() << ")" << std::endl;
  p.x() = 0.3;
  p.y() = 0.4;
  std::cout << "point_2d (" << p.x() << ", " << p.y() << ")" << std::endl;

  point_2d<double> q = p;
  segment_2d<point_2d<double>> s(p, q);
  s.v0() = point_2d<double>(0.1, 0.2);
  std::cout << "segment_2d [(" << s.v0().x() << ", " << s.v0().y() << "), (" << s.v1().x() << ", " << s.v1().y() << ")] " << std::endl;
  std::cout << "length of segment: " << s.length() << std::endl;

  point_2d<double> r = p;
  triangle_2d<point_2d<double>> t(p, q, r);
  p.x() = 0.1;
  p.y() = 0.2;
  t.v0() = p;
  t.v2() = point_2d<double>(0.5, 0.7);
  std::cout << "triangle [(" << t.v0().x() << ", " << t.v0().y() << "), (" << t.v1().x() << ", " << t.v1().y() << "), (" << t.v2().x() << ", " << t.v2().y() << ")] "<< std::endl;
  auto n = t.outward_normal(0);
  std::cout << "outward normal of the first edge: (" << n.x() << ", " << n.y() << ")" << std::endl;

  auto dot_prod = dot_product(p, q);
  std::cout << "dot product of (0.1, 0.2) and (0.3, 0.4): " << dot_prod << std::endl;

  // 2D triangle mesh
  double vertices[] = {0, 0, 0.5, 0, 1, 0, 0, 0.5, 0.5, 0.5, 1, 0.5, 0, 1, 0.5, 1, 1, 1};
  int conns[] = {0, 1, 3, 1, 4, 3, 1, 2, 4, 2, 5, 4, 3, 4, 6, 4, 7, 6, 4, 5, 7, 5, 8, 7};

  simple_triangular_mesh_2d<double, int> mesh;
  mesh.fill_vertices(vertices, vertices + 18);
  mesh.fill_connectivity(conns, conns + 24);
  mesh.build_topology();

  std::cout << "2D Triangle mesh:" << std::endl;
  std::cout << "# triangles: " << std::endl;
  for (int j = 0; j < mesh.num_cells(); ++j)
  {
    auto triangle = mesh.get_cell(j);
    std::cout << "[(";
    std::cout << triangle.v0().x() << ", " << triangle.v0().y() << "), (";
    std::cout << triangle.v1().x() << ", " << triangle.v1().y() << "), (";
    std::cout << triangle.v2().x() << ", " << triangle.v2().y() << ")]";
    std::cout << std::endl;
  }

  std::vector<int> cell_interfaces;
  mesh.get_face_mapping(std::back_inserter(cell_interfaces));
  for (int j = 0; j < mesh.num_cells(); ++j)
  {
    std::cout << "neighbor of cell " << j << " at edge 0: (" << cell_interfaces[j * 3] / 4 << ", " << (cell_interfaces[j * 3] & 3) << ")" << std::endl;
    std::cout << "neighbor of cell " << j << " at edge 1: (" << cell_interfaces[j * 3 + 1] / 4 << ", " << (cell_interfaces[j * 3 + 1] & 3) << ")" << std::endl;
    std::cout << "neighbor of cell " << j << " at edge 2: (" << cell_interfaces[j * 3 + 2] / 4 << ", " << (cell_interfaces[j * 3 + 2] & 3) << ")" << std::endl;
  }
  std::cout << std::endl;

  // import/export mesh
  auto mesh_ptr = gmsh_importer_exporter<double, int>::import_triangle_mesh_2d("nonexisting.msh");
  std::cout << "imported mesh from a non-existing MSH file has number of cells = " << mesh_ptr->num_cells() << std::endl;
  std::cout << "imported mesh from a non-existing MSH file has number of vertices = " << mesh_ptr->num_vertices() << std::endl;

  mesh_ptr = gmsh_importer_exporter<double, int>::import_triangle_mesh_2d("square_domain_4cells.msh");
  std::cout << "imported mesh from a non-existing MSH file has number of cells = " << mesh_ptr->num_cells() << std::endl;
  std::cout << "imported mesh from a non-existing MSH file has number of vertices = " << mesh_ptr->num_vertices() << std::endl;
  gmsh_importer_exporter<double, int>::export_triangle_mesh_2d(*mesh_ptr, "exported_square_domain_4cells.msh");
  std::cout << "triangle mesh exported." << std::endl;

  return 0;
}
