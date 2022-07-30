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

#include <cmath>
#include <iostream>
#include <limits>
#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <cuda_runtime.h>

#include "basic_geom_2d.h"
#include "simple_triangular_mesh_2d.h"
#include "gmsh_importer_exporter.h"
#include "stokes_2d.h"
#include "axpy.h"
#include "explicit_runge_kutta.h"
#if !defined USE_CPU_ONLY
#include "d_stokes_2d.cuh"
#include "k_stokes_2d.cuh"
#endif

template<typename ZipItr>
double compute_error_norm(ZipItr it_ref, ZipItr it, int size)
{
  double err = 0.0;
  for(int i = 0; i < size; ++i)
  {
    auto tuple = it[i];
    auto tupleRef = it_ref[i];
    err += (thrust::get<0>(tupleRef) - thrust::get<0>(tuple)) * (thrust::get<0>(tupleRef) - thrust::get<0>(tuple)) +
           (thrust::get<1>(tupleRef) - thrust::get<1>(tuple)) * (thrust::get<1>(tupleRef) - thrust::get<1>(tuple)) +
           (thrust::get<2>(tupleRef) - thrust::get<2>(tuple)) * (thrust::get<2>(tupleRef) - thrust::get<2>(tuple));
  }
  return err / size;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

  const std::string meshFile = "../square_domain_coarse.msh";
  const auto meshPtr = dgc::gmsh_importer_exporter<double, int>::import_triangle_mesh_2d(meshFile);

  int numCells = meshPtr->num_cells();
  int order = 1;
#if !defined USE_CPU_ONLY
  int blockSize = 1024;
#endif
  if (argc > 1)
  {
    order = std::atoi(argv[1]);
#if !defined USE_CPU_ONLY
    blockSize = std::atoi(argv[2]);
#endif
  }

  stokes_2d<double, dgc::simple_triangular_mesh_2d<double, int>> op(*meshPtr, order);
#if !defined USE_CPU_ONLY
  thrust::host_vector<double> inv_jacobians;
  thrust::host_vector<double> Js;
  thrust::host_vector<double> face_Js;
  op.fill_cell_mappings(std::back_inserter(inv_jacobians), std::back_inserter(Js), std::back_inserter(face_Js));

  thrust::host_vector<int> interface_cells;
  thrust::host_vector<int> interface_faces;
  op.fill_cell_interfaces(std::back_inserter(interface_cells), std::back_inserter(interface_faces));

  // for this particular problem we do not need positions of boundary nodes - this is just for completeness
  thrust::host_vector<double> boundary_node_Xs;
  thrust::host_vector<double> boundary_node_Ys;
  int num_boundary_nodes = op.fill_boundary_nodes(std::back_inserter(boundary_node_Xs), std::back_inserter(boundary_node_Ys));

  thrust::host_vector<double> outward_normal_Xs;
  thrust::host_vector<double> outward_normal_Ys;
  op.fill_outward_normals(std::back_inserter(outward_normal_Xs), std::back_inserter(outward_normal_Ys));

  thrust::device_vector<int> d_face_0_nodes = op.F0_Nodes;
  thrust::device_vector<int> d_face_1_nodes = op.F1_Nodes;
  thrust::device_vector<int> d_face_2_nodes = op.F2_Nodes;
  thrust::device_vector<double> d_inv_jacobians = inv_jacobians;
  thrust::device_vector<double> d_Js = Js;
  thrust::device_vector<double> d_face_Js = face_Js;
  thrust::device_vector<int> d_interface_cells = interface_cells;
  thrust::device_vector<int> d_interface_faces = interface_faces;
  thrust::device_vector<double> d_boundary_node_Xs = boundary_node_Xs;
  thrust::device_vector<double> d_boundary_node_Ys = boundary_node_Ys;
  thrust::device_vector<double> d_outward_normal_Xs = outward_normal_Xs;
  thrust::device_vector<double> d_outward_normal_Ys = outward_normal_Ys;

  d_stokes_2d<double, int>* dOp = create_device_object(numCells, order, op.m_Dr.data(), op.m_Ds.data(), op.m_L.data(),
                                                       d_face_0_nodes, d_face_1_nodes, d_face_2_nodes,
                                                       d_inv_jacobians, d_Js, d_face_Js,
                                                       d_interface_cells, d_interface_faces, num_boundary_nodes,
                                                       d_boundary_node_Xs, d_boundary_node_Ys,
                                                       d_outward_normal_Xs, d_outward_normal_Ys);
#endif

  // "H" stands for host
  using HDblIterator = thrust::host_vector<double>::iterator;
  using HIteratorTuple = thrust::tuple<HDblIterator, HDblIterator, HDblIterator>;
  using HZipIterator = thrust::zip_iterator<HIteratorTuple>;

  // node positions and initial conditions
  int numNodes = op.total_num_nodes();
  thrust::host_vector<dgc::point_2d<double>> x(numNodes);
  thrust::host_vector<double> Ux0(numNodes);
  thrust::host_vector<double> Uy0(numNodes);
  thrust::host_vector<double> P(numNodes);
  auto it0 = thrust::make_zip_iterator(thrust::make_tuple(Ux0.begin(), Uy0.begin()));
  auto it0P = thrust::make_zip_iterator(thrust::make_tuple(Ux0.begin(), Uy0.begin(), P.begin()));
  op.initialize_dofs(x.begin(), it0);

  // allocate work space for the Runge-Kutta loop and the reference solution
  thrust::host_vector<double> Ux1(numNodes);
  thrust::host_vector<double> Uy1(numNodes);
  auto it1 = thrust::make_zip_iterator(thrust::make_tuple(Ux1.begin(), Uy1.begin()));
  thrust::host_vector<double> Ux2(numNodes);
  thrust::host_vector<double> Uy2(numNodes);
  auto it2 = thrust::make_zip_iterator(thrust::make_tuple(Ux2.begin(), Uy2.begin()));
  
  // time advancing loop
  const double T = 1;
  double t = 0.0;
  double l = 2.0 / std::sqrt(numCells); // length scale of a typical cell
  double dt = (2.0 / 3.0) * (l / (double)order) * (l / (2.0 + std::sqrt(2.0))); // see p.199
#if !defined USE_CPU_ONLY
  int blockDim = (numCells + blockSize - 1) / blockSize;

  thrust::device_vector<double> d_Ux0 = Ux0;
  thrust::device_vector<double> d_Uy0 = Uy0;
  auto d_it0 = thrust::make_zip_iterator(thrust::make_tuple(d_Ux0.begin(), d_Uy0.begin()));
  thrust::device_vector<double> d_Ux1 = Ux1;
  thrust::device_vector<double> d_Uy1 = Uy1;
  auto d_it1 = thrust::make_zip_iterator(thrust::make_tuple(d_Ux1.begin(), d_Uy1.begin()));
  thrust::device_vector<double> d_Ux2 = Ux2;
  thrust::device_vector<double> d_Uy2 = Uy2;
  auto d_it2 = thrust::make_zip_iterator(thrust::make_tuple(d_Ux2.begin(), d_Uy2.begin()));
  thrust::device_vector<double> d_P = P;
#endif

  auto t0 = std::chrono::system_clock::now();
  while (t < T)
  {
    if (t + dt > T) dt = T - t; // the last increment may be less than the pre-defined value
#if defined USE_CPU_ONLY
    //dgc::rk4(it0, numNodes, t, dt, op, &dgc::axpy_n<double, HZipIterator, HZipIterator>, it1, it2, it3, it4, it5);

#else
    rk4_on_device(blockDim, blockSize, d_it0, numNodes, t, dt, dOp, d_it1, d_it2, d_it3, d_it4, d_it5);
    cudaDeviceSynchronize();
#endif
    t += dt;
  }
  auto t1 = std::chrono::system_clock::now();

  // exact solution
  thrust::host_vector<double> Ux_Ref(numNodes);
  thrust::host_vector<double> Uy_Ref(numNodes);
  thrust::host_vector<double> P_Ref(numNodes);
  auto it_Ref = thrust::make_zip_iterator(thrust::make_tuple(Ux_Ref.begin(), Uy_Ref.begin(), P_Ref.begin()));
  op.exact_solution(t, it_Ref);

  // output the last error
#if !defined USE_CPU_ONLY
  Ux0 = d_Ux0;
  Uy0 = d_Uy0;
  P0 = d_P;
  it0P = thrust::make_zip_iterator(thrust::make_tuple(Ux0.begin(), Uy0.begin(), P0.begin()));
#endif
  double errNorm = compute_error_norm(it_Ref, it0P, numNodes);
  std::cout << "T = " << t << ", error norm = " << errNorm << std::endl;
  std::cout << "time used: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << std::endl;

#if !defined USE_CPU_ONLY
  destroy_device_object(dOp);
#endif

  return 0;
}
