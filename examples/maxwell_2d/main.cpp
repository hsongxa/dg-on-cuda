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
#include <thrust/iterator/zip_iterator.h>
#include <cuda_runtime.h>

#include "basic_geom_2d.h"
#include "simple_triangular_mesh_2d.h"
#include "gmsh_importer_exporter.h"
#include "maxwell_2d.h"
#include "axpy.h"
#include "explicit_runge_kutta.h"
#if !defined USE_CPU_ONLY
#include "d_maxwell_2d.cuh"
#include "k_maxwell_2d.cuh"
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

  const std::string meshFile = "square_domain_coarse.msh";
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

  maxwell_2d<double, dgc::simple_triangular_mesh_2d<double, int>> op(*meshPtr, order);
#if !defined USE_CPU_ONLY
  std::vector<double> inv_jacobians;
  std::vector<double> Js;
  std::vector<double> face_Js;
  op.fill_cell_mappings(std::back_inserter(inv_jacobians), std::back_inserter(Js), std::back_inserter(face_Js));

  std::vector<int> interface_cells;
  std::vector<int> interface_faces;
  op.fill_cell_interfaces(std::back_inserter(interface_cells), std::back_inserter(interface_faces));

  // for this particular problem we do not need positions of boundary nodes - this is just for completeness
  std::vector<double> boundary_node_Xs;
  std::vector<double> boundary_node_Ys;
  int num_boundary_nodes = op.fill_boundary_nodes(std::back_inserter(boundary_node_Xs), std::back_inserter(boundary_node_Ys));

  std::vector<double> outward_normal_Xs;
  std::vector<double> outward_normal_Ys;
  op.fill_outward_normals(std::back_inserter(outward_normal_Xs), std::back_inserter(outward_normal_Ys));

  double *d_inv_jacobians, *d_Js, *d_face_Js, *d_boundary_node_Xs, *d_boundary_node_Ys, *d_outward_normal_Xs, *d_outward_normal_Ys;
  int *d_face0_nodes, *d_face1_nodes, *d_face2_nodes, *d_interface_cells, *d_interface_faces;
  d_maxwell_2d<double, int>* dOp = create_device_object(numCells, order, op.m_Dr.data(), op.m_Ds.data(), op.m_L.data(),
                                                          op.F0_Nodes.data(), op.F1_Nodes.data(), op.F2_Nodes.data(),
                                                          inv_jacobians.data(), Js.data(), face_Js.data(),
                                                          interface_cells.data(), interface_faces.data(), num_boundary_nodes,
                                                          boundary_node_Xs.data(), boundary_node_Ys.data(),
                                                          outward_normal_Xs.data(), outward_normal_Ys.data(), &d_face0_nodes,
                                                          &d_face1_nodes, &d_face2_nodes, &d_inv_jacobians, &d_Js, &d_face_Js,
                                                          &d_interface_cells, &d_interface_faces, &d_boundary_node_Xs,
                                                          &d_boundary_node_Ys, &d_outward_normal_Xs, &d_outward_normal_Ys);
#endif

  using HDblIterator = thrust::host_vector<double>::iterator;
  using HIteratorTuple = thrust::tuple<HDblIterator, HDblIterator, HDblIterator>;
  using HZipIterator = thrust::zip_iterator<HIteratorTuple>;

  // node positions and initial conditions
  int numNodes = op.total_num_nodes();
  thrust::host_vector<dgc::point_2d<double>> x(numNodes);
  thrust::host_vector<double> Hx0(numNodes);
  thrust::host_vector<double> Hy0(numNodes);
  thrust::host_vector<double> Ez0(numNodes);
  auto it0 = thrust::make_zip_iterator(thrust::make_tuple(Hx0.begin(), Hy0.begin(), Ez0.begin()));
  op.initialize_dofs(x.begin(), it0);

  // allocate work space for the Runge-Kutta loop and the reference solution
  thrust::host_vector<double> Hx1(numNodes);
  thrust::host_vector<double> Hy1(numNodes);
  thrust::host_vector<double> Ez1(numNodes);
  auto it1 = thrust::make_zip_iterator(thrust::make_tuple(Hx1.begin(), Hy1.begin(), Ez1.begin()));
  thrust::host_vector<double> Hx2(numNodes);
  thrust::host_vector<double> Hy2(numNodes);
  thrust::host_vector<double> Ez2(numNodes);
  auto it2 = thrust::make_zip_iterator(thrust::make_tuple(Hx2.begin(), Hy2.begin(), Ez2.begin()));
  thrust::host_vector<double> Hx3(numNodes);
  thrust::host_vector<double> Hy3(numNodes);
  thrust::host_vector<double> Ez3(numNodes);
  auto it3 = thrust::make_zip_iterator(thrust::make_tuple(Hx3.begin(), Hy3.begin(), Ez3.begin()));
  thrust::host_vector<double> Hx4(numNodes);
  thrust::host_vector<double> Hy4(numNodes);
  thrust::host_vector<double> Ez4(numNodes);
  auto it4 = thrust::make_zip_iterator(thrust::make_tuple(Hx4.begin(), Hy4.begin(), Ez4.begin()));
  thrust::host_vector<double> Hx5(numNodes);
  thrust::host_vector<double> Hy5(numNodes);
  thrust::host_vector<double> Ez5(numNodes);
  auto it5 = thrust::make_zip_iterator(thrust::make_tuple(Hx5.begin(), Hy5.begin(), Ez5.begin()));
  thrust::host_vector<double> Hx_Ref(numNodes);
  thrust::host_vector<double> Hy_Ref(numNodes);
  thrust::host_vector<double> Ez_Ref(numNodes);
  auto it_Ref = thrust::make_zip_iterator(thrust::make_tuple(Hx_Ref.begin(), Hy_Ref.begin(), Ez_Ref.begin()));
  
  // time advancing loop
  const double T = 1;
  double t = 0.0;
  double l = 2.0 / std::sqrt(numCells); // length scale of a typical cell
  double dt = (2.0 / 3.0) * (l / (double)order) * (l / (2.0 + std::sqrt(2.0))); // see p.199
#if !defined USE_CPU_ONLY
  int blockDim = (numCells + blockSize - 1) / blockSize;
#endif

  auto t0 = std::chrono::system_clock::now();
  while (t < T)
  {
    if (t + dt > T) dt = T - t; // the last increment may be less than the pre-defined value
#if defined USE_CPU_ONLY
    dgc::rk4(it0, numNodes, t, dt, op, &dgc::axpy_n<double, HZipIterator, HZipIterator>, it1, it2, it3, it4, it5);
#else
//    rk4_on_device(blockDim, blockSize, v, numNodes, t, dt, dOp, v1, v2, v3, v4, v5);
//    cudaDeviceSynchronize();
#endif
    t += dt;
  }
  auto t1 = std::chrono::system_clock::now();

  // exact solution
  op.exact_solution(t, it_Ref);

  // output the last error
  double errNorm = compute_error_norm(it_Ref, it0, numNodes);
  std::cout << "T = " << t << ", error norm = " << errNorm << std::endl;
  std::cout << "time used: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << std::endl;

#if !defined USE_CPU_ONLY
  destroy_device_object(dOp, d_face0_nodes, d_face1_nodes, d_face2_nodes, d_inv_jacobians, d_Js, d_face_Js,
                        d_interface_cells, d_interface_faces, d_boundary_node_Xs, d_boundary_node_Ys,
                        d_outward_normal_Xs, d_outward_normal_Ys);
#endif

  return 0;
}
