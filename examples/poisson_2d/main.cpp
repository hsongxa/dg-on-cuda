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
#include <algorithm>
#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cusparse.h>
#include <cusolverSp.h>

#include "basic_geom_2d.h"
#include "simple_triangular_mesh_2d.h"
#include "gmsh_importer_exporter.h"
#include "poisson_2d.h"
#if !defined USE_CPU_ONLY
#include "d_poisson_2d.cuh"
#include "k_poisson_2d.cuh"
#endif

template<typename Itr>
double compute_error_norm(Itr it_ref, Itr it, int size)
{
  double err = 0.0;
  for(int i = 0; i < size; ++i)
    err += (it_ref[i] - it[i]) * (it_ref[i] - it[i]);
  return err / size;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  using namespace std;

  const string meshFile = "../square_domain_coarse.msh";
  const auto meshPtr = dgc::gmsh_importer_exporter<double, int>::import_triangle_mesh_2d(meshFile);

  int numCells = meshPtr->num_cells();
  int order = 1;
#if !defined USE_CPU_ONLY
  int blockSize = 256;
#endif
  if (argc > 1)
  {
    order = atoi(argv[1]);
#if !defined USE_CPU_ONLY
    blockSize = atoi(argv[2]);
#endif
  }

  poisson_2d<double, dgc::simple_triangular_mesh_2d<double, int>> op(*meshPtr, order);

  int numNodes = op.total_num_nodes();
  thrust::host_vector<double> ref_u(numNodes);
  thrust::host_vector<double> u(numNodes);
  thrust::host_vector<double> rhs(numNodes);
  op.rhs(rhs.begin()); // populate the rhs
  
#if !defined USE_CPU_ONLY
  thrust::host_vector<double> inv_jacobians;
  thrust::host_vector<double> Js;
  thrust::host_vector<double> face_Js;
  op.fill_cell_mappings(back_inserter(inv_jacobians), back_inserter(Js), back_inserter(face_Js));

  thrust::host_vector<int> interface_cells;
  thrust::host_vector<int> interface_faces;
  op.fill_cell_interfaces(back_inserter(interface_cells), back_inserter(interface_faces));

  thrust::host_vector<double> boundary_node_Xs;
  thrust::host_vector<double> boundary_node_Ys;
  int num_boundary_nodes = op.fill_boundary_nodes(back_inserter(boundary_node_Xs), back_inserter(boundary_node_Ys));

  thrust::host_vector<double> outward_normal_Xs;
  thrust::host_vector<double> outward_normal_Ys;
  op.fill_outward_normals(back_inserter(outward_normal_Xs), back_inserter(outward_normal_Ys));

  thrust::host_vector<double> dummy_host_vector(numCells * 3 * (order + 1));
  thrust::device_vector<double> d_qx = u; // the construction of device_vector by d_qx(numNodes) does not work well
  thrust::device_vector<double> d_qy = u; // in this .cpp file (not .cu file) which is compiled by g++ (not nvcc);
  thrust::device_vector<double> d_du = dummy_host_vector;        // so use this construction-by-assignment instead
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

  d_poisson_2d<double, int>* dOp = create_device_object(numCells, order, op.m_Dr.data(), op.m_Ds.data(), op.m_L.data(),
                                                        d_qx, d_qy, d_du,
                                                        d_face_0_nodes, d_face_1_nodes, d_face_2_nodes,
                                                        d_inv_jacobians, d_Js, d_face_Js,
                                                        d_interface_cells, d_interface_faces, num_boundary_nodes,
                                                        d_boundary_node_Xs, d_boundary_node_Ys,
                                                        d_outward_normal_Xs, d_outward_normal_Ys);
#endif
  auto t0 = chrono::system_clock::now();

  // matrix-filling loop - CUDA linear solvers need the matrix A
  // explicitly as they are direct solvers
  vector<tuple<int, int, double>> aEntries;
  for (size_t i = 0; i < u.size(); ++i)
  {
    std::fill(u.begin(), u.end(), 0.0);
    // TODO: layout for GPU and CPU should be different!
    u[i] = 1.0;

#if defined USE_CPU_ONLY
    op(u.begin());
#else 
    thrust::device_vector<double> d_u = u;
    int blockDim = (numCells + blockSize - 1) / blockSize;
    op_on_device(blockDim, blockSize, d_u, d_u.size(), dOp, numCells);
    u = d_u;
#endif

    for (int j = 0; j < u.size(); ++ j)
      if (abs(u[j]) > 1e-9) aEntries.push_back(make_tuple(j, i, u[j]));
  }
  
  // convert to csr format
  sort(aEntries.begin(), aEntries.end(),
       [](const tuple<int, int, double>& a, const tuple<int, int, double>& b)
       { return get<0>(a) == get<0>(b) ? get<1>(a) < get<1>(b) : get<0>(a) < get<0>(b); });

  vector<int> csrRowPtrA;
  vector<int> csrColIndA;
  vector<double> csrValA;
  int prev_row = -1;
  for(int entryIdx = 0; entryIdx < aEntries.size(); ++entryIdx)
  {
    auto entry = aEntries[entryIdx];
    int row = get<0>(entry);
    if (row == prev_row)
    {
      csrColIndA.push_back(get<1>(entry));
      csrValA.push_back(get<2>(entry));
    }
    else
    {
      assert(row > prev_row);
      for (int i = prev_row; i < row; ++i)
        csrRowPtrA.push_back(csrValA.size());
      prev_row = row;
      entryIdx--; // must use signed int for this index as it may become negative when first enter here
    }
  }
  for (int i = csrRowPtrA.size(); i <= u.size(); ++i)
    csrRowPtrA.push_back(csrValA.size());
  assert(csrValA.size() == aEntries.size());

  auto t1 = chrono::system_clock::now();
  cout << "time used in computing the A matrix: " << chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() << " ms" << endl;
  t0 = chrono::system_clock::now();

  // initialize cuSPARSE
  cusparseHandle_t handle;
  cusparseStatus_t cusparse_status = cusparseCreate(&handle);
  assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

  // descriptor for sparse matrix A
  cusparseMatDescr_t descrA;
  cusparse_status = cusparseCreateMatDescr(&descrA);
  assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

  // cuSPARSE linear solver with LU factorization
  cusolverSpHandle_t solver_handle;
  cusolverStatus_t cusolver_status = cusolverSpCreate(&solver_handle);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

  int singularity;
  cusolver_status = cusolverSpDcsrlsvluHost(solver_handle, u.size(), csrValA.size(),
                                            descrA, csrValA.data(), csrRowPtrA.data(), csrColIndA.data(),
                                            rhs.data(), 0.000001, 0, u.data(), &singularity);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
  assert(singularity < 0);

  cusolver_status = cusolverSpDestroy(solver_handle);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

  t1 = chrono::system_clock::now();
  cout << "time used in solving the linear system: " << chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() << " ms" << endl;

  // exact solution
  op.exact_solution(ref_u.begin());

  // output the error
  double errNorm = compute_error_norm(ref_u.begin(), u.begin(), numNodes);
  cout << "error norm = " << errNorm << endl;

#if !defined USE_CPU_ONLY
  destroy_device_object(dOp);
#endif

  // output to visualize
  //std::ofstream file;
  //file.open("Poisson2DOutputData.txt");
  //file.precision(std::numeric_limits<double>::digits10);
  //file << "#             u" << std::endl;
  //for(int i = 0; i < numNodes; ++i)
  //  file << i << "  " << u[i] << std::endl;
  //file << std::endl;
  //file << "#             ref_u" << std::endl;
  //for(int i = 0; i < numNodes; ++i)
  //  file << i << "  " << ref_u[i] << std::endl;

  return 0;
}
