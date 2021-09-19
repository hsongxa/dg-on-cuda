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
#include <vector>
#include <iostream>
#include <fstream>
#include <limits>
#include <chrono>

#include <cuda_runtime.h>

#include "basic_geom_2d.h"
#include "simple_triangular_mesh_2d.h"
#include "gmsh_importer_exporter.h"
#include "advection_2d.h"
#include "explicit_runge_kutta.h"
#if !defined USE_CPU_ONLY
#include "d_advection_2d.cuh"
#include "k_advection_2d.cuh"
#endif

double compute_error_norm(double* ref_v, double* v, int size)
{
  double err = 0.0;
  for(int i = 0; i < size; ++i)
    err += (ref_v[i] - v[i]) * (ref_v[i] - v[i]);
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

  advection_2d<double, dgc::simple_triangular_mesh_2d<double, int>> op(*meshPtr, order);
#if !defined USE_CPU_ONLY
  d_advection_1d<double>* dOp = create_device_object(numCells, order, op.m_V.data(), op.m_L.data());
#endif

  // DOF positions and initial conditions
  int numDOFs = op.num_dofs();
  std::vector<dgc::point_2d<double>> x(numDOFs);
  double* v;
  cudaMallocManaged(&v, numDOFs * sizeof(double)); // unified memory
  op.initialize_dofs(x.begin(), v);

  // allocate work space for the Runge-Kutta loop and the reference solution
  double* v1;
  double* v2;
  double* v3;
  double* v4;
  double* v5;
  double* ref_v;
  cudaMallocManaged(&v1, numDOFs * sizeof(double));
  cudaMallocManaged(&v2, numDOFs * sizeof(double));
  cudaMallocManaged(&v3, numDOFs * sizeof(double));
  cudaMallocManaged(&v4, numDOFs * sizeof(double));
  cudaMallocManaged(&v5, numDOFs * sizeof(double));
  cudaMallocManaged(&ref_v, numDOFs * sizeof(double));
  
  // time advancing loop
  const int totalTSs = 10000;
  double t = 0.0;
  double l = 2.0 / std::sqrt(numCells); // length scale of a typical cell
  double dt = (2.0 / 3.0) * (l / (double)order) * (l / (2.0 + std::sqrt(2.0))); // see p.199
#if !defined USE_CPU_ONLY
  int blockDim = (numCells + blockSize - 1) / blockSize;
#endif

  auto t0 = std::chrono::system_clock::now();
  for (int i = 0; i < totalTSs; ++i)
  {
#if defined USE_CPU_ONLY
    dgc::rk4(v, numDOFs, t, dt, op, &dgc::axpy_n<const double*, double, double*>, v1, v2, v3, v4, v5);
#else
    rk4_on_device(blockDim, blockSize, v, numDOFs, t, dt, dOp, v1, v2, v3, v4, v5);
    cudaDeviceSynchronize();
#endif
    t += dt;
  }
  auto t1 = std::chrono::system_clock::now();

  // exact solution
  op.exact_solution(t, ref_v);

  // output the last error
  double errNorm = compute_error_norm(ref_v, v, numDOFs);
  std::cout << "t = " << t << ", error norm = " << errNorm << std::endl;
  std::cout << "time used: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << std::endl;

  // output to visualize
  std::ofstream file;
  file.open("Advection2DDataFile.txt");
  file.precision(std::numeric_limits<double>::digits10);
  file << "#         x         y         u" << std::endl;
  for(int i = 0; i < numDOFs; ++i)
    file << x[i].x() << "  " << x[i].y() << "  " << v[i] << std::endl;
  file << std::endl;
  file << "#         x         y         reference solution" << std::endl;
  for(int i = 0; i < numDOFs; ++i)
    file << x[i].x() << "  " << x[i].y() << "  " << ref_v[i] << std::endl;

  cudaFree(v);
  cudaFree(v1);
  cudaFree(v2);
  cudaFree(v3);
  cudaFree(v4);
  cudaFree(v5);
  cudaFree(ref_v);
#if !defined USE_CPU_ONLY
  destroy_device_object(dOp);
#endif

  return 0;
}
