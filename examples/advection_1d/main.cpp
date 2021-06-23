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

#include <cstdlib>
#include <vector>
#include <iterator>
#include <iostream>
#include <fstream>
#include <limits>
#include <chrono>

#include <cuda_runtime.h>

#include "advection_1d.h"
#include "explicit_runge_kutta.h"
#if !defined USE_CPU_ONLY
#include "d_advection_1d.cuh"
#include "k_advection_1d.cuh"
#endif

double compute_error_norm(std::vector<double>& x, double* v, double t)
{
  double err = 0.0;
  for(int i = 0; i < x.size(); ++i)
    err += (std::sin(x[i] - 2. * M_PI * t) - v[i]) * (std::sin(x[i] - 2. * M_PI * t) - v[i]);
  return err / x.size();
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

  int numCells = 1024 * 8;
  int order = 6;
  if (argc > 1)
  {
    numCells = std::atoi(argv[1]);
    order = std::atoi(argv[2]);
  }

  advection_1d<double> op(numCells, order);
#if !defined USE_CPU_ONLY
  // create the device object (on device)
  d_advection_1d<double>* dOp;
  cudaMalloc(&dOp, sizeof(d_advection_1d<double>));
  double* dMM;
  double* dML;
  cudaMalloc(&dMM, (order + 1) * (order + 1) * sizeof(double));
  cudaMalloc(&dML, (order + 1) * (order + 1) * sizeof(double));
  cudaMemcpy(dMM, op.m_M.data(), (order + 1) * (order + 1) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dML, op.m_L.data(), (order + 1) * (order + 1) * sizeof(double), cudaMemcpyHostToDevice);

  d_advection_1d<double> tmp;
  tmp.m_M = dMM;
  tmp.m_L = dML;
  tmp.m_NumRows = order + 1;
  tmp.m_NumCols = order + 1;
  tmp.m_NumCells = numCells;
  cudaMemcpy(dOp, &tmp, sizeof(d_advection_1d<double>), cudaMemcpyHostToDevice);
#endif

  std::vector<double> x;
  op.dof_positions(std::back_inserter(x));

  // initial condition
  int numDOFs = numCells * (order +  1);
  double* v;
  cudaMallocManaged(&v, numDOFs * sizeof(double)); // unified memory
  for (int i = 0; i < numDOFs; ++i) v[i] = std::sin(x[i]);

  // allocate work space for Runge-Kutta loop
  double* v1;
  double* v2;
  double* v3;
  double* v4;
  double* v5;
  cudaMallocManaged(&v1, numDOFs * sizeof(double));
  cudaMallocManaged(&v2, numDOFs * sizeof(double));
  cudaMallocManaged(&v3, numDOFs * sizeof(double));
  cudaMallocManaged(&v4, numDOFs * sizeof(double));
  cudaMallocManaged(&v5, numDOFs * sizeof(double));
  
  // time advancing loop
  int totalTSs = 10000;
  double t = 0.0;
  double dt = 0.25 / order / order * op.min_elem_size() / op.wave_speed();
  int blockSize = 1024;
  int blockDim = (numCells + blockSize - 1) / blockSize;
  auto t0 = std::chrono::system_clock::now();
  for (int i = 0; i < totalTSs; ++i)
  {
#if defined USE_CPU_ONLY
    dgc::rk4(v, numDOFs, t, dt, op, &dgc::axpy_n<const double*, double, double*>, v1, v2, v3, v4, v5);
#else
    k_advection_1d(blockDim, blockSize, v, numDOFs, t, dt, dOp, v1, v2, v3, v4, v5);
    cudaDeviceSynchronize();
#endif
    t += dt;
  }
  auto t1 = std::chrono::system_clock::now();

  // output the last error
  double errNorm = compute_error_norm(x, v, t);
  std::cout << "t = " << t << ", error norm = " << errNorm << std::endl;
  std::cout << "time used: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << std::endl;

  // output to visualize
  std::ofstream file;
  file.open("Advection1DDataFile.txt");
  file.precision(std::numeric_limits<double>::digits10);
  file << "#         x         y" << std::endl;
  for(int i = 0; i < numDOFs; ++i)
    file << x[i] << "  " << v[i] << std::endl;
  file << std::endl;
  file << "#         x         reference solution" << std::endl;
  for(int i = 0; i < numDOFs; ++i)
    file << x[i] << " " << std::sin(x[i] - op.wave_speed() * t) << std::endl;

  cudaFree(v);
  cudaFree(v1);
  cudaFree(v2);
  cudaFree(v3);
  cudaFree(v4);
  cudaFree(v5);
#if !defined USE_CPU_ONLY
  cudaFree(dOp);
  cudaFree(dMM);
  cudaFree(dML);
#endif

  return 0;
}
