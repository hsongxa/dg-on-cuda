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

#include "k_advection_2d.cuh"

#include "device_SemiDiscOp_wrapper.cuh"
#include "explicit_runge_kutta.h"
#include "k_axpy.cuh"

// NOTE: The sole purpose of this .cu file is to have an entry point to start the nvcc compilation
// NOTE: as all the rest code is in header files only (except the main() function). The reason we
// NOTE: do not instantiate kernel templates in the main() funciton is that .cpp files are compiled
// NOTE: by "g++ -std=c++17" whereas the .cu files are compiled by "nvcc -std=c++14" due to the CUDA
// NOTE: version we use. If we could use c++17 for CUDA code, we wouldn't need this .cu file -- we
// NOTE: could simply instantiate the kernel templates in the main() function and change main.cpp
// NOTE: to main.cu.

__constant__ double c_D[(MAX_APPROX_ORDER + 1) * (MAX_APPROX_ORDER + 1)];
__constant__ double c_L[(MAX_APPROX_ORDER + 1) * (MAX_APPROX_ORDER + 1)];

d_advection_2d<double>* create_device_object(int num_cells, int order, double* m_d, double* m_l)
{
  cudaMemcpyToSymbol(c_D, m_d, (order + 1) * (order + 1) * sizeof(double));
  cudaMemcpyToSymbol(c_L, m_l, (order + 1) * (order + 1) * sizeof(double));

  double* d_D;
  double* d_L;
  cudaGetSymbolAddress((void**)&d_D, c_D);
  cudaGetSymbolAddress((void**)&d_L, c_L);

  d_advection_2d<double> tmp;
  tmp.m_D = d_D;
  tmp.m_L = d_L;
  tmp.m_NumRows = order + 1;
  tmp.m_NumCells = num_cells;

  d_advection_2d<double>* dOp;
  cudaMalloc(&dOp, sizeof(d_advection_2d<double>));
  cudaMemcpy(dOp, &tmp, sizeof(d_advection_2d<double>), cudaMemcpyHostToDevice);

  return dOp;
}

void rk4_on_device(int gridSize, int blockSize, double* inout, std::size_t size, double t, double dt,
                   d_advection_2d<double>* d_op, double* wk0, double* wk1, double* wk2, double* wk3, double* wk4)
{ 
  // NOTE: For the same reason as documented at the beginning of this file, the instantiation of the wrapper object
  // NOTE: has to be here, rather than in the main(). But ideally it should be pulled to the main() and just do the
  // NOTE: instantiation once outside the time advancing loop, instead of repeatedly doing it here at every time step.
  dgc::device_SemiDiscOp_wrapper<d_advection_2d<double>> w;
  w.m_Dop = d_op;
  w.m_GridSize = gridSize;
  w.m_BlockSize = blockSize;

  dgc::rk4(inout, size, t, dt, w, &dgc::k_axpy_auto<double>, wk0, wk1, wk2, wk3, wk4);
}

void destroy_device_object(d_advection_2d<double>* device_obj)
{ cudaFree(device_obj); }

