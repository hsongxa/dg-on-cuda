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

#include "k_advection_1d.cuh"

#include "explicit_runge_kutta.h"
#include "device_SemiDiscOp_wrapper.h"
#include "k_axpy.cuh"

// NOTE: The sole purpose of this .cu file is to have an entry point to start the nvcc compilation
// NOTE: as all the rest code is in header files only (except the main() function). The reason we
// NOTE: do not instantiate kernel templates in the main() funciton is that .cpp files are compiled
// NOTE: by "g++ -std=c++17" whereas the .cu files are compiled by "nvcc -std=c++14" due to the CUDA
// NOTE: version we use. If we could use c++17 for CUDA code, we wouldn't need this .cu file -- we
// NOTE: could simply instantiate the kernel templates in the main() function and change main.cpp
// NOTE: to main.cu.
void k_advection_1d(int gridSize, int blockSize, double* inout, std::size_t size, double t, double dt,
                    d_advection_1d<double>* d_op, double* wk0, double* wk1, double* wk2, double* wk3, double* wk4)
{ 
  dgc::device_SemiDiscOp_wrapper<d_advection_1d<double>> w;
  w.m_Dop = d_op;
  w.m_GridSize = gridSize;
  w.m_BlockSize = blockSize;

  dgc::rk4(inout, size, t, dt, w, &dgc::k_axpy_auto<double>, wk0, wk1, wk2, wk3, wk4);
}
