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

#include <cassert>

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

__constant__ double c_Dr[MAX_NUM_NODES * MAX_NUM_NODES];
__constant__ double c_Ds[MAX_NUM_NODES * MAX_NUM_NODES];
__constant__ double c_L[MAX_NUM_NODES * 3 * MAX_NUM_FACE_NODES];

d_advection_2d<double, int>* create_device_object(int num_cells, int order, double* dr, double* ds, double* l,
                                                  double* inv_jacobians, double* Js, double* face_Js,
                                                  int* interface_cells, int* interface_faces, int num_boundary_nodes,
                                                  double* boundary_node_Xs, double* boundary_node_Ys,
                                                  double** d_inv_jacobians, double** d_Js, double** d_face_Js,
                                                  int** d_interface_cells, int** d_interface_faces,
                                                  double** d_boundary_node_Xs, double** d_boundary_node_Ys)
{
  assert(num_cells > 0);
  assert(order > 0 && order < 7);

  cudaMemcpyToSymbol(c_Dr, dr, (order + 1) * (order + 1) * (order + 2) * (order + 2) / 4 * sizeof(double));
  cudaMemcpyToSymbol(c_Ds, ds, (order + 1) * (order + 1) * (order + 2) * (order + 2) / 4 * sizeof(double));
  cudaMemcpyToSymbol(c_L, l, (order + 1) * (order + 2) * 3 * (order + 1) / 2 * sizeof(double));

  double *d_Dr, *d_Ds, *d_L;
  cudaGetSymbolAddress((void**)&d_Dr, c_Dr);
  cudaGetSymbolAddress((void**)&d_Ds, c_Ds);
  cudaGetSymbolAddress((void**)&d_L, c_L);

  cudaMalloc(d_inv_jacobians, num_cells * 4 * sizeof(double));
  cudaMalloc(d_Js, num_cells * sizeof(double));
  cudaMalloc(d_face_Js, num_cells * 3 * sizeof(double));
  cudaMemcpy(*d_inv_jacobians, inv_jacobians, num_cells * 4 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(*d_Js, Js, num_cells * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(*d_face_Js, face_Js, num_cells * 3 * sizeof(double), cudaMemcpyHostToDevice);

  cudaMalloc(d_interface_cells, num_cells * 3 * sizeof(int));
  cudaMemcpy(*d_interface_cells, interface_cells, num_cells * 3 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(d_interface_faces, num_cells * 3 * sizeof(int));
  cudaMemcpy(*d_interface_faces, interface_faces, num_cells * 3 * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc(d_boundary_node_Xs, num_boundary_nodes * sizeof(double));
  cudaMemcpy(*d_boundary_node_Xs, boundary_node_Xs, num_boundary_nodes * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc(d_boundary_node_Ys, num_boundary_nodes * sizeof(double));
  cudaMemcpy(*d_boundary_node_Ys, boundary_node_Ys, num_boundary_nodes * sizeof(double), cudaMemcpyHostToDevice);
  
  d_advection_2d<double, int> tmp;
  tmp.m_Dr = d_Dr;
  tmp.m_Ds = d_Ds;
  tmp.m_L = d_L;
  tmp.m_Inv_Jacobian = *d_inv_jacobians;
  tmp.m_J = *d_Js;
  tmp.m_Face_J = *d_face_Js;
  tmp.m_Interfaces_Cell = *d_interface_cells;
  tmp.m_Interfaces_Face = *d_interface_faces;
  tmp.m_Boundary_Nodes_X = *d_boundary_node_Xs;
  tmp.m_Boundary_Nodes_Y = *d_boundary_node_Ys;
  tmp.m_Order = order;
  tmp.m_Num_Cells = num_cells;

  d_advection_2d<double, int>* dOp;
  cudaMalloc(&dOp, sizeof(d_advection_2d<double, int>));
  cudaMemcpy(dOp, &tmp, sizeof(d_advection_2d<double, int>), cudaMemcpyHostToDevice);

  return dOp;
}

void rk4_on_device(int gridSize, int blockSize, double* inout, std::size_t size, double t, double dt,
                   d_advection_2d<double, int>* d_op, double* wk0, double* wk1, double* wk2, double* wk3, double* wk4)
{ 
  // NOTE: For the same reason as documented at the beginning of this file, the instantiation of the wrapper object
  // NOTE: has to be here, rather than in the main(). But ideally it should be pulled to the main() and just do the
  // NOTE: instantiation once outside the time advancing loop, instead of repeatedly doing it here at every time step.
  dgc::device_SemiDiscOp_wrapper<d_advection_2d<double, int>> w;
  w.m_Dop = d_op;
  w.m_GridSize = gridSize;
  w.m_BlockSize = blockSize;

  dgc::rk4(inout, size, t, dt, w, &dgc::k_axpy_auto<double>, wk0, wk1, wk2, wk3, wk4);
}

void destroy_device_object(d_advection_2d<double, int>* device_obj, double* d_inv_jacobians, double* d_Js, double* d_face_Js,
                           int* d_interface_cells, int* d_interface_faces, double* d_boundary_node_Xs, double* d_boundary_node_Ys)
{
  cudaFree(d_inv_jacobians); 
  cudaFree(d_Js); 
  cudaFree(d_face_Js); 
  cudaFree(d_interface_cells); 
  cudaFree(d_interface_faces); 
  cudaFree(d_boundary_node_Xs); 
  cudaFree(d_boundary_node_Ys); 

  cudaFree(device_obj);
}

