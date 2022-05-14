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

#include <cassert>

#include "k_poisson_2d.cuh"
#include "kernel_adapter.cuh"

// NOTE: The sole purpose of this .cu file is to have an entry point to start the nvcc compilation
// NOTE: as all the rest code is in header files only (except the main() function). The reason we
// NOTE: do not instantiate kernel templates in the main() funciton is that .cpp files are compiled
// NOTE: by "g++ -std=c++17" whereas the .cu files are compiled by "nvcc -std=c++14" due to the CUDA
// NOTE: version we use. If we could use c++17 for CUDA code, we wouldn't need this .cu file -- we
// NOTE: could simply instantiate the kernel templates in the main() function and change main.cpp
// NOTE: to main.cu.

__constant__ double Dr[MAX_NUM_CELL_NODES * MAX_NUM_CELL_NODES];
__constant__ double Ds[MAX_NUM_CELL_NODES * MAX_NUM_CELL_NODES];
__constant__ double L[MAX_NUM_CELL_NODES * 3 * MAX_NUM_FACE_NODES];

d_poisson_2d<double, int>* create_device_object(int num_cells, int order, double* dr, double* ds, double* l,
                                                DDblVector& qx, DDblVector& qy, DDblVector& du,
                                                const DIntVector& face_0_nodes,
                                                const DIntVector& face_1_nodes,
                                                const DIntVector& face_2_nodes,
                                                const DDblVector& inv_jacobians,
                                                const DDblVector& Js,
                                                const DDblVector& face_Js,
                                                const DIntVector& interface_cells,
                                                const DIntVector& interface_faces,
                                                int num_boundary_nodes,
                                                const DDblVector& boundary_node_Xs,
                                                const DDblVector& boundary_node_Ys,
                                                const DDblVector& outward_normal_Xs,
                                                const DDblVector& outward_normal_Ys)
{
  assert(num_cells > 0);
  assert(order > 0 && order < 7);

  cudaMemcpyToSymbol(Dr, dr, (order + 1) * (order + 1) * (order + 2) * (order + 2) / 4 * sizeof(double));
  cudaMemcpyToSymbol(Ds, ds, (order + 1) * (order + 1) * (order + 2) * (order + 2) / 4 * sizeof(double));
  cudaMemcpyToSymbol(L, l, (order + 1) * (order + 2) * 3 * (order + 1) / 2 * sizeof(double));

  double *d_Dr, *d_Ds, *d_L;
  cudaGetSymbolAddress((void**)&d_Dr, Dr);
  cudaGetSymbolAddress((void**)&d_Ds, Ds);
  cudaGetSymbolAddress((void**)&d_L, L);
  
  d_poisson_2d<double, int> tmp;
  tmp.NumCells = num_cells;
  tmp.Order = order;
  tmp.Dr = d_Dr;
  tmp.Ds = d_Ds;
  tmp.L = d_L;
  tmp.qx = thrust::raw_pointer_cast(qx.data());
  tmp.qy = thrust::raw_pointer_cast(qy.data());
  tmp.du = thrust::raw_pointer_cast(du.data());
  tmp.Face_0_Nodes = thrust::raw_pointer_cast(face_0_nodes.data());
  tmp.Face_1_Nodes = thrust::raw_pointer_cast(face_1_nodes.data());
  tmp.Face_2_Nodes = thrust::raw_pointer_cast(face_2_nodes.data());
  tmp.Inv_Jacobian = thrust::raw_pointer_cast(inv_jacobians.data());
  tmp.J = thrust::raw_pointer_cast(Js.data());
  tmp.Face_J = thrust::raw_pointer_cast(face_Js.data());
  tmp.Interfaces_Cell = thrust::raw_pointer_cast(interface_cells.data());
  tmp.Interfaces_Face = thrust::raw_pointer_cast(interface_faces.data());
  tmp.Boundary_Nodes_X = thrust::raw_pointer_cast(boundary_node_Xs.data());
  tmp.Boundary_Nodes_Y = thrust::raw_pointer_cast(boundary_node_Ys.data());
  tmp.Outward_Normals_X = thrust::raw_pointer_cast(outward_normal_Xs.data());
  tmp.Outward_Normals_Y = thrust::raw_pointer_cast(outward_normal_Ys.data());

  d_poisson_2d<double, int>* dOp;
  cudaMalloc((void**)&dOp, sizeof(d_poisson_2d<double, int>));
  cudaMemcpy(dOp, &tmp, sizeof(d_poisson_2d<double, int>), cudaMemcpyHostToDevice);

  return dOp;
}

void op_on_device(int gridSize, int blockSize, DDblVector& inout, std::size_t size,
                  d_poisson_2d<double, int>* d_op, std::size_t num_execs)
{ 
  // NOTE: For the same reason as documented at the beginning of this file, the instantiation of the wrapper object
  // NOTE: has to be here, rather than in the main(). But ideally it should be pulled to the main() and just do the
  // NOTE: instantiation once outside the computation loop, instead of repeatedly doing it here at every computation.
  dgc::kernel_adapter<d_poisson_2d<double, int>> k(d_op, gridSize, blockSize, num_execs);
  k.execute_phase(phase_one(), thrust::raw_pointer_cast(inout.data()), size);
  k.execute_phase(phase_two(), thrust::raw_pointer_cast(inout.data()), size);
}

void destroy_device_object(d_poisson_2d<double, int>* device_obj)
{
  cudaFree(device_obj);
}

