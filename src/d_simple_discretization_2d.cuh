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

#ifndef D_SIMPLE_DISCRETIZATION_2D
#define D_SIMPLE_DISCRETIZATION_2D

#include <cassert>
#include <cuda_runtime.h>

#include "config.h"

BEGIN_NAMESPACE

// approximation order up to 6
#define MAX_NUM_CELL_NODES 28
#define MAX_NUM_FACE_NODES 7

// Simple discretization on device - "simple" means conformal, single cell shape (i.e.,
// triangle) mesh with same approximation order across all cells, and no adaptation.
// This class is intended to be the base class of concrete discrete-operators on device.
template<typename F, typename I>
struct d_simple_discretization_2d
{
  // To conduct DG calculations, we need:
  //
  // 0. mapping of cell index => starting positions to the respective variable vectors
  // 1. mapping of cell index => reference element
  // 2. mapping of cell index => Jacobian matrix and J per cell and face J per face; 
  // 3. mapping of cell face to cell face;
  // 4. geometry information of faces (outward normals) and boundary nodes if boundary
  //    conditions depend on it.
  //
  // For simple triangle mesh (i.e., no adaptation) and fixed approximation order, the
  // data below are sufficient. All pointers point to device address and this class
  // is not responsible for allocating/deallocating them.

  I NumCells;

  // other data (those supplied by reference element) such as number of DOFs per element,
  // number of DOFs on each face, ..., etc., can be derived from the approximation order,
  // assuming single element type of triangle
  int Order;

  // indices of nodes on the three faces of the reference element
  const int* Face_0_Nodes;
  const int* Face_1_Nodes;
  const int* Face_2_Nodes;

  // mapping of reference element to physical elements
  const F* Inv_Jacobian;
  const F* J;
  const F* Face_J;

  // cell interfaces (mapping of [cell, face] to [nbCell, nbFace])
  // in the case of a boundary face, the mapping becomes
  // [cell, face] to [cell (self), offset-to-Boundary_Nodes_X(Y)]
  const I* Interfaces_Cell;
  const I* Interfaces_Face;

  // the only geometry information needed is outward normals of all
  // faces and positions of boundary nodes
  const F* Outward_Normals_X;
  const F* Outward_Normals_Y;
  const F* Boundary_Nodes_X;
  const F* Boundary_Nodes_Y;
};

template<typename F, typename I>
void
create_simple_discretization_2d_on_device(I num_cells, int order, int* h_face0_nodes, int* h_face1_nodes,
                                          int* h_face2_nodes, F* h_inv_jacobians, F* h_Js, F* h_face_Js,
                                          I* h_interface_cells, I* h_interface_faces, I num_boundary_nodes,
                                          F* h_boundary_node_Xs, F* h_boundary_node_Ys, F* h_outward_normal_Xs,
                                          F* h_outward_normal_Ys, int** d_face0_nodes, int** d_face1_nodes,
                                          int** d_face2_nodes, F** d_inv_jacobians, F** d_Js, F** d_face_Js,
                                          I** d_interface_cells, I** d_interface_faces, F** d_boundary_node_Xs,
                                          F** d_boundary_node_Ys, F** d_outward_normal_Xs, F** d_outward_normal_Ys)
{
  assert(num_cells > 0);
  assert(order > 0 && order < 7);

  cudaMalloc(d_face0_nodes, (order + 1) * sizeof(int));
  cudaMalloc(d_face1_nodes, (order + 1) * sizeof(int));
  cudaMalloc(d_face2_nodes, (order + 1) * sizeof(int));
  cudaMemcpy(*d_face0_nodes, h_face0_nodes, (order + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(*d_face1_nodes, h_face1_nodes, (order + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(*d_face2_nodes, h_face2_nodes, (order + 1) * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc(d_inv_jacobians, num_cells * 4 * sizeof(F));
  cudaMalloc(d_Js, num_cells * sizeof(F));
  cudaMalloc(d_face_Js, num_cells * 3 * sizeof(F));
  cudaMemcpy(*d_inv_jacobians, h_inv_jacobians, num_cells * 4 * sizeof(F), cudaMemcpyHostToDevice);
  cudaMemcpy(*d_Js, h_Js, num_cells * sizeof(F), cudaMemcpyHostToDevice);
  cudaMemcpy(*d_face_Js, h_face_Js, num_cells * 3 * sizeof(F), cudaMemcpyHostToDevice);

  cudaMalloc(d_interface_cells, num_cells * 3 * sizeof(I));
  cudaMalloc(d_interface_faces, num_cells * 3 * sizeof(I));
  cudaMemcpy(*d_interface_cells, h_interface_cells, num_cells * 3 * sizeof(I), cudaMemcpyHostToDevice);
  cudaMemcpy(*d_interface_faces, h_interface_faces, num_cells * 3 * sizeof(I), cudaMemcpyHostToDevice);

  cudaMalloc(d_boundary_node_Xs, num_boundary_nodes * sizeof(F));
  cudaMalloc(d_boundary_node_Ys, num_boundary_nodes * sizeof(F));
  cudaMemcpy(*d_boundary_node_Xs, h_boundary_node_Xs, num_boundary_nodes * sizeof(F), cudaMemcpyHostToDevice);
  cudaMemcpy(*d_boundary_node_Ys, h_boundary_node_Ys, num_boundary_nodes * sizeof(F), cudaMemcpyHostToDevice);
  
  cudaMalloc(d_outward_normal_Xs, num_cells * 3 * sizeof(F));
  cudaMalloc(d_outward_normal_Ys, num_cells * 3 * sizeof(F));
  cudaMemcpy(*d_outward_normal_Xs, h_outward_normal_Xs, num_cells * 3 * sizeof(F), cudaMemcpyHostToDevice);
  cudaMemcpy(*d_outward_normal_Ys, h_outward_normal_Ys, num_cells * 3 * sizeof(F), cudaMemcpyHostToDevice);
}

template<typename F, typename I>
void
destroy_simple_discretization_2d_on_device(int* d_face0_nodes, int* d_face1_nodes, int* d_face2_nodes,
                                           F* d_inv_jacobians, F* d_Js, F* d_face_Js, I* d_interface_cells,
                                           I* d_interface_faces, F* d_boundary_node_Xs, F* d_boundary_node_Ys,
                                           F* d_outward_normal_Xs, F* d_outward_normal_Ys)
{
  cudaFree(d_face0_nodes);
  cudaFree(d_face1_nodes);
  cudaFree(d_face2_nodes);
  cudaFree(d_inv_jacobians);
  cudaFree(d_Js);
  cudaFree(d_face_Js);
  cudaFree(d_interface_cells);
  cudaFree(d_interface_faces);
  cudaFree(d_boundary_node_Xs);
  cudaFree(d_boundary_node_Ys);
  cudaFree(d_outward_normal_Xs);
  cudaFree(d_outward_normal_Ys);
}

END_NAMESPACE

#endif
