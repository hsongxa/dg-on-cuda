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

#ifndef D_ADVECTION_2D_CUH
#define D_ADVECTION_2D_CUH

#include <cstddef>
#include <cmath>
#include <math.h>

#include "gemv.cuh"

#define MAX_NUM_CELL_NODES 28
#define MAX_NUM_FACE_NODES 7

// device code which is a simplified version of the advection_2d class on host
template<typename T, typename I>
struct d_advection_2d
{
  const T* m_Dr;
  const T* m_Ds;
  const T* m_L;

  // indices of nodes on the three faces of the reference element
  // these arrays and the matrices above are stored in the constant memory
  // as they do not change
  const int* m_Face_0_Nodes;
  const int* m_Face_1_Nodes;
  const int* m_Face_2_Nodes;

  // mapping of reference element to physical elements
  const T* m_Inv_Jacobian;
  const T* m_J;
  const T* m_Face_J;

  // cell interfaces (mapping of [cell, face] to [nbCell, nbFace])
  // in the case of a boundary face, the mapping becomes [cell, face] to [cell (self), offset-to-m_Boundary_Nodes_X(Y)]
  const I* m_Interfaces_Cell;
  const I* m_Interfaces_Face;

  // other data (those supplied by reference element) such as number of DOFs per element,
  // number of DOFs on each face, ..., etc., can be derived from the approximation order,
  // assuming single element type of triangle
  int m_Order;
  I m_Num_Cells;

  // the only geometry information needed is outward normals of all faces and positions of boundary nodes
  const T* m_Outward_Normals_X;
  const T* m_Outward_Normals_Y;
  const T* m_Boundary_Nodes_X;
  const T* m_Boundary_Nodes_Y;

  // To conduct DG calculations, we need:
  //
  // 0. mapping of cell index => starting positions to the respective variable vectors
  // 1. mapping of cell index => reference element (shape, approximation order) per vairable
  //                          => D & L matrices per variable;
  // 2. mapping of cell index => Jacobian matrix and J per cell and face J per face; 
  // 3. mapping of cell face to cell face;
  // 4. geometry information of faces (outward normals) and boundary nodes if boundary conditions depend on it.
  //
  // For triangle mesh and problems with one scalar variable of fixed approximation order,
  // the above data are sufficient.

  // boundary conditions
  __device__ T bc_dirichlet(T t) const
  { return  0; }

  // numerical flux
  __device__ T numerical_flux(T a, T b) const 
  { return 0; }

  // process the specified cell 
  __device__ void operator()(std::size_t cid, const T* in, std::size_t size, T t, T* out) const
  {
    // NOTE: never dynamically allocate arrays on device - sooooo slow
    T D[MAX_NUM_CELL_NODES * MAX_NUM_CELL_NODES]; // matrix to apply in the physical coordinate system
    T mFl[3 * MAX_NUM_FACE_NODES]; // mapped numerical fluxes
    T sU[MAX_NUM_CELL_NODES];  // output of the surface integration
    for (int i = 0; i < MAX_NUM_CELL_NODES; ++i) sU[i] = 0; // MUST INITIALIZE THIS MEMORY!

    // NOTE: hard-coded logic here - this is the reference element stuff on CPU
    int numCellNodes = (m_Order + 1) * (m_Order + 2) / 2;
    int numFaceNodes = m_Order + 1;

    // volume integration
    T invJ0 = m_Inv_Jacobian[cid * 4];
    T invJ1 = m_Inv_Jacobian[cid * 4 + 2];
    for(int i = 0; i < numCellNodes * numCellNodes; ++i)
      D[i] = m_Dr[i] * invJ0 + m_Ds[i] * invJ1; 
    // NOTE: coalesed memory access of "in"
    dgc::gemv(D, false, numCellNodes, numCellNodes, (T)(1.0L), in + cid, m_Num_Cells, (T)(0.0L), out + cid, m_Num_Cells);

    invJ0 = m_Inv_Jacobian[cid * 4 + 1];
    invJ1 = m_Inv_Jacobian[cid * 4 + 3];
    for(int i = 0; i < numCellNodes * numCellNodes; ++i)
      D[i] = m_Dr[i] * invJ0 + m_Ds[i] * invJ1; 
    // NOTE: coalesed memory access of "in"
    dgc::gemv(D, false, numCellNodes, numCellNodes, (T)(1.0L), in + cid, m_Num_Cells, (T)(1.0L), out + cid, m_Num_Cells);

    // surface integration
    for (int e = 0; e < 3; ++e)
    {
      I faceIdx = 3 * cid + e;
      I nbCell = m_Interfaces_Cell[faceIdx];
      I nbFace = m_Interfaces_Face[faceIdx];

      const int* faceNodes = e == 0 ? m_Face_0_Nodes : (e == 1 ? m_Face_1_Nodes : m_Face_2_Nodes);
      const int* nbFaceNodes = nbCell == cid ? nullptr : 
                               (nbFace == 0 ? m_Face_0_Nodes : (nbFace == 1 ? m_Face_1_Nodes : m_Face_2_Nodes));

      for (int d = 0; d < numFaceNodes; ++d)
      {
        // NOTE: no more coalesed memory access pattern
        T aU = in[faceNodes[d] * m_Num_Cells + cid];
        T bU = nbCell == cid ?
               // boundary face: nbFace is actually the offset to m_Boundary_Nodes_X/Y
               std::sin((m_Boundary_Nodes_X[nbFace + d] + (T)(1.0L) - t) * M_PI) *
               std::sin((m_Boundary_Nodes_Y[nbFace + d] + (T)(1.0L) - t) * M_PI) :
               // interior face: flip direction to match with neighbors' d.o.f. - this only works for 2D!
               in[nbFaceNodes[numFaceNodes - d - 1] * m_Num_Cells + nbCell];
        T nX = m_Outward_Normals_X[faceIdx];
        T nY = m_Outward_Normals_Y[faceIdx];
        T U = (nX + nY >= (T)(0.0L)) ? aU : bU;
        // numerical flux needs to be projected to the outward unit normal of the edge!
        mFl[e * numFaceNodes + d] = U * (nX + nY) * m_Face_J[faceIdx];
      }
    }
    dgc::gemv(m_L, false, numCellNodes, 3 * numFaceNodes, - (T)(1.0L) / m_J[cid], mFl, 1, (T)(0.0L), sU, 1);

    // update the final results
    for(int i = 0; i < numCellNodes; ++i)
      out[i * m_Num_Cells + cid] += sU[i];
  }

  // in addition, also need to tell kernel how many cells in total
  __device__ I num_cells() const { return m_Num_Cells; }
};

#endif
