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
#include <math.h>

#include "gemv.cuh"

#define MAX_NUM_NODES 28
#define MAX_NUM_FACE_NODES 7

// device code which is a simplified version of the advection_2d class on host
template<typename T, typename I>
struct d_advection_2d
{
  T* m_Dr;
  T* m_Ds;
  T* m_L;

  // mapping of reference element to physical elements
  T* m_Inv_Jacobian;
  T* m_J;
  T* m_Face_J;

  // cell interfaces (mapping of [cell, face] to [nbCell, nbFace])
  // in the case of a boundary face, the mapping becomes [cell, face] to [cell (self), offset-to-m_Boundary_Nodes_X(Y)]
  I* m_Interfaces_Cell;
  I* m_Interfaces_Face;

  // other data (those supplied by reference element) such as number of DOFs per element,
  // number of DOFs on each face, ..., etc., can be derived from the approximation order,
  // assuming single element type of triangle
  int m_Order;
  I m_Num_Cells;

  // boundary geometry
  T* m_Boundary_Nodes_X;
  T* m_Boundary_Nodes_Y;

  // To conduct DG calculations, we need:
  //
  // 0. mapping of cell index => starting positions to the respective variable vectors
  // 1. mapping of cell index => reference element (shape, approximation order) per vairable
  //                          => D & L matrices per variable;
  // 2. mapping of cell index => Jacobian matrix and J per cell and face J per face; 
  // 3. mapping of cell face to cell face;
  // 4. geometry information of boundary faces if boundary conditions depend on it.
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
    // coalesced memory access requires a certain layout of entries of "in" and "out"
/*
    // NOTE: mapping #0 and #1
    dgc::gemv(m_D, false, m_NumRows, m_NumRows, -s_waveSpeed, in + cid, m_NumCells, (T)(0.0L), out + cid, m_NumCells);

    // lifting starts from calculating numerical fluxes
    T aL = cid == 0 ? bc_dirichlet(t) : *(in + m_NumCells * (m_NumRows - 1) + cid - 1);
    T aR = *(in + cid);
    T bL = *(in + m_NumCells * (m_NumRows - 1) + cid);
    T bR = cid == m_NumCells - 1 ? bL : *(in + cid + 1);

    // IMPORTANT LEARNING: on device, dynamic memory allocation on heap is many times (>20) slower than static allocation!
    // We could remove the hard-coded size of the static array allocation - for 1D problems what only matters is the first
    // and last entries so we could just allocate two and simplify the following gemv calculation by directly calculating
    // the results of these two entries. But we do not pursue it here as this won't carry to 2D and 3D problems. The general
    // solution is to define a maximum approximation order at compile time and allocate according to that maximum here.
    T inVec[MAX_APPROX_ORDER + 1]; // = (T*)malloc(m_NumRows * sizeof(T));
    T outVec[MAX_APPROX_ORDER + 1];// = (T*)malloc(m_NumRows * sizeof(T));
    for (int i = 0; i < m_NumRows; ++i)
    {
      inVec[i] = (T)(0.0L);
      outVec[i] = (T)(0.0L);
    }
    inVec[0] = numerical_flux(aL, aR) - s_waveSpeed * *(in + cid);
    inVec[m_NumRows - 1] = s_waveSpeed * *(in + m_NumCells * (m_NumRows - 1) + cid) - numerical_flux(bL, bR);

    // NOTE: mapping #0 and #1
    dgc::gemv(m_L, false, m_NumRows, m_NumRows, (T)(1.0L), inVec, 1, (T)(1.0L), outVec, 1);

    for (int j = 0; j < m_NumRows; ++j)
      *(out + m_NumCells * j + cid) += outVec[j];
*/
    // IMPORTANT LEARNING: on device, dynamic memory allocation on heap is many times (>20) slower than static allocation!
    //free(inVec);
    //free(outVec);
  }

  // in addition, also need to tell kernel how many cells in total
  __device__ I num_cells() const { return m_Num_Cells; }
};

#endif
