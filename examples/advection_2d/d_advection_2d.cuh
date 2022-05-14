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

#include "d_simple_discretization_2d.cuh"
#include "gemv.cuh"


// a simplified version of the advection_2d class on device
template<typename T, typename I>
struct d_advection_2d : public dgc::d_simple_discretization_2d<T, I>
{
  const T* Dr;
  const T* Ds;
  const T* L;

  // process the specified cell 
  __device__ void operator()(std::size_t cid, const T* in, std::size_t size, T t, T* out) const
  {
    // NOTE: never dynamically allocate arrays on device - sooooo slow
    T D[MAX_NUM_CELL_NODES * MAX_NUM_CELL_NODES]; // matrix to apply in the physical coordinate system
    T mFl[3 * MAX_NUM_FACE_NODES]; // mapped numerical fluxes

    // NOTE: hard-coded logic here - this is the reference element stuff on CPU
    int numCellNodes = (this->Order + 1) * (this->Order + 2) / 2;
    int numFaceNodes = this->Order + 1;

    // volume integration
    T invJ0 = this->Inv_Jacobian[cid * 4];
    T invJ1 = this->Inv_Jacobian[cid * 4 + 2];
    for(int i = 0; i < numCellNodes * numCellNodes; ++i)
      D[i] = Dr[i] * invJ0 + Ds[i] * invJ1; 
    // NOTE: coalesed memory access of "in"
    dgc::gemv(D, false, numCellNodes, numCellNodes, (T)(1.0L), in + cid, this->NumCells, (T)(0.0L), out + cid, this->NumCells);

    invJ0 = this->Inv_Jacobian[cid * 4 + 1];
    invJ1 = this->Inv_Jacobian[cid * 4 + 3];
    for(int i = 0; i < numCellNodes * numCellNodes; ++i)
      D[i] = Dr[i] * invJ0 + Ds[i] * invJ1; 
    // NOTE: coalesed memory access of "in"
    dgc::gemv(D, false, numCellNodes, numCellNodes, (T)(1.0L), in + cid, this->NumCells, (T)(1.0L), out + cid, this->NumCells);

    // surface integration
    for (int e = 0; e < 3; ++e)
    {
      I faceIdx = 3 * cid + e;
      I nbCell = this->Interfaces_Cell[faceIdx];
      I nbFace = this->Interfaces_Face[faceIdx];

      const int* faceNodes = e == 0 ? this->Face_0_Nodes : (e == 1 ? this->Face_1_Nodes : this->Face_2_Nodes);
      const int* nbFaceNodes = nbCell == cid ? nullptr : 
                               (nbFace == 0 ? this->Face_0_Nodes : (nbFace == 1 ? this->Face_1_Nodes : this->Face_2_Nodes));

      T faceJ = this->Face_J[faceIdx];
      for (int d = 0; d < numFaceNodes; ++d)
      {
        // NOTE: no more coalesed memory access pattern
        T aU = in[faceNodes[d] * this->NumCells + cid];
        T bU = nbCell == cid ?
               // boundary face: nbFace is actually the offset to m_Boundary_Nodes_X/Y
               std::sin((this->Boundary_Nodes_X[nbFace + d] + (T)(1.0L) - t) * M_PI) *
               std::sin((this->Boundary_Nodes_Y[nbFace + d] + (T)(1.0L) - t) * M_PI) :
               // interior face: flip direction to match with neighbors' d.o.f. - this only works for 2D!
               in[nbFaceNodes[numFaceNodes - d - 1] * this->NumCells + nbCell];
        T nX = this->Outward_Normals_X[faceIdx];
        T nY = this->Outward_Normals_Y[faceIdx];
        T U = (nX + nY >= (T)(0.0L)) ? aU : bU;
        // numerical flux needs to be projected to the outward unit normal of the edge!
        mFl[e * numFaceNodes + d] = U * (nX + nY) * faceJ;
      }
    }
    dgc::gemv(L, false, numCellNodes, 3 * numFaceNodes, - (T)(1.0L) / this->J[cid], mFl, 1, (T)(1.0L), out + cid, this->NumCells);
  }

  // in addition, also need to tell kernel how many cells in total
  __device__ I num_cells() const { return this->NumCells; }
};

#endif
