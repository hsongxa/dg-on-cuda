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

#ifndef D_MAXWELL_2D_CUH
#define D_MAXWELL_2D_CUH

#include <cstddef>
#include <cmath>
#include <math.h>

#include "d_simple_discretization_2d.cuh"
#include "gemv.cuh"


// a simplified version of the maxwell_2d class on device
template<typename T, typename I>
struct d_maxwell_2d : public dgc::d_simple_discretization_2d<T, I>
{
  const T* Dr;
  const T* Ds;
  const T* L;

  // "D" stands for device
  using DDblIterator = thrust::device_vector<double>::iterator;
  using DIteratorTuple = thrust::tuple<DDblIterator, DDblIterator, DDblIterator>;
  using DZipIterator = thrust::zip_iterator<DIteratorTuple>;

  // process the specified cell 
  __device__ void operator()(std::size_t cid, DZipIterator in, std::size_t size, T t, DZipIterator out) const
  {
    // NOTE: never dynamically allocate arrays on device - sooooo slow
    T D[MAX_NUM_CELL_NODES * MAX_NUM_CELL_NODES]; // matrix to apply in the physical coordinate system
    T fHx[3 * MAX_NUM_FACE_NODES]; // face-mapped numerical fluxes
    T fHy[3 * MAX_NUM_FACE_NODES]; // face-mapped numerical fluxes
    T fEz[3 * MAX_NUM_FACE_NODES]; // face-mapped numerical fluxes

    // NOTE: hard-coded logic here - this is the reference element stuff on CPU
    int numCellNodes = (this->Order + 1) * (this->Order + 2) / 2;
    int numFaceNodes = this->Order + 1;

    // "un-zip" the iterators
    const auto inTuple = in.get_iterator_tuple();
    const auto inHx = thrust::get<0>(inTuple);
    const auto inHy = thrust::get<1>(inTuple);
    const auto inEz = thrust::get<2>(inTuple);
    auto outTuple = out.get_iterator_tuple();
    auto outHx = thrust::get<0>(outTuple);
    auto outHy = thrust::get<1>(outTuple);
    auto outEz = thrust::get<2>(outTuple);

    // volume integration
    T invJ0 = this->Inv_Jacobian[cid * 4];
    T invJ1 = this->Inv_Jacobian[cid * 4 + 2];
    for(int i = 0; i < numCellNodes * numCellNodes; ++i)
      D[i] = Dr[i] * invJ0 + Ds[i] * invJ1; 
    // NOTE: coalesed memory access of "in"
    dgc::gemv(D, false, numCellNodes, numCellNodes, - (T)(1.0L), inEz + cid, this->NumCells, (T)(0.0L), outHy + cid, this->NumCells);
    dgc::gemv(D, false, numCellNodes, numCellNodes, - (T)(1.0L), inHy + cid, this->NumCells, (T)(0.0L), outEz + cid, this->NumCells);

    invJ0 = this->Inv_Jacobian[cid * 4 + 1];
    invJ1 = this->Inv_Jacobian[cid * 4 + 3];
    for(int i = 0; i < numCellNodes * numCellNodes; ++i)
      D[i] = Dr[i] * invJ0 + Ds[i] * invJ1; 
    // NOTE: coalesed memory access of "in"
    dgc::gemv(D, false, numCellNodes, numCellNodes, (T)(1.0L), inEz + cid, this->NumCells, (T)(0.0L), outHx + cid, this->NumCells);
    dgc::gemv(D, false, numCellNodes, numCellNodes, (T)(1.0L), inHx + cid, this->NumCells, (T)(1.0L), outEz + cid, this->NumCells);

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
        int inIdxA = faceNodes[d] * this->NumCells + cid;
        T aHx = *(inHx + inIdxA);
        T aHy = *(inHy + inIdxA);
        T aEz = *(inEz + inIdxA);

        int inIdxB = nbCell == cid ? -1 : nbFaceNodes[numFaceNodes - d - 1] * this->NumCells + nbCell;
        T bHx = nbCell == cid ? aHx : *(inHx + inIdxB);
        T bHy = nbCell == cid ? aHy : *(inHy + inIdxB);
        T bEz = nbCell == cid ? - aEz : *(inEz + inIdxB);

        int outIdx = e * numFaceNodes + d;
        T nX = this->Outward_Normals_X[faceIdx];
        T nY = this->Outward_Normals_Y[faceIdx];
        T dltHx = aHx - bHx;
        T dltHy = aHy - bHy;
        T sumEz = aEz + bEz;
        // numerical flux needs to be projected to the outward unit normal of the edge!
        fHx[outIdx] = (T)(0.5L) * nY * (sumEz + nY * dltHx - nX * dltHy) * faceJ;
        fHy[outIdx] = - (T)(0.5L) * nX * (sumEz + nY * dltHx - nX * dltHy) * faceJ;
        fEz[outIdx] = (T)(0.5L) * (nY * (aHx + bHx) - nX * (aHy + bHy) + aEz - bEz) * faceJ;
      }
    }

    T cellJ = this->J[cid];
    dgc::gemv(L, false, numCellNodes, 3 * numFaceNodes, - (T)(1.0L) / cellJ, fHx, 1, (T)(1.0L), outHx + cid, this->NumCells);
    dgc::gemv(L, false, numCellNodes, 3 * numFaceNodes, - (T)(1.0L) / cellJ, fHy, 1, (T)(1.0L), outHy + cid, this->NumCells);
    dgc::gemv(L, false, numCellNodes, 3 * numFaceNodes, - (T)(1.0L) / cellJ, fEz, 1, (T)(1.0L), outEz + cid, this->NumCells);
  }

  // in addition, also need to tell kernel how many cells in total
  __device__ I num_cells() const { return this->NumCells; }
};

#endif
