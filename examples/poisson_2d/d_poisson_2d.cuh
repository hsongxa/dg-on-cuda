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

#ifndef D_POISSON_2D_CUH
#define D_POISSON_2D_CUH 

#include <cstddef>
#include <cmath>
#include <math.h>

#include "d_simple_discretization_2d.cuh"
#include "gemv.cuh"

// the laplace operator is executed in two phases:
// first the gradient and then the divergence, so
// it will need two separate kernels, respectively
struct phase_one {};
struct phase_two {};

// a simplified version of the poisson_2d class on device
template<typename T, typename I>
struct d_poisson_2d : public dgc::d_simple_discretization_2d<T, I>
{
  const T* Dr;
  const T* Ds;
  const T* L;

  // workspaces 
  T* qx;
  T* qy;
  T* du;

  // process the specified cell 
  __device__ void operator()(std::size_t cid, T* inout, std::size_t size) const
  {
    // NOTE: never dynamically allocate arrays on device - sooooo slow
    T D[MAX_NUM_CELL_NODES * MAX_NUM_CELL_NODES]; // matrix to apply in the physical coordinate system
    T toLiftX[3 * MAX_NUM_FACE_NODES]; // face-mapped numerical fluxes
    T toLiftY[3 * MAX_NUM_FACE_NODES]; // face-mapped numerical fluxes

    // NOTE: hard-coded logic here - this is the reference element stuff on CPU
    int numCellNodes = (this->Order + 1) * (this->Order + 2) / 2;
    int numFaceNodes = this->Order + 1;

    // first pass to get (qx, qy)

    // volume integration - coalesed memory access of inout, qx, and qy
    T invJ0 = this->Inv_Jacobian[cid * 4];
    T invJ1 = this->Inv_Jacobian[cid * 4 + 2];
    for(int i = 0; i < numCellNodes * numCellNodes; ++i)
      D[i] = Dr[i] * invJ0 + Ds[i] * invJ1; 
    dgc::gemv(D, false, numCellNodes, numCellNodes, (T)(1.0L), inout + cid, this->NumCells, (T)(0.0L), qx + cid, this->NumCells);

    invJ0 = this->Inv_Jacobian[cid * 4 + 1];
    invJ1 = this->Inv_Jacobian[cid * 4 + 3];
    for(int i = 0; i < numCellNodes * numCellNodes; ++i)
      D[i] = Dr[i] * invJ0 + Ds[i] * invJ1; 
    dgc::gemv(D, false, numCellNodes, numCellNodes, (T)(1.0L), inout + cid, this->NumCells, (T)(0.0L), qy + cid, this->NumCells);

    // surface integration
    T deltaU;
    for (int e = 0; e < 3; ++e)
    {
      I faceIdx = 3 * cid + e;
      I nbCell = this->Interfaces_Cell[faceIdx];
      I nbFace = this->Interfaces_Face[faceIdx];

      const int* faceNodes = e == 0 ? this->Face_0_Nodes : (e == 1 ? this->Face_1_Nodes : this->Face_2_Nodes);
      const int* nbFaceNodes = nbCell == cid ? nullptr : 
                               (nbFace == 0 ? this->Face_0_Nodes : (nbFace == 1 ? this->Face_1_Nodes : this->Face_2_Nodes));

      T faceJ = this->Face_J[faceIdx];
      T nX = this->Outward_Normals_X[faceIdx];
      T nY = this->Outward_Normals_Y[faceIdx];
      for (int d = 0; d < numFaceNodes; ++d)
      {
        // NOTE: no more coalesed memory access pattern
        int inIdxA = faceNodes[d] * this->NumCells + cid;
        int inIdxB = nbCell == cid ? -1 : nbFaceNodes[numFaceNodes - d - 1] * this->NumCells + nbCell;
        T bU = nbCell == cid ? (T)(0.0L) : *(inout + inIdxB); // u = 0 at boundary
        deltaU = *(inout + inIdxA) - bU;

        int outIdx = e * numFaceNodes + d;
        du[3 * cid * numFaceNodes + outIdx] = deltaU; // store du - can the layout be optimized?
        toLiftX[outIdx] = (T)(0.5L) * nX * deltaU * faceJ;
        toLiftY[outIdx] = (T)(0.5L) * nY * deltaU * faceJ;
      }
    }

    T cellJ = this->J[cid];
    dgc::gemv(L, false, numCellNodes, 3 * numFaceNodes, - (T)(1.0L) / cellJ, toLiftX, 1, (T)(1.0L), qx + cid, this->NumCells);
    dgc::gemv(L, false, numCellNodes, 3 * numFaceNodes, - (T)(1.0L) / cellJ, toLiftY, 1, (T)(1.0L), qy + cid, this->NumCells);
/*
/////////////////////////////////////////////////////////////////// BARRIER ! //////////////////////////////////////////// 
    // second pass to get residuals

    // volume integration - coalesed memory access of inout, qx, and qy
    dgc::gemv(D, false, numCellNodes, numCellNodes, (T)(1.0L), qy + cid, this->NumCells, (T)(0.0L), inout + cid, this->NumCells);
    invJ0 = this->Inv_Jacobian[cid * 4];
    invJ1 = this->Inv_Jacobian[cid * 4 + 2];
    for(int i = 0; i < numCellNodes * numCellNodes; ++i)
      D[i] = Dr[i] * invJ0 + Ds[i] * invJ1; 
    dgc::gemv(D, false, numCellNodes, numCellNodes, (T)(1.0L), qx + cid, this->NumCells, (T)(1.0L), inout + cid, this->NumCells);

    // surface integration
    T dqx, dqy;
    for (int e = 0; e < 3; ++e)
    {
      I faceIdx = 3 * cid + e;
      I nbCell = this->Interfaces_Cell[faceIdx];
      I nbFace = this->Interfaces_Face[faceIdx];

      const int* faceNodes = e == 0 ? this->Face_0_Nodes : (e == 1 ? this->Face_1_Nodes : this->Face_2_Nodes);
      const int* nbFaceNodes = nbCell == cid ? nullptr : 
                               (nbFace == 0 ? this->Face_0_Nodes : (nbFace == 1 ? this->Face_1_Nodes : this->Face_2_Nodes));

      T faceJ = this->Face_J[faceIdx];
      T nX = this->Outward_Normals_X[faceIdx];
      T nY = this->Outward_Normals_Y[faceIdx];
      for (int d = 0; d < numFaceNodes; ++d)
      {
        // NOTE: no more coalesed memory access pattern
        int inIdxA = faceNodes[d] * this->NumCells + cid;
        int inIdxB = nbCell == cid ? -1 : nbFaceNodes[numFaceNodes - d - 1] * this->NumCells + nbCell;
        T x = nbCell == cid ? this->Boundary_Nodes_X[nbFace + d] : (T)(0.0L);
        T y = nbCell == cid ? this->Boundary_Nodes_Y[nbFace + d] : (T)(0.0L);
        dqx = nbCell == cid ? M_PI * std::cos(x * M_PI) * std::sin(y * M_PI) : *(qx + inIdxB); // boundary condition
        dqy = nbCell == cid ? M_PI * std::sin(x * M_PI) * std::cos(y * M_PI) : *(qy + inIdxB); // boundary condition
        dqx = *(qx + inIdxA) - dqx;
        dqy = *(qy + inIdxA) - dqy;

        int outIdx = e * numFaceNodes + d;
        toLiftX[outIdx] = (T)(0.5L) * (nX * dqx + nY * dqy + (T)(2.0L) * du[3 * cid * numFaceNodes + outIdx]) * faceJ;
      }
    }

//    T cellJ = this->J[cid];
    dgc::gemv(L, false, numCellNodes, 3 * numFaceNodes, - (T)(1.0L) / cellJ, toLiftX, 1, (T)(1.0L), inout + cid, this->NumCells);
*/  }

  // in addition, also need to tell kernel how many cells in total
  __device__ I num_cells() const { return this->NumCells; }

  // grad(u)
  __device__ void operator()(phase_one phase, std::size_t cid, T* inout, std::size_t size) const
  {/*
    // NOTE: never dynamically allocate arrays on device - sooooo slow
    T D[MAX_NUM_CELL_NODES * MAX_NUM_CELL_NODES]; // matrix to apply in the physical coordinate system
    T toLiftX[3 * MAX_NUM_FACE_NODES]; // face-mapped numerical fluxes
    T toLiftY[3 * MAX_NUM_FACE_NODES]; // face-mapped numerical fluxes

    // NOTE: hard-coded logic here - this is the reference element stuff on CPU
    int numCellNodes = (this->Order + 1) * (this->Order + 2) / 2;
    int numFaceNodes = this->Order + 1;

    // first pass to get (qx, qy)

    // volume integration - coalesed memory access of inout, qx, and qy
    T invJ0 = this->Inv_Jacobian[cid * 4];
    T invJ1 = this->Inv_Jacobian[cid * 4 + 2];
    for(int i = 0; i < numCellNodes * numCellNodes; ++i)
      D[i] = Dr[i] * invJ0 + Ds[i] * invJ1; 
    dgc::gemv(D, false, numCellNodes, numCellNodes, (T)(1.0L), inout + cid, this->NumCells, (T)(0.0L), qx + cid, this->NumCells);

    invJ0 = this->Inv_Jacobian[cid * 4 + 1];
    invJ1 = this->Inv_Jacobian[cid * 4 + 3];
    for(int i = 0; i < numCellNodes * numCellNodes; ++i)
      D[i] = Dr[i] * invJ0 + Ds[i] * invJ1; 
    dgc::gemv(D, false, numCellNodes, numCellNodes, (T)(1.0L), inout + cid, this->NumCells, (T)(0.0L), qy + cid, this->NumCells);

    // surface integration
    T deltaU;
    for (int e = 0; e < 3; ++e)
    {
      I faceIdx = 3 * cid + e;
      I nbCell = this->Interfaces_Cell[faceIdx];
      I nbFace = this->Interfaces_Face[faceIdx];

      const int* faceNodes = e == 0 ? this->Face_0_Nodes : (e == 1 ? this->Face_1_Nodes : this->Face_2_Nodes);
      const int* nbFaceNodes = nbCell == cid ? nullptr : 
                               (nbFace == 0 ? this->Face_0_Nodes : (nbFace == 1 ? this->Face_1_Nodes : this->Face_2_Nodes));

      T faceJ = this->Face_J[faceIdx];
      T nX = this->Outward_Normals_X[faceIdx];
      T nY = this->Outward_Normals_Y[faceIdx];
      for (int d = 0; d < numFaceNodes; ++d)
      {
        // NOTE: no more coalesed memory access pattern
        int inIdxA = faceNodes[d] * this->NumCells + cid;
        int inIdxB = nbCell == cid ? -1 : nbFaceNodes[numFaceNodes - d - 1] * this->NumCells + nbCell;
        T bU = nbCell == cid ? (T)(0.0L) : *(inout + inIdxB); // u = 0 at boundary
        deltaU = *(inout + inIdxA) - bU;

        int outIdx = e * numFaceNodes + d;
        du[3 * cid * numFaceNodes + outIdx] = deltaU; // store du - can the layout be optimized?
        toLiftX[outIdx] = (T)(0.5L) * nX * deltaU * faceJ;
        toLiftY[outIdx] = (T)(0.5L) * nY * deltaU * faceJ;
      }
    }

    T cellJ = this->J[cid];
    dgc::gemv(L, false, numCellNodes, 3 * numFaceNodes, - (T)(1.0L) / cellJ, toLiftX, 1, (T)(1.0L), qx + cid, this->NumCells);
    dgc::gemv(L, false, numCellNodes, 3 * numFaceNodes, - (T)(1.0L) / cellJ, toLiftY, 1, (T)(1.0L), qy + cid, this->NumCells);
  */}

  // div(qx, qy)
  __device__ void operator()(phase_two phase, std::size_t cid, T* inout, std::size_t size) const
  {
    T D[MAX_NUM_CELL_NODES * MAX_NUM_CELL_NODES]; // matrix to apply in the physical coordinate system
    T toLiftX[3 * MAX_NUM_FACE_NODES]; // face-mapped numerical fluxes

    // NOTE: hard-coded logic here - this is the reference element stuff on CPU
    int numCellNodes = (this->Order + 1) * (this->Order + 2) / 2;
    int numFaceNodes = this->Order + 1;

    // volume integration - coalesed memory access of inout, qx, and qy
// TODO: fetch invJ0 and invJ1 first!    
    dgc::gemv(D, false, numCellNodes, numCellNodes, (T)(1.0L), qy + cid, this->NumCells, (T)(0.0L), inout + cid, this->NumCells);
    T invJ0 = this->Inv_Jacobian[cid * 4];
    T invJ1 = this->Inv_Jacobian[cid * 4 + 2];
    for(int i = 0; i < numCellNodes * numCellNodes; ++i)
      D[i] = Dr[i] * invJ0 + Ds[i] * invJ1; 
    dgc::gemv(D, false, numCellNodes, numCellNodes, (T)(1.0L), qx + cid, this->NumCells, (T)(1.0L), inout + cid, this->NumCells);

    // surface integration
    T dqx, dqy;
    for (int e = 0; e < 3; ++e)
    {
      I faceIdx = 3 * cid + e;
      I nbCell = this->Interfaces_Cell[faceIdx];
      I nbFace = this->Interfaces_Face[faceIdx];

      const int* faceNodes = e == 0 ? this->Face_0_Nodes : (e == 1 ? this->Face_1_Nodes : this->Face_2_Nodes);
      const int* nbFaceNodes = nbCell == cid ? nullptr : 
                               (nbFace == 0 ? this->Face_0_Nodes : (nbFace == 1 ? this->Face_1_Nodes : this->Face_2_Nodes));

      T faceJ = this->Face_J[faceIdx];
      T nX = this->Outward_Normals_X[faceIdx];
      T nY = this->Outward_Normals_Y[faceIdx];
      for (int d = 0; d < numFaceNodes; ++d)
      {
        // NOTE: no more coalesed memory access pattern
        int inIdxA = faceNodes[d] * this->NumCells + cid;
        int inIdxB = nbCell == cid ? -1 : nbFaceNodes[numFaceNodes - d - 1] * this->NumCells + nbCell;
        T x = nbCell == cid ? this->Boundary_Nodes_X[nbFace + d] : (T)(0.0L);
        T y = nbCell == cid ? this->Boundary_Nodes_Y[nbFace + d] : (T)(0.0L);
        dqx = nbCell == cid ? M_PI * std::cos(x * M_PI) * std::sin(y * M_PI) : *(qx + inIdxB); // boundary condition
        dqy = nbCell == cid ? M_PI * std::sin(x * M_PI) * std::cos(y * M_PI) : *(qy + inIdxB); // boundary condition
        dqx = *(qx + inIdxA) - dqx;
        dqy = *(qy + inIdxA) - dqy;

        int outIdx = e * numFaceNodes + d;
        toLiftX[outIdx] = (T)(0.5L) * (nX * dqx + nY * dqy + (T)(2.0L) * du[3 * cid * numFaceNodes + outIdx]) * faceJ;
      }
    }

    T cellJ = this->J[cid];
    dgc::gemv(L, false, numCellNodes, 3 * numFaceNodes, - (T)(1.0L) / cellJ, toLiftX, 1, (T)(1.0L), inout + cid, this->NumCells);
  }
};

#endif
