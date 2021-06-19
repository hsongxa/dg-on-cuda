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

#ifndef D_LINEAR_WAVE_1D_CUH
#define D_LINEAR_WAVE_1D_CUH

#include <cstddef>
#include <math.h>

#include "gemv.cuh"

// device code which is a simplified version of the linear_wave_1d class on host
template<typename T>
struct d_linear_wave_1d
{
  T* m_M;
  T* m_L;

  int m_NumRows;
  int m_NumCols;
  int m_NumCells;

  // Future Generalization -- to conduct DG calculations, what we really need are:
  //
  // 0. mapping of cell index => starting positions to the respective variable vectors
  // 1. mapping of cell index => reference element (shape, approximation order) per vairable
  //                          => these matrices per variable;
  // 2. mapping of cell index => J (geometry);
  // 3. mapping of cell face to cell face;
  //
  // In one dimensional space, #3 is trivial and #1 and #2 degenerates to the data above for
  // uniform mesh and problem with one scalar variable of fixed approximation order. But for
  // higher dimensional spaces and unstructured meshes with h/p adaptivity, we need explicit
  // mappings of all three.

  // problem definitions
  const T s_waveSpeed = (T)(2.L) * (T)(M_PI);
  const T s_alpha = (T)(0.0L); // 0 for upwind and 1 for central flux

  // boundary conditions
  __device__ T bc_dirichlet(T t) const
  { return  - sin(2.0 * M_PI * t); }

  // numerical flux
  __device__ T numerical_flux(T a, T b) const 
  { return s_waveSpeed * (T)(0.5L) * (a + b) + s_waveSpeed * (T)(0.5L) * ((T)(1.0L) - s_alpha) * (a - b); }

  // process the specified cell 
  __device__ void operator()(std::size_t cid, const T* in, std::size_t size, T t, T* out) const
  {
    // NOTE: mapping #0 and #1
    dgc::gemv(m_M, false, m_NumRows, m_NumCols, -s_waveSpeed, in + cid * m_NumRows, (T)(0.0L), out + cid * m_NumRows);

    // lifting starts from calculating numerical fluxes
    T aL = cid == 0 ? bc_dirichlet(t) : *(in + cid * m_NumRows - 1);
    T aR = *(in + cid * m_NumRows);
    T bL = *(in + (cid  + 1) * m_NumRows - 1);
    T bR = cid == m_NumCells - 1 ? bL : *(in + (cid + 1) * m_NumRows);

    // IMPORTANT LEARNING: on device, dynamic memory allocation on heap is many times (>20) slower than static allocation!
    // We could remove the hard-coded size of the static array allocation - for 1D problems what only matters is the first
    // and last entries so we could just allocate two and simplify the following gemv calculation by directly calculating
    // the results of these two entries. But we do not pursue it here as this won't carry to 2D and 3D problems. The general
    // solution is to define a maximum approximation order at compile time and allocate according to that maximum here.
    T inVec[10]; // = (T*)malloc(m_NumRows * sizeof(T));
    T outVec[10];// = (T*)malloc(m_NumRows * sizeof(T));
    for (int i = 0; i < m_NumRows; ++i)
    {
      inVec[i] = (T)(0.0L);
      outVec[i] = (T)(0.0L);
    }
    inVec[0] = numerical_flux(aL, aR) - s_waveSpeed * *(in + (cid * m_NumRows));
    inVec[m_NumRows - 1] = s_waveSpeed * *(in + ((cid + 1) * m_NumRows - 1)) - numerical_flux(bL, bR);

    // NOTE: mapping #0 and #1
    dgc::gemv(m_L, false, m_NumRows, m_NumCols, (T)(1.0L), inVec, (T)(1.0L), outVec);

    for (int j = 0; j < m_NumRows; ++j)
      *(out + (cid * m_NumRows + j)) += outVec[j];

    // IMPORTANT LEARNING: on device, dynamic memory allocation on heap is many times (>20) slower than static allocation!
    //free(inVec);
    //free(outVec);
  }

  // in addition, also need to tell kernel how many cells in total
  __device__ std::size_t num_cells() const { return m_NumCells; }
};

#endif
