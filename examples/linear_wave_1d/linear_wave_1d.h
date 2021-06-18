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

#ifndef LINEAR_WAVE_1D_H
#define LINEAR_WAVE_1D_H

#include <vector>
#include <iterator>
#include <algorithm>
#include <math.h>

#include "dense_matrix.h"
#include "reference_segment.h"

// host code that represent the problem of linear wave equation in one dimensional space
template<typename T>
class linear_wave_1d
{
public:
  linear_wave_1d(int numCells, int order); 
  ~linear_wave_1d(){}
  
  T wave_speed() const { return s_waveSpeed; }

  // reference element and mapping
  T min_elem_size() const { return s_domainSize / m_numCells; }

  template<typename OutputIterator>
  void dof_positions(OutputIterator it) const;

  // put it all together: it is a discrete operator
  template<typename ConstItr, typename Itr>
  void operator()(ConstItr in_cbegin, std::size_t size, T t, Itr out_begin) const;

  // need to easily copy these matrices to device so make them public
  using dense_matrix = dgc::dense_matrix<T, false>; // row major
  dense_matrix m_M;
  dense_matrix m_L;

private:
  template<typename ConstItr>
  void numerical_fluxes(ConstItr cbegin, T t) const; // time t is used for boundary conditions

  // numerical flux
  T numerical_flux(T a, T b) const 
  { return s_waveSpeed * (T)(0.5L) * (a + b) + s_waveSpeed * (T)(0.5L) * ((T)(1.0L) - s_alpha) * (a - b); }

private:
  using reference_element = dgc::reference_segment<T>;

  // numerical flux constant
  T s_alpha = (T)(0.0L); // 1.0 for central flux and 0.0 for unwind flux

  // numerical scheme data (could be constants if never change)
  int m_numCells;
  int m_order;

  // problem definitions
  const T s_domainSize = (T)(2.L) * (T)(M_PI);
  const T s_waveSpeed = (T)(2.L) * (T)(M_PI);

  // work space for numerical fluxes to avoid repeated allocations
  mutable std::vector<T> m_numericalFluxes;
};

template<typename T>
linear_wave_1d<T>::linear_wave_1d(int numCells, int order)
  : m_numCells(numCells), m_order(order)
{
  reference_element refElem;
  dense_matrix v = refElem.vandermonde_matrix(m_order);
  dense_matrix mInv = v * v.transpose();

  T h = s_domainSize / m_numCells;

  m_L = mInv;
  m_L = m_L * ((T)(2.0L) / h);

  dense_matrix dr = refElem.grad_vandermonde_matrix(m_order) * v.inverse();
  dense_matrix s = mInv.inverse() * dr;
  m_M = m_L * s;
}

template<typename T> template<typename OutputIterator>
void linear_wave_1d<T>::dof_positions(OutputIterator it) const
{
  reference_element refElem;
  std::vector<T> pos;
  refElem.node_positions(m_order, std::back_inserter(pos));

  // could be extracted to a class of mapping
  T h = s_domainSize / m_numCells;
  for (int i = 0; i < m_numCells; ++i)
    for (int j = 0; j < pos.size(); ++j)
      it = i * h + ((T)(1.L) + pos[j]) * (T)(0.5L) * h;
}

template<typename T> template<typename ConstItr>
void linear_wave_1d<T>::numerical_fluxes(ConstItr cbegin, T t) const
{
  int numFluxes = m_numCells + 1;
  if (m_numericalFluxes.size() < numFluxes) m_numericalFluxes.resize(numFluxes);

  T a, b;
  int np = m_order + 1; // d.o.f. management !
  for (int i = 0; i < numFluxes; ++i)
  {
    if (i > 0) a = *(cbegin + (i * np - 1));
    else a = - sin(2.0 * M_PI * t); // inflow boundary condition
    if (i < numFluxes - 1) b = *(cbegin + (i * np));
    else b = *(cbegin + (i * np - 1)); // outflow boundary condition - alternatively, may be set to zero ?
    m_numericalFluxes[i] = numerical_flux(a, b);
  }
}

template<typename T> template<typename ConstItr, typename Itr>
void linear_wave_1d<T>::operator()(ConstItr in_cbegin, std::size_t size, T t, Itr out_begin) const
{
  numerical_fluxes(in_cbegin, t);

  int np = m_order + 1; // scalar unknown variable
  std::vector<T> in_vec(np); // recalculated below for each cell
  std::vector<T> out_vec(np); // recalculated below for each cell

  for (int cell = 0; cell < m_numCells; ++cell)
  {
    m_M.gemv(-s_waveSpeed, in_cbegin + (cell * np), (T)(0.0L), out_begin + (cell * np));

    std::fill(in_vec.begin(), in_vec.end(), (T)(0.0L));
    std::fill(out_vec.begin(), out_vec.end(), (T)(0.0L));
    in_vec[0] = m_numericalFluxes[cell] - s_waveSpeed * *(in_cbegin + (cell * np));
    in_vec[np - 1] = s_waveSpeed * *(in_cbegin + ((cell + 1) * np - 1)) - m_numericalFluxes[cell + 1];
    m_L.gemv((T)(1.0L), &in_vec[0], (T)(1.0L), &out_vec[0]);

    for (int j = 0; j < np; ++j)
      *(out_begin + (cell * np + j)) += out_vec[j];
  }
}

#endif
