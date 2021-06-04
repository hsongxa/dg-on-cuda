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

#include "linear_wave_1d.h"

linear_wave_1d::linear_wave_1d(int numCells, int order, bool useWeekForm)
  : m_numCells(numCells), m_order(order), m_useWeekForm(useWeekForm)
{
  reference_element refElem;
  dense_matrix v = refElem.vandermonde_matrix(m_order);
  dense_matrix mInv = v * v.transpose();

  real h = s_domainSize / m_numCells;

  m_L = mInv;
  m_L = m_L * ((real)(2.0L) / h);

  dense_matrix dr = refElem.grad_vandermonde_matrix(m_order) * v.inverse();
  dense_matrix s = mInv.inverse() * dr;
  if (m_useWeekForm)
    m_M = m_L * s.transpose();
  else
    m_M = m_L * s;
}
