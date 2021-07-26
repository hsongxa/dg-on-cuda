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

#ifndef BASIC_GEOM_2D_H
#define BASIC_GEOM_2D_H

#include <cmath>
#include <cassert>

#include "config.h"

BEGIN_NAMESPACE

template<typename CT>
struct point_2d
{
  using coordinate_type = CT;

  point_2d(CT x, CT y) : m_x(x), m_y(y) {}

  CT x() const { return m_x; }

  CT& x() { return m_x; }

  CT y() const { return m_y; }

  CT& y() { return m_y; }

private:
  CT m_x;
  CT m_y;
};

template<typename PT>
struct segment_2d
{
  segment_2d(PT v0, PT v1) : m_v0(v0), m_v1(v1) {}

  PT v0() const { return m_v0; }

  PT& v0() { return m_v0; }

  PT v1() const { return m_v1; }

  PT& v1() { return m_v1; }

  typename PT::coordinate_type length() const
  {
    return std::sqrt((m_v1.x() - m_v0.x()) * (m_v1.x() - m_v0.x()) +
                     (m_v1.y() - m_v0.y()) * (m_v1.y() - m_v0.y()));
  }

private:
  PT m_v0;
  PT m_v1;
};

template<typename PT>
struct triangle_2d
{
  triangle_2d(PT v0, PT v1, PT v2) : m_v0(v0), m_v1(v1), m_v2(v2) {}

  PT v0() const { return m_v0; }

  PT& v0() { return m_v0; }

  PT v1() const { return m_v1; }

  PT& v1() { return m_v1; }

  PT v2() const { return m_v2; }

  PT& v2() { return m_v2; }

  PT outward_normal(int edge) const;

private:
  PT m_v0;
  PT m_v1;
  PT m_v2;
};

template<typename PT>
PT triangle_2d<PT>::outward_normal(int edge) const
{
  assert(edge >= 0 && edge <= 2);

  using CT = typename PT::coordinate_type;
  CT x, y, x1, y1;
  PT p0 = v0();
  PT p1 = v1();
  PT p2 = v2();

  switch (edge)
  {
    case 0:
      x = p1.x() - p0.x();
      y = p1.y() - p0.y();
      x1 = p2.x() - p0.x();
      y1 = p2.y() - p0.y();
      break;
    case 1:
      x = p2.x() - p1.x();
      y = p2.y() - p1.y();
      x1 = p0.x() - p1.x();
      y1 = p0.y() - p1.y();
      break;
    case 2:
      x = p0.x() - p2.x();
      y = p0.y() - p2.y();
      x1 = p1.x() - p2.x();
      y1 = p1.y() - p2.y();
      break;
  }

  if ((y * x1 - x * y1) > static_cast<CT>(0)) // dot product of normal and the third point
  {
    y = -y;
    x = -x;
  }

  CT l = std::sqrt(y * y + x * x);
  assert(l > static_cast<CT>(0));

  return PT(y / l, - x / l);
}

template<typename CT>
CT dot_product(const point_2d<CT>& vector0, const point_2d<CT>& vector1)
{ return vector0.x() * vector1.x() + vector0.y() * vector1.y(); }

END_NAMESPACE

#endif
