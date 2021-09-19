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

#ifndef MAPPING_2D_H
#define MAPPING_2D_H

#include <cmath>
#include <cassert>

#include "const_val.h"
#include "basic_geom_2d.h"
#include "dense_matrix.h"

BEGIN_NAMESPACE

// mapping between the reference_triangle and an arbitrary triangle in 2D space
// the number type is also mapped from that used in the supplied geometry to T
template<typename T>
class mapping_2d
{
public:
  template<typename Tri, typename Pnt>
  static Pnt xy_to_rs(const Tri& triangle, const Pnt& xy);

  template<typename Tri, typename Pnt>
  static Pnt rs_to_xy(const Tri& triangle, const Pnt& rs);

  template<typename Tri>
  static dense_matrix<T> jacobian_matrix(const Tri& triangle);

  template<typename Tri>
  static T J(const Tri& triangle);

  template<typename Tri>
  static T face_J(const Tri& triangle, int face);
};

template<typename T> template<typename Tri, typename Pnt>
Pnt mapping_2d<T>::xy_to_rs(const Tri& triangle, const Pnt& xy)
{
  const T v1x = triangle.v0().x();
  const T v1y = triangle.v0().y();
  const T v2x = triangle.v1().x();
  const T v2y = triangle.v1().y();
  const T v3x = triangle.v2().x();
  const T v3y = triangle.v2().y();

  const T x = xy.x();
  const T y = xy.y();
  const T l3 = ((v1y - y) * (v3x - v1x) - (v1x - x) * (v3y - v1y)) / ((v2x - v1x) * ( v3y - v1y) - (v2y - v1y) * (v3x - v1x));
  assert(((v2x - v1x) * ( v3y - v1y) - (v2y - v1y) * (v3x - v1x)) != (const_val<T, 0>));
  const T l1 = std::abs(v3x - v1x) > std::abs(v3y - v1y) ?
                       (x - v1x - l3 * (v2x - v1x)) / (v3x - v1x) :
                       (y - v1y - l3 * (v2y - v1y)) / (v3y - v1y);
  const T l2 = const_val<T, 1> - l3 - l1;

  return Pnt(const_val<T, 1> - const_val<T, 2> * l1 - const_val<T, 2> * l2,
             const_val<T, 1> - const_val<T, 2> * l3 - const_val<T, 2> * l2);
}

template<typename T> template<typename Tri, typename Pnt>
Pnt mapping_2d<T>::rs_to_xy(const Tri& triangle, const Pnt& rs)
{
  const T half = const_val<T, 1> / const_val<T, 2>;

  const T r = rs.x();
  const T s = rs.y();

  const T l1 = half * (s + const_val<T, 1>);
  const T l2 = - half * (r + s);
  const T l3 = half * (r + const_val<T, 1>);

  const Pnt v1 = triangle.v0();
  const Pnt v2 = triangle.v1();
  const Pnt v3 = triangle.v2();

  return Pnt(l2 * v1.x() + l3 * v2.x() + l1 * v3.x(), l2 * v1.y() + l3 * v2.y() + l1 * v3.y());
}

template<typename T> template<typename Tri>
dense_matrix<T> mapping_2d<T>::jacobian_matrix(const Tri& triangle)
{
  const T half = const_val<T, 1> / const_val<T, 2>;

  using Pnt = typename Tri::point_type;
  const Pnt v1 = triangle.v0();
  const Pnt v2 = triangle.v1();
  const Pnt v3 = triangle.v2();

  dense_matrix<T> jacobian(2, 2);
  jacobian(0, 0) = half * (v2.x() - v1.x());
  jacobian(0, 1) = half * (v3.x() - v1.x());
  jacobian(1, 0) = half * (v2.y() - v1.y());
  jacobian(1, 1) = half * (v3.y() - v1.y());

  return jacobian;
}

template<typename T> template<typename Tri>
T mapping_2d<T>::J(const Tri& triangle)
{
  using Pnt = typename Tri::point_type;
  const Pnt v1 = triangle.v0();
  const Pnt v2 = triangle.v1();
  const Pnt v3 = triangle.v2();

  return (const_val<T, 1> / const_val<T, 4>) * ((v2.x() - v1.x()) * (v3.y() - v1.y()) - (v3.x() - v1.x()) * (v2.y() - v1.y()));
}

template<typename T> template<typename Tri>
T mapping_2d<T>::face_J(const Tri& triangle, int face)
{
  assert(face >= 0 && face <= 2);

  using Pnt = typename Tri::point_type;

  Pnt p0, p1;
  switch (face)
  {
    case 0:
      p0 = triangle.v0();
      p1 = triangle.v1();
      break;
    case 1:
      p0 = triangle.v1();
      p1 = triangle.v2();
      break;
    case 2:
      p0 = triangle.v2();
      p1 = triangle.v0();
      break;
  }

  T dx = p0.x() - p1.x();
  T dy = p0.y() - p1.y();
  return (const_val<T, 1> / const_val<T, 2>) * std::sqrt(dx * dx + dy * dy);
}

END_NAMESPACE

#endif
