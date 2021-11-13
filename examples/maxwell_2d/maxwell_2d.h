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

#ifndef MAXWELL_2D_H
#define MAXWELL_2D_H 

#include <utility>
#include <vector>
#include <math.h>

#include <thrust/tuple.h>

#include "config.h"
#include "dense_matrix.h"
#include "reference_triangle.h"
#include "basic_geom_2d.h"
#include "mapping_2d.h"

// host code that represent the problem of Maxwell's equation in two dimensional space
// using triangle mesh
template<typename T, typename M> // T - number type, M - mesh type
class maxwell_2d
{
public:
  maxwell_2d(const M& mesh, int order); 
  ~maxwell_2d(){}
  
  int total_num_nodes() const; 

  // the layout of DOFs in memory are different for CPU execution and GPU execution;
  // the first iterator sets the DOF positions and the second iterator sets the
  // initial values of the DOFs
  template<typename PosItr, typename ZipItr>
  void initialize_dofs(PosItr posItr, ZipItr zipItr) const;

  // the layout of DOFs in memory are different for CPU execution and GPU execution
  template<typename ZipItr>
  void exact_solution(T t, ZipItr it) const;

  // CPU execution 
  template<typename ConstZipItr, typename ZipItr>
  void operator()(ConstZipItr in, std::size_t size, T t, ZipItr out) const;

  // prepare for GPU execution
  template<typename OutputIterator1, typename OutputIterator2, typename OutputIterator3>
  void fill_cell_mappings(OutputIterator1 it1, OutputIterator2 it2, OutputIterator3 it3) const;
  template<typename OutputIterator1, typename OutputIterator2>
  void fill_cell_interfaces(OutputIterator1 it1, OutputIterator2 it2) const;
  template<typename OutputIterator1, typename OutputIterator2>
  int fill_boundary_nodes(OutputIterator1 it1, OutputIterator2 it2) const; // return the number of boundary nodes
  template<typename OutputIterator1, typename OutputIterator2>
  void fill_outward_normals(OutputIterator1 it1, OutputIterator2 it2) const;

  // need to easily copy these data to device so make them public
  using dense_matrix = dgc::dense_matrix<T, false>; // row major
  dense_matrix m_Dr;
  dense_matrix m_Ds;
  dense_matrix m_L;
  std::vector<int> m_F0_Nodes;
  std::vector<int> m_F1_Nodes;
  std::vector<int> m_F2_Nodes;
  // inverse Jacobian matrix for each element
  std::vector<T> m_invJacobians;

private:
  template<typename ConstZipItr>
  void numerical_fluxes(ConstZipItr input, T t) const; // time t is used for boundary conditions

private:
  using reference_element = dgc::reference_triangle<T>;
  using mapping = dgc::mapping_2d<T>;

  const M*  m_mesh; // no mesh adaptation here so use a constant pointer
  int       m_order;

  // work space for numerical fluxes to avoid repeated allocations
  mutable std::vector<T> m_numericalFlux_Hx;
  mutable std::vector<T> m_numericalFlux_Hy;
  mutable std::vector<T> m_numericalFlux_Ez;
};

template<typename T, typename M>
maxwell_2d<T, M>::maxwell_2d(const M& mesh, int order)
  : m_mesh(std::addressof(mesh)), m_order(order)
{
  // volume integration matrices
  reference_element refElem;
  dense_matrix v = refElem.vandermonde_matrix(m_order);
  dense_matrix mInv = v * v.transpose();
  dense_matrix m = mInv.inverse();

  auto vGrad = refElem.grad_vandermonde_matrix(m_order);
  m_Dr = v * vGrad.first.transpose() * m;
  m_Ds = v * vGrad.second.transpose() * m;

  // face node information
  refElem.face_nodes(m_order, 0, std::back_inserter(m_F0_Nodes));
  refElem.face_nodes(m_order, 1, std::back_inserter(m_F1_Nodes));
  refElem.face_nodes(m_order, 2, std::back_inserter(m_F2_Nodes));

  // surface integration matrix for triangle element
  auto numFaceNodes = refElem.num_face_nodes(m_order);
  dense_matrix mE(refElem.num_nodes(m_order), 3 * numFaceNodes, dgc::const_val<T, 0>);
  for (int e = 0; e < 3; ++e)
  {
    const std::vector<int>& eN = e == 0 ? m_F0_Nodes : (e == 1 ? m_F1_Nodes : m_F2_Nodes);

    dense_matrix eV = refElem.face_vandermonde_matrix(m_order, e);
    dense_matrix eM = eV * eV.transpose();
    eM = eM.inverse();
    for (std::size_t row = 0; row < eM.size_row(); ++row)
      for (std::size_t col = 0; col < eM.size_col(); ++col)
        mE(eN[row], e * numFaceNodes + col) = eM(row, col);
  }
  m_L = mInv * mE;

  // populate mapping
  m_invJacobians.resize(m_mesh->num_cells() * 4);
  for (int c = 0; c < m_mesh->num_cells(); ++c)
  {
    dense_matrix jInv = mapping::jacobian_matrix(m_mesh->get_cell(c));
    jInv = jInv.inverse();
    m_invJacobians[c * 4] = jInv(0, 0);
    m_invJacobians[c * 4 + 1] = jInv(0, 1);
    m_invJacobians[c * 4 + 2] = jInv(1, 0);
    m_invJacobians[c * 4 + 3] = jInv(1, 1);
  }

  // space allocation for numerical fluxes
  m_numericalFlux_Hx.resize(m_mesh->num_cells() * 3 * numFaceNodes);
  m_numericalFlux_Hy.resize(m_mesh->num_cells() * 3 * numFaceNodes);
  m_numericalFlux_Ez.resize(m_mesh->num_cells() * 3 * numFaceNodes);
}

template<typename T, typename M>
int maxwell_2d<T, M>::total_num_nodes() const
{
  reference_element refElem;
  return m_mesh->num_cells() * refElem.num_nodes(m_order);
}

template<typename T, typename M> template<typename PosItr, typename ZipItr>
void maxwell_2d<T, M>::initialize_dofs(PosItr posItr, ZipItr zipItr) const
{
  using point_type = dgc::point_2d<T>;
  using cell_type = typename M::cell_type;

  reference_element refElem;
  std::vector<std::pair<T, T>> pos;
  refElem.node_positions(m_order, std::back_inserter(pos));

  auto itTuple = zipItr.get_iterator_tuple();
  auto itHx = thrust::get<0>(itTuple);
  auto itHy = thrust::get<1>(itTuple);
  auto itEz = thrust::get<2>(itTuple);

#if defined USE_CPU_ONLY
  for (int i = 0; i < m_mesh->num_cells(); ++i)
    for (std::size_t j = 0; j < pos.size(); ++j)
#else
  for (std::size_t j = 0; j < pos.size(); ++j)
    for (int i = 0; i < m_mesh->num_cells(); ++i)
#endif
    {
      const cell_type cell = m_mesh->get_cell(i);
      const point_type pnt = mapping::rs_to_xy(cell, point_type(pos[j].first, pos[j].second)); 
      *posItr++ = pnt;
      *itHx++ = dgc::const_val<T, 0>;
      *itHy++ = dgc::const_val<T, 0>;
      *itEz++ = std::sin(pnt.x() * M_PI) * std::sin(pnt.y() * M_PI);
    }
}

template<typename T, typename M> template<typename ZipItr>
void maxwell_2d<T, M>::exact_solution(T t, ZipItr it) const
{
  using point_type = dgc::point_2d<T>;
  using cell_type = typename M::cell_type;

  reference_element refElem;
  std::vector<std::pair<T, T>> pos;
  refElem.node_positions(m_order, std::back_inserter(pos));

  auto itTuple = it.get_iterator_tuple();
  auto itHx = thrust::get<0>(itTuple);
  auto itHy = thrust::get<1>(itTuple);
  auto itEz = thrust::get<2>(itTuple);

  T sqrt2 = std::sqrt(dgc::const_val<T, 2>);
#if defined USE_CPU_ONLY
  for (int i = 0; i < m_mesh->num_cells(); ++i)
    for (std::size_t j = 0; j < pos.size(); ++j)
#else
  for (std::size_t j = 0; j < pos.size(); ++j)
    for (int i = 0; i < m_mesh->num_cells(); ++i)
#endif
    {
      const cell_type cell = m_mesh->get_cell(i);
      const point_type pnt = mapping::rs_to_xy(cell, point_type(pos[j].first, pos[j].second)); 
      *itHx++ = - std::sin(pnt.x() * M_PI) * std::cos(pnt.y() * M_PI) * std::sin(sqrt2 * M_PI * t) / sqrt2;
      *itHy++ = std::cos(pnt.x() * M_PI) * std::sin(pnt.y() * M_PI) * std::sin(sqrt2 * M_PI * t) / sqrt2;
      *itEz++ = std::sin(pnt.x() * M_PI) * std::sin(pnt.y() * M_PI) * std::cos(sqrt2 * M_PI * t);
    }
}

template<typename T, typename M> template<typename ConstZipItr>
void maxwell_2d<T, M>::numerical_fluxes(ConstZipItr input, T t) const
{
  using point_type = dgc::point_2d<T>;
  using cell_type = typename M::cell_type;

  reference_element refElem;
  std::vector<std::pair<T, T>> pos;
  refElem.node_positions(m_order, std::back_inserter(pos));

  int numCellNodes = refElem.num_nodes(m_order);
  int numFaceNodes = refElem.num_face_nodes(m_order);
  T aHx, bHx, aHy, bHy, aEz, bEz; // "a" is interior or "-" and "b" is exterior or "+"
  T dltHx, dltHy, sumEz, nx, ny;

  // one flux is computed twice from two different cells, rather than for each PAIR of half-edges
  for (int c = 0; c < m_mesh->num_cells(); ++c)
  {
    const cell_type cell = m_mesh->get_cell(c);
    for (int e = 0; e < 3; ++e)
    {
      bool isBoundary;
      int nbCell, nbLocalEdgeIdx;
      std::tie(isBoundary, nbCell, nbLocalEdgeIdx) = m_mesh->get_face_neighbor(c, e);

      const std::vector<int>& faceNodes = e == 0 ? m_F0_Nodes : (e == 1 ? m_F1_Nodes : m_F2_Nodes);
      for (int d = 0; d < numFaceNodes; ++d)
      {
        int inIdxA = c * numCellNodes + faceNodes[d]; // d.o.f. on this edge itself

        // fetch the value on interior
        aHx = thrust::get<0>(input[inIdxA]);
        aHy = thrust::get<1>(input[inIdxA]);
        aEz = thrust::get<2>(input[inIdxA]);

        if(isBoundary)
        {
          bHx = aHx;
          bHy = aHy;
          bEz = - aEz;
        }
        else
        {
          const std::vector<int>& nbFaceNodes = nbLocalEdgeIdx == 0 ? m_F0_Nodes : (nbLocalEdgeIdx == 1 ? m_F1_Nodes : m_F2_Nodes);

          // flip direction to match neighbor's edge d.o.f.'s with this edge's !
          // THIS ONLY WORKS FOR 2D--FOR 3D WE MAY NEED TO GEOMETRICALLY MATCH D.O.F.'S !
          // TODO: spot for future generalization and abstraction
          int inIdxB = nbCell * numCellNodes + nbFaceNodes[numFaceNodes - d - 1];
          assert(inIdxA != inIdxB);

          bHx = thrust::get<0>(input[inIdxB]);
          bHy = thrust::get<1>(input[inIdxB]);
          bEz = thrust::get<2>(input[inIdxB]);

          // extra verification that A and B are geometrically the same point
          // can be commented out wihtout affecting calculations
          point_type A = mapping::rs_to_xy(cell, point_type(pos[faceNodes[d]].first, pos[faceNodes[d]].second));
          point_type B = mapping::rs_to_xy(m_mesh->get_cell(nbCell),
                                           point_type(pos[nbFaceNodes[numFaceNodes - d - 1]].first,
                                                      pos[nbFaceNodes[numFaceNodes - d - 1]].second));
          assert(std::abs(A.x() - B.x()) < 1.0e-10);
          assert(std::abs(A.y() - B.y()) < 1.0e-10);
        }
        
        int outIdx = c * 3 * numFaceNodes + e * numFaceNodes + d;
        point_type n = cell.outward_normal(e);
        nx = n.x();
        ny = n.y();
        dltHx = aHx - bHx;
        dltHy = aHy - bHy;
        sumEz = aEz + bEz;
        m_numericalFlux_Hx[outIdx] = (T)(0.5L) * ny * (sumEz + ny * dltHx - nx * dltHy);
        m_numericalFlux_Hy[outIdx] = - (T)(0.5L) * nx * (sumEz + ny * dltHx - nx * dltHy);
        m_numericalFlux_Ez[outIdx] = (T)(0.5L) * (ny * (aHx + bHx) - nx * (aHy + bHy) + aEz - bEz);
      }
    }
  }
}

template<typename T, typename M> template<typename ConstZipItr, typename ZipItr>
void maxwell_2d<T, M>::operator()(ConstZipItr in, std::size_t size, T t, ZipItr out) const
{
  numerical_fluxes(in, t);

  reference_element refElem;
  int numCellNodes = refElem.num_nodes(m_order);
  int numFaceNodes = refElem.num_face_nodes(m_order);

  // surface-mapped numerical fluxes
  std::vector<T> fHx(3 * numFaceNodes), fHy(3 * numFaceNodes), fEz(3 * numFaceNodes);

  // "un-zip" the iterators
  const auto inTuple = in.get_iterator_tuple();
  const auto inHx = thrust::get<0>(inTuple);
  const auto inHy = thrust::get<1>(inTuple);
  const auto inEz = thrust::get<2>(inTuple);
  auto outTuple = out.get_iterator_tuple();
  auto outHx = thrust::get<0>(outTuple);
  auto outHy = thrust::get<1>(outTuple);
  auto outEz = thrust::get<2>(outTuple);

  for (int c = 0; c < m_mesh->num_cells(); ++c)
  {

    dense_matrix Dx = m_Dr * m_invJacobians[c * 4] + m_Ds * m_invJacobians[c * 4 + 2];
    dense_matrix Dy = m_Dr * m_invJacobians[c * 4 + 1] + m_Ds * m_invJacobians[c * 4 + 3];

    int offsetC = c * numCellNodes;
    Dy.gemv(dgc::const_val<T, 1>, inEz + offsetC, dgc::const_val<T, 0>, outHx + offsetC);
    Dx.gemv(- dgc::const_val<T, 1>, inEz + offsetC, dgc::const_val<T, 0>, outHy + offsetC);
    Dx.gemv(- dgc::const_val<T, 1>, inHy + offsetC, dgc::const_val<T, 0>, outEz + offsetC);
    Dy.gemv(dgc::const_val<T, 1>, inHx + offsetC, dgc::const_val<T, 1>, outEz + offsetC);

    // fetch numerical fluxes and apply face mapping
    const auto cell = m_mesh->get_cell(c);
    int offsetF = c * 3 * numFaceNodes;
    for (int e = 0; e < 3; ++e)
    {
      T faceJ = mapping::face_J(cell, e);
      for (int i = 0; i < numFaceNodes; ++i)
      {
        int ei = e * numFaceNodes + i;
        fHx[ei] = m_numericalFlux_Hx[offsetF + ei] * faceJ;
        fHy[ei] = m_numericalFlux_Hy[offsetF + ei] * faceJ;
        fEz[ei] = m_numericalFlux_Ez[offsetF + ei] * faceJ;
      }
    }

    T cellJ = mapping::J(cell);
    m_L.gemv(- dgc::const_val<T, 1> / cellJ, fHx.cbegin(), dgc::const_val<T, 1>, outHx + offsetC);
    m_L.gemv(- dgc::const_val<T, 1> / cellJ, fHy.cbegin(), dgc::const_val<T, 1>, outHy + offsetC);
    m_L.gemv(- dgc::const_val<T, 1> / cellJ, fEz.cbegin(), dgc::const_val<T, 1>, outEz + offsetC);
  }
}

template<typename T, typename M> template<typename OutputIterator1, typename OutputIterator2, typename OutputIterator3>
void maxwell_2d<T, M>::fill_cell_mappings(OutputIterator1 it1, OutputIterator2 it2, OutputIterator3 it3) const
{
  for (int c = 0; c < m_mesh->num_cells(); ++c)
  {
    *it1++ = m_invJacobians[c * 4];
    *it1++ = m_invJacobians[c * 4 + 1];
    *it1++ = m_invJacobians[c * 4 + 2];
    *it1++ = m_invJacobians[c * 4 + 3];

    const auto cell = m_mesh->get_cell(c);
    *it2++ = mapping::J(cell);
    *it3++ = mapping::face_J(cell, 0);
    *it3++ = mapping::face_J(cell, 1);
    *it3++ = mapping::face_J(cell, 2);
  }
} 

// for interior faces, [cell, face] pair is mapped to [nbCell, nbFace] pair;
// for a boundary face, the cell is mapped to itself and in the place of nbFace is the
// offset to the array of boundary node coordinates populated by fill_boundary_nodes,
// therefore, this function must iterate those boundary faces in the same order as
// in fill_boundary_nodes
template<typename T, typename M> template<typename OutputIterator1, typename OutputIterator2>
void maxwell_2d<T, M>::fill_cell_interfaces(OutputIterator1 it1, OutputIterator2 it2) const
{
  reference_element refElem;
  const int numFaceNodes = refElem.num_face_nodes(m_order);
  int offset = 0;

  bool isBoundary;
  int nbCell, nbLocalEdgeIdx;
  for (int c = 0; c < m_mesh->num_cells(); ++c)
    for (int e = 0; e < 3; ++e)
    {
      std::tie(isBoundary, nbCell, nbLocalEdgeIdx) = m_mesh->get_face_neighbor(c, e);

      if (isBoundary)
      {
        *it1++ = c;
        *it2++ = offset;
        offset += numFaceNodes;
      }
      else
      {
        *it1++ = nbCell;
        *it2++ = nbLocalEdgeIdx;
      }
    }
}

// as noted in above, the order of iterating these boundary faces must be the same as
// in fill_cell_interfaces
template<typename T, typename M> template<typename OutputIterator1, typename OutputIterator2>
int maxwell_2d<T, M>::fill_boundary_nodes(OutputIterator1 it1, OutputIterator2 it2) const
{
  reference_element refElem;
  const int numFaceNodes = refElem.num_face_nodes(m_order);
  int numBoundaryNodes = 0;

  using point_type = dgc::point_2d<T>;
  std::vector<std::pair<T, T>> pos;
  refElem.node_positions(m_order, std::back_inserter(pos));

  bool isBoundary;
  int nbCell, nbLocalEdgeIdx;
  for (int c = 0; c < m_mesh->num_cells(); ++c)
  {
    const auto cell = m_mesh->get_cell(c);
    for (int e = 0; e < 3; ++e)
    {
      std::tie(isBoundary, nbCell, nbLocalEdgeIdx) = m_mesh->get_face_neighbor(c, e);

      if (isBoundary)
      {
        const std::vector<int>& faceNodes = e == 0 ? m_F0_Nodes : (e == 1 ? m_F1_Nodes : m_F2_Nodes);
        for (int i = 0; i < numFaceNodes; ++i)
        {
          point_type xyPos = mapping::rs_to_xy(cell, point_type(pos[faceNodes[i]].first, pos[faceNodes[i]].second));
          *it1++ = xyPos.x();
          *it2++ = xyPos.y();
        }

        numBoundaryNodes += numFaceNodes;
      }
    }
  }

  return numBoundaryNodes;
}

template<typename T, typename M>  template<typename OutputIterator1, typename OutputIterator2>
void maxwell_2d<T, M>::fill_outward_normals(OutputIterator1 it1, OutputIterator2 it2) const
{
  for (int c = 0; c < m_mesh->num_cells(); ++c)
  {
    const auto cell = m_mesh->get_cell(c);
    for (int e = 0; e < 3; ++e)
    {
      dgc::point_2d<T> n = cell.outward_normal(e);
      *it1++ = n.x();
      *it2++ = n.y();
    }
  }
}

#endif
