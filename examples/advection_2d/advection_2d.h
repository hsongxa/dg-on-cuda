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

#ifndef ADVECTION_2D_H
#define ADVECTION_2D_H 

#include <utility>
#include <vector>
#include <math.h>

#include "config.h"
#include "dense_matrix.h"
#include "reference_triangle.h"
#include "basic_geom_2d.h"
#include "mapping_2d.h"

// host code that represent the problem of linear advection equation in two dimensional space
// using triangle mesh
template<typename T, typename M> // T - number type, M - mesh type
class advection_2d
{
public:
  advection_2d(const M& mesh, int order); 
  ~advection_2d(){}
  
  int num_dofs() const; 

  // the layout of DOFs in memory are different for CPU execution and GPU execution;
  // the first iterator sets the DOF positions and the second iterator sets the
  // initial values of the DOFs
  template<typename OutputIterator1, typename OutputIterator2>
  void initialize_dofs(OutputIterator1 it1, OutputIterator2 it2) const;

  // the layout of DOFs in memory are different for CPU execution and GPU execution
  template<typename OutputIterator>
  void exact_solution(T t, OutputIterator it) const;

  // CPU execution 
  template<typename ConstItr, typename Itr>
  void operator()(ConstItr in_cbegin, std::size_t size, T t, Itr out_begin) const;

  // need to easily copy these data to device so make them public
  using dense_matrix = dgc::dense_matrix<T, false>; // row major
  dense_matrix m_Dr;
  dense_matrix m_Ds;
  dense_matrix m_L;
  // inverse Jacobian matrix for each element
  std::vector<T> m_invJacobians;

private:
  template<typename ConstItr>
  void numerical_fluxes(ConstItr cbegin, T t) const; // time t is used for boundary conditions

private:
  using reference_element = dgc::reference_triangle<T>;
  using mapping = dgc::mapping_2d<T>;

  const M*  m_mesh; // no mesh adaptation here so use a constant pointer
  int       m_order;

  // work space for numerical fluxes to avoid repeated allocations
  mutable std::vector<T> m_numericalFluxes;
};

template<typename T, typename M>
advection_2d<T, M>::advection_2d(const M& mesh, int order)
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

  // surface integration matrix for triangle element
  auto numFaceNodes = refElem.num_face_nodes(m_order);
  dense_matrix mE(refElem.num_nodes(m_order), 3 * numFaceNodes, dgc::const_val<T, 0>);
  for (int e = 0; e < 3; ++e)
  {
    std::vector<int> eN;
    refElem.face_nodes(m_order, e, std::back_inserter(eN));
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
  m_numericalFluxes.resize(m_mesh->num_cells() * 3 * numFaceNodes);
}

template<typename T, typename M>
int advection_2d<T, M>::num_dofs() const
{
  reference_element refElem;
  // scalar variable - one DOF per node
  return m_mesh->num_cells() * refElem.num_nodes(m_order);
}

template<typename T, typename M> template<typename OutputIterator1, typename OutputIterator2>
void advection_2d<T, M>::initialize_dofs(OutputIterator1 it1, OutputIterator2 it2) const
{
  using point_type = dgc::point_2d<T>;
  using cell_type = typename M::cell_type;

  reference_element refElem;
  std::vector<std::pair<T, T>> pos;
  refElem.node_positions(m_order, std::back_inserter(pos));

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
      *it1++ = pnt;
      *it2++ = std::sin((pnt.x() + dgc::const_val<T, 1>) * M_PI) * std::sin((pnt.y() + dgc::const_val<T, 1>) * M_PI);
    }
}

template<typename T, typename M> template<typename OutputIterator>
void advection_2d<T, M>::exact_solution(T t, OutputIterator it) const
{
  using point_type = dgc::point_2d<T>;
  using cell_type = typename M::cell_type;

  reference_element refElem;
  std::vector<std::pair<T, T>> pos;
  refElem.node_positions(m_order, std::back_inserter(pos));

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
      *it++ = std::sin((pnt.x() + dgc::const_val<T, 1> - t) * M_PI) * std::sin((pnt.y() + dgc::const_val<T, 1> - t) * M_PI);
    }
}

template<typename T, typename M> template<typename ConstItr>
void advection_2d<T, M>::numerical_fluxes(ConstItr cbegin, T t) const
{
  using point_type = dgc::point_2d<T>;
  using cell_type = typename M::cell_type;

  reference_element refElem;
  std::vector<std::pair<T, T>> pos;
  refElem.node_positions(m_order, std::back_inserter(pos));

  std::vector<int> allFaceNodes[3];
  refElem.face_nodes(m_order, 0, std::back_inserter(allFaceNodes[0]));
  refElem.face_nodes(m_order, 1, std::back_inserter(allFaceNodes[1]));
  refElem.face_nodes(m_order, 2, std::back_inserter(allFaceNodes[2]));

  int numCellNodes = refElem.num_nodes(m_order);
  int numFaceNodes = refElem.num_face_nodes(m_order);
  T aU, bU; // "a" is interior or "-" and "b" is exterior or "+"

  // one flux is computed twice from two different cells, rather than for each PAIR of half-edges
  for (int c = 0; c < m_mesh->num_cells(); ++c)
  {
    const cell_type cell = m_mesh->get_cell(c);
    for (int e = 0; e < 3; ++e)
    {
      bool isBoundary;
      int nbCell, nbLocalEdgeIdx;
      std::tie(isBoundary, nbCell, nbLocalEdgeIdx) = m_mesh->get_face_neighbor(c, e);

      std::vector<int>& faceNodes = allFaceNodes[e];
      for (int d = 0; d < numFaceNodes; ++d)
      {
        int inIdxA = c * numCellNodes + faceNodes[d]; // d.o.f. on this edge itself

        // fetch the value on interior
        aU = *(cbegin + inIdxA);

        if(isBoundary)
        {
          point_type xyPos = mapping::rs_to_xy(cell, point_type(pos[faceNodes[d]].first, pos[faceNodes[d]].second));
          bU = std::sin((xyPos.x() + dgc::const_val<T, 1> - t) * M_PI) * std::sin((xyPos.y() + dgc::const_val<T, 1> - t) * M_PI);
        }
        else
        {
          std::vector<int>& nbFaceNodes = allFaceNodes[nbLocalEdgeIdx];

          // flip direction to match neighbor's edge d.o.f.'s with this edge's !
          // THIS ONLY WORKS FOR 2D--FOR 3D WE MAY NEED TO GEOMETRICALLY MATCH D.O.F.'S !
          // TODO: spot for future generalization and abstraction
          int inIdxB = nbCell * numCellNodes + nbFaceNodes[numFaceNodes - d - 1];
          assert(inIdxA != inIdxB);

          bU = *(cbegin + inIdxB);

          // extra verification that A and B are geometrically the same point
          // can be commented out wihtout affecting calculations
          point_type A = mapping::rs_to_xy(cell, point_type(pos[faceNodes[d]].first, pos[faceNodes[d]].second));
          point_type B = mapping::rs_to_xy(m_mesh->get_cell(nbCell),
                                           point_type(pos[nbFaceNodes[numFaceNodes - d - 1]].first,
                                                      pos[nbFaceNodes[numFaceNodes - d - 1]].second));
          assert(std::abs(A.x() - B.x()) < 1.0e-10);
          assert(std::abs(A.y() - B.y()) < 1.0e-10);
        }
        
        point_type n = cell.outward_normal(e);
        T U = (n.x() + n.y()) >= dgc::const_val<T, 0> ? aU : bU;
        // numerical flux needs to be projected to the outward unit normal of the edge!
        m_numericalFluxes[c * 3 * numFaceNodes + e * numFaceNodes + d] = U * n.x() + U * n.y();
      }
    }
  }
}

template<typename T, typename M> template<typename ConstItr, typename Itr>
void advection_2d<T, M>::operator()(ConstItr in_cbegin, std::size_t size, T t, Itr out_begin) const
{
  numerical_fluxes(in_cbegin, t);

  reference_element refElem;
  int numCellNodes = refElem.num_nodes(m_order);
  int numFaceNodes = refElem.num_face_nodes(m_order);

  std::vector<T> mFl(3 * numFaceNodes); // surface-mapped numerical fluxes
  std::vector<T> vUx(numCellNodes), vUy(numCellNodes); // volume integrations
  std::vector<T> sU(numCellNodes); // surface integrations

  for (int c = 0; c < m_mesh->num_cells(); ++c)
  {
    const auto cell = m_mesh->get_cell(c);

    dense_matrix Dx = m_Dr * m_invJacobians[c * 4] + m_Ds * m_invJacobians[c * 4 + 2];
    dense_matrix Dy = m_Dr * m_invJacobians[c * 4 + 1] + m_Ds * m_invJacobians[c * 4 + 3];

    Dx.gemv(dgc::const_val<T, 1>, in_cbegin + (c * numCellNodes), dgc::const_val<T, 0>, vUx.begin());
    Dy.gemv(dgc::const_val<T, 1>, in_cbegin + (c * numCellNodes), dgc::const_val<T, 0>, vUy.begin());

    // fetch numerical fluxes and apply face mapping
    for (int e = 0; e < 3; ++e)
    {
      T faceJ = mapping::face_J(cell, e);
      for (int i = 0; i < numFaceNodes; ++i)
      {
        int ei = e * numFaceNodes + i;
        mFl[ei] = m_numericalFluxes[c * 3 * numFaceNodes + ei] * faceJ;
      }
    }

    m_L.gemv(-dgc::const_val<T, 1> / mapping::J(cell), mFl.cbegin(), dgc::const_val<T, 0>, sU.begin());

    // assemble the final results
    for (int j = 0; j < numCellNodes; ++j)
      *(out_begin + (c * numCellNodes + j)) = vUx[j] + vUy[j] + sU[j];
  }
}

#endif
