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
#include <math.h>

#include "simple_discretization_2d.h"

// host code that represent the problem of linear advection equation in two dimensional space
// using triangle mesh
template<typename T, typename M> // T - number type, M - mesh type
class advection_2d : public dgc::simple_discretization_2d<T, M>
{
public:
  advection_2d(const M& mesh, int order); 
  ~advection_2d(){}
  
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
  using typename dgc::simple_discretization_2d<T, M>::dense_matrix_t;
  dense_matrix_t m_Dr;
  dense_matrix_t m_Ds;
  dense_matrix_t m_L;

private:
  using typename dgc::simple_discretization_2d<T, M>::reference_element;
  using typename dgc::simple_discretization_2d<T, M>::mapping;

  template<typename ConstItr>
  void numerical_fluxes(ConstItr cbegin, T t) const; // time t is used for boundary conditions

  // work space for numerical fluxes to avoid repeated allocations
  mutable std::vector<T> m_numericalFluxes;
};

template<typename T, typename M>
advection_2d<T, M>::advection_2d(const M& mesh, int order)
  : dgc::simple_discretization_2d<T, M>(mesh, order)
{
  // volume integration matrices
  reference_element refElem;
  dense_matrix_t v = refElem.vandermonde_matrix(this->m_order);
  dense_matrix_t mInv = v * v.transpose();
  dense_matrix_t m = mInv.inverse();

  auto vGrad = refElem.grad_vandermonde_matrix(this->m_order);
  m_Dr = v * vGrad.first.transpose() * m; // this is the Dr in weak form
  m_Ds = v * vGrad.second.transpose() * m; // this is the Ds in weak form

  // surface integration matrix for triangle element
  auto numFaceNodes = refElem.num_face_nodes(this->m_order);
  dense_matrix_t mE(refElem.num_nodes(this->m_order), 3 * numFaceNodes, dgc::const_val<T, 0>);
  for (int e = 0; e < 3; ++e)
  {
    const std::vector<int>& eN = e == 0 ? this->F0_Nodes : (e == 1 ? this->F1_Nodes : this->F2_Nodes);

    dense_matrix_t eV = refElem.face_vandermonde_matrix(this->m_order, e);
    dense_matrix_t eM = eV * eV.transpose();
    eM = eM.inverse();
    for (std::size_t row = 0; row < eM.size_row(); ++row)
      for (std::size_t col = 0; col < eM.size_col(); ++col)
        mE(eN[row], e * numFaceNodes + col) = eM(row, col);
  }
  m_L = mInv * mE;

  // space allocation for numerical fluxes
  m_numericalFluxes.resize(this->m_mesh->num_cells() * 3 * numFaceNodes);
}

template<typename T, typename M> template<typename OutputIterator1, typename OutputIterator2>
void advection_2d<T, M>::initialize_dofs(OutputIterator1 it1, OutputIterator2 it2) const
{
  using point_type = dgc::point_2d<T>;
  using cell_type = typename M::cell_type;

  reference_element refElem;
  std::vector<std::pair<T, T>> pos;
  refElem.node_positions(this->m_order, std::back_inserter(pos));

#if defined USE_CPU_ONLY
  for (int i = 0; i < this->m_mesh->num_cells(); ++i)
    for (std::size_t j = 0; j < pos.size(); ++j)
#else
  for (std::size_t j = 0; j < pos.size(); ++j)
    for (int i = 0; i < this->m_mesh->num_cells(); ++i)
#endif
    {
      const cell_type cell = this->m_mesh->get_cell(i);
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
  refElem.node_positions(this->m_order, std::back_inserter(pos));

#if defined USE_CPU_ONLY
  for (int i = 0; i < this->m_mesh->num_cells(); ++i)
    for (std::size_t j = 0; j < pos.size(); ++j)
#else
  for (std::size_t j = 0; j < pos.size(); ++j)
    for (int i = 0; i < this->m_mesh->num_cells(); ++i)
#endif
    {
      const cell_type cell = this->m_mesh->get_cell(i);
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
  refElem.node_positions(this->m_order, std::back_inserter(pos));

  int numCellNodes = refElem.num_nodes(this->m_order);
  int numFaceNodes = refElem.num_face_nodes(this->m_order);
  T aU, bU; // "a" is interior or "-" and "b" is exterior or "+"

  // one flux is computed twice from two different cells, rather than for each PAIR of half-edges
  for (int c = 0; c < this->m_mesh->num_cells(); ++c)
  {
    const cell_type cell = this->m_mesh->get_cell(c);
    for (int e = 0; e < 3; ++e)
    {
      bool isBoundary;
      int nbCell, nbLocalEdgeIdx;
      std::tie(isBoundary, nbCell, nbLocalEdgeIdx) = this->m_mesh->get_face_neighbor(c, e);

      const std::vector<int>& faceNodes = e == 0 ? this->F0_Nodes : (e == 1 ? this->F1_Nodes : this->F2_Nodes);
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
          const std::vector<int>& nbFaceNodes = nbLocalEdgeIdx == 0 ? this->F0_Nodes : (nbLocalEdgeIdx == 1 ? this->F1_Nodes : this->F2_Nodes);

          // flip direction to match neighbor's edge d.o.f.'s with this edge's !
          // THIS ONLY WORKS FOR 2D--FOR 3D WE MAY NEED TO GEOMETRICALLY MATCH D.O.F.'S !
          // TODO: spot for future generalization and abstraction
          int inIdxB = nbCell * numCellNodes + nbFaceNodes[numFaceNodes - d - 1];
          assert(inIdxA != inIdxB);

          bU = *(cbegin + inIdxB);

          // extra verification that A and B are geometrically the same point
          // can be commented out wihtout affecting calculations
          point_type A = mapping::rs_to_xy(cell, point_type(pos[faceNodes[d]].first, pos[faceNodes[d]].second));
          point_type B = mapping::rs_to_xy(this->m_mesh->get_cell(nbCell),
                                           point_type(pos[nbFaceNodes[numFaceNodes - d - 1]].first,
                                                      pos[nbFaceNodes[numFaceNodes - d - 1]].second));
          assert(std::abs(A.x() - B.x()) < 1.0e-10);
          assert(std::abs(A.y() - B.y()) < 1.0e-10);
        }
        
        point_type n = cell.outward_normal(e);
        T U = (n.x() + n.y()) >= dgc::const_val<T, 0> ? aU : bU;
        // numerical flux needs to be projected to the outward unit normal of the edge!
        m_numericalFluxes[c * 3 * numFaceNodes + e * numFaceNodes + d] = U * (n.x() + n.y());
      }
    }
  }
}

template<typename T, typename M> template<typename ConstItr, typename Itr>
void advection_2d<T, M>::operator()(ConstItr in_cbegin, std::size_t size, T t, Itr out_begin) const
{
  numerical_fluxes(in_cbegin, t);

  reference_element refElem;
  int numCellNodes = refElem.num_nodes(this->m_order);
  int numFaceNodes = refElem.num_face_nodes(this->m_order);

  std::vector<T> mFl(3 * numFaceNodes); // surface-mapped numerical fluxes

  for (int c = 0; c < this->m_mesh->num_cells(); ++c)
  {
    dense_matrix_t Dx = m_Dr * this->Inv_Jacobians[c * 4] + m_Ds * this->Inv_Jacobians[c * 4 + 2];
    dense_matrix_t Dy = m_Dr * this->Inv_Jacobians[c * 4 + 1] + m_Ds * this->Inv_Jacobians[c * 4 + 3];

    int offset = c * numCellNodes;
    Dx.gemv(dgc::const_val<T, 1>, in_cbegin + offset, dgc::const_val<T, 0>, out_begin + offset);
    Dy.gemv(dgc::const_val<T, 1>, in_cbegin + offset, dgc::const_val<T, 1>, out_begin + offset);

    // fetch numerical fluxes and apply face mapping
    const auto cell = this->m_mesh->get_cell(c);
    for (int e = 0; e < 3; ++e)
    {
      T faceJ = mapping::face_J(cell, e);
      for (int i = 0; i < numFaceNodes; ++i)
      {
        int ei = e * numFaceNodes + i;
        mFl[ei] = m_numericalFluxes[c * 3 * numFaceNodes + ei] * faceJ;
      }
    }

    m_L.gemv(- dgc::const_val<T, 1> / mapping::J(cell), mFl.cbegin(), dgc::const_val<T, 1>, out_begin + offset);
  }
}

#endif
