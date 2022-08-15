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

#ifndef GRAD_2D_H
#define GRAD_2D_H 

#include <algorithm>
#include <utility>

#include "simple_discretization_2d.h"

BEGIN_NAMESPACE

// discrete gradient operator with central flux in two dimensional space
// using simple discretization
template<typename T, typename M> // T - number type, M - mesh type
class grad_2d : public simple_discretization_2d<T, M>
{
public:
  grad_2d(const M& mesh, int order); 
  ~grad_2d(){}
  
  // operate in place - the result is stored in in0
  template<typename RandItr, typename BC2D>
  void operator()(RandItr in0, RandItr in1, std::size_t size, BC2D bc) const;

private:
  using typename simple_discretization_2d<T, M>::reference_element;
  using typename simple_discretization_2d<T, M>::mapping;
  using typename simple_discretization_2d<T, M>::dense_matrix_t;
  dense_matrix_t m_Dr;
  dense_matrix_t m_Ds;
  dense_matrix_t m_L;

  template<typename RandItr, typename BC2D>
  void numerical_fluxes(RandItr in, BC2D bc) const;
  mutable std::vector<T> m_numericalFluxes;
};

template<typename T, typename M>
grad_2d<T, M>::grad_2d(const M& mesh, int order) : simple_discretization_2d<T, M>(mesh, order)
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
  dense_matrix_t mE(refElem.num_nodes(this->m_order), 3 * numFaceNodes, const_val<T, 0>);
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

template<typename T, typename M> template<typename RandItr, typename BC2D>
void grad_2d<T, M>::numerical_fluxes(RandItr in, BC2D bc) const
{
  using point_type = point_2d<T>;
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
        aU = *(in + inIdxA);

        if(isBoundary)
        {
          point_type xyPos = mapping::rs_to_xy(cell, point_type(pos[faceNodes[d]].first, pos[faceNodes[d]].second));
          bU = bc.exterior_val(xyPos.x(), xyPos.y(), aU);
        }
        else
        {
          const std::vector<int>& nbFaceNodes = nbLocalEdgeIdx == 0 ? this->F0_Nodes : (nbLocalEdgeIdx == 1 ? this->F1_Nodes : this->F2_Nodes);

          // flip direction to match neighbor's edge d.o.f.'s with this edge's !
          // THIS ONLY WORKS FOR 2D--FOR 3D WE MAY NEED TO GEOMETRICALLY MATCH D.O.F.'S !
          int inIdxB = nbCell * numCellNodes + nbFaceNodes[numFaceNodes - d - 1];
          assert(inIdxA != inIdxB);
          bU = *(in + inIdxB);
        }
        
        // central flux to be projected to the outward normal later
        m_numericalFluxes[c * 3 * numFaceNodes + e * numFaceNodes + d] = (aU + bU) / const_val<T, 2>;
      }
    }
  }
}

template<typename T, typename M> template<typename RandItr, typename BC2D>
void grad_2d<T, M>::operator()(RandItr in0, RandItr in1, std::size_t size, BC2D bc) const
{
  numerical_fluxes(in0, bc);

  reference_element refElem;
  int numCellNodes = refElem.num_nodes(this->m_order);
  int numFaceNodes = refElem.num_face_nodes(this->m_order);

  std::vector<T> mFlX(3 * numFaceNodes); // surface-mapped numerical flux in X
  std::vector<T> mFlY(3 * numFaceNodes); // surface-mapped numerical flux in Y
  std::vector<T> cpy(numCellNodes); // cache of in0 for one cell at a time

  for (int c = 0; c < this->m_mesh->num_cells(); ++c)
  {
    dense_matrix_t Dx = m_Dr * this->Inv_Jacobians[c * 4] + m_Ds * this->Inv_Jacobians[c * 4 + 2];
    dense_matrix_t Dy = m_Dr * this->Inv_Jacobians[c * 4 + 1] + m_Ds * this->Inv_Jacobians[c * 4 + 3];

    int offset = c * numCellNodes;
    std::copy(in0 + offset, in0 + offset + numCellNodes, cpy.begin());
    Dx.gemv(- const_val<T, 1>, cpy.cbegin(), const_val<T, 0>, in0 + offset);
    Dy.gemv(- const_val<T, 1>, cpy.cbegin(), const_val<T, 0>, in1 + offset);

    // fetch numerical fluxes and apply face mapping as well as normal direction
    const auto cell = this->m_mesh->get_cell(c);
    for (int e = 0; e < 3; ++e)
    {
      T faceJ = mapping::face_J(cell, e);
      point_type n = cell.outward_normal(e);
      for (int i = 0; i < numFaceNodes; ++i)
      {
        int ei = e * numFaceNodes + i;
        mFlX[ei] = m_numericalFluxes[c * 3 * numFaceNodes + ei] * n.x() * faceJ;
        mFlY[ei] = m_numericalFluxes[c * 3 * numFaceNodes + ei] * n.y() * faceJ;
      }
    }

    m_L.gemv(const_val<T, 1> / mapping::J(cell), mFlX.cbegin(), const_val<T, 1>, in0 + offset);
    m_L.gemv(const_val<T, 1> / mapping::J(cell), mFlY.cbegin(), const_val<T, 1>, in1 + offset);
  }
}

END_NAMESPACE

#endif
