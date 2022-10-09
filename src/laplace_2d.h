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

#ifndef LAPLACE_2D_H
#define LAPLACE_2D_H 

#include <utility>
#include <algorithm>
#include <vector>

#include "simple_discretization_2d.h"

BEGIN_NAMESPACE

// discrete laplace operator in two dimensional space using interior penalty method
template<typename T, typename M> // T - number type, M - mesh type
class laplace_2d : public simple_discretization_2d<T, M>
{
public:

  laplace_2d(const M& mesh, int order); 
  ~laplace_2d(){}
  
  // all the following oprations are in-place
  template<typename RandAccItr, typename BC2D>
  void operator()(RandAccItr it, const BC2D& bc) const;

  // only queries the BC's kind, i.e., Dirichlet or Neumann
  template<typename RandAccItr, typename BC2D>
  void apply_homogeneous(RandAccItr it, const BC2D& bc) const;

  // the input is incremented by results from the inhomogeneous BC
  template<typename RandAccItr, typename BC2D>
  void apply_inhomogeneous_as_rhs(RandAccItr it, const BC2D& bc) const;

private:
  bool is_boundary_cell(typename M::index_type c) const;

  using typename simple_discretization_2d<T, M>::dense_matrix_t;
  using typename simple_discretization_2d<T, M>::reference_element;
  using typename simple_discretization_2d<T, M>::mapping;
  dense_matrix_t m_Dr;
  dense_matrix_t m_Ds;
  dense_matrix_t m_L;

  // work spaces
  mutable std::vector<T> m_qx;
  mutable std::vector<T> m_qy;
  // note: u is a scalar but du (jump of u) is a vector which
  // note: is always in the outward normal direction, thus
  // note: it is sufficient to store du as a scalar
  mutable std::vector<T> m_du;
};

template<typename T, typename M>
laplace_2d<T, M>::laplace_2d(const M& mesh, int order)
  : simple_discretization_2d<T, M>(mesh, order)
{
  // volume integration matrices
  reference_element refElem;
  dense_matrix_t v = refElem.vandermonde_matrix(this->m_order);
  dense_matrix_t vInv = v.inverse();
  dense_matrix_t mInv = v * v.transpose();

  auto vGrad = refElem.grad_vandermonde_matrix(this->m_order);
  m_Dr = vGrad.first * vInv; // this is the Dr in strong form
  m_Ds = vGrad.second * vInv; // this is the Ds in strong form

  // surface integration matrix for triangle element
  auto numCellNodes = refElem.num_nodes(this->m_order);
  auto numFaceNodes = refElem.num_face_nodes(this->m_order);
  dense_matrix_t mE(numCellNodes, 3 * numFaceNodes, const_val<T, 0>);
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
  m_L = mInv * mE; // need to work with faceJ's and cellJ^-1 later

  // work space allocation
  auto numCells = this->m_mesh->num_cells();
  m_qx.resize(numCells * numCellNodes);
  m_qy.resize(numCells * numCellNodes);
  m_du.resize(numCells * 3 * numFaceNodes);
}

template<typename T, typename M> template<typename RandAccItr, typename BC2D>
void laplace_2d<T, M>::operator()(RandAccItr it, const BC2D& bc) const
{
  apply_homogeneous(it, bc);
  for(std::size_t i = 0; i < m_qx.size(); ++i) *(it + i ) = - *(it + i);
  apply_inhomogeneous_as_rhs(it, bc);
  for(std::size_t i = 0; i < m_qx.size(); ++i) *(it + i ) = - *(it + i);
}

template<typename T, typename M> template<typename RandAccItr, typename BC2D>
void laplace_2d<T, M>::apply_homogeneous(RandAccItr it, const BC2D& bc) const
{
  using point_type = point_2d<T>;

  reference_element refElem;
  int numCellNodes = refElem.num_nodes(this->m_order);
  int numFaceNodes = refElem.num_face_nodes(this->m_order);

  std::vector<std::pair<T, T>> pos;
  refElem.node_positions(this->m_order, std::back_inserter(pos));

  std::vector<T> toLiftX(3 * numFaceNodes);
  std::vector<T> toLiftY(3 * numFaceNodes);

  // first pass to get (qx, qy)
  auto qxItr = m_qx.begin();
  auto qyItr = m_qy.begin();
  for (int c = 0; c < this->m_mesh->num_cells(); ++c)
  {
    dense_matrix_t Dx = m_Dr * this->Inv_Jacobians[c * 4] + m_Ds * this->Inv_Jacobians[c * 4 + 2];
    dense_matrix_t Dy = m_Dr * this->Inv_Jacobians[c * 4 + 1] + m_Ds * this->Inv_Jacobians[c * 4 + 3];

    int cOffset = c * numCellNodes;
    Dx.gemv(const_val<T, 1>, it + cOffset, const_val<T, 0>, qxItr + cOffset);
    Dy.gemv(const_val<T, 1>, it + cOffset, const_val<T, 0>, qyItr + cOffset);

    // numerical flux of u (p. 275)
    T du;
    const auto cell = this->m_mesh->get_cell(c);
    for (int e = 0; e < 3; ++e)
    {
      bool isBoundary;
      int nbCell, nbLocalEdgeIdx;
      std::tie(isBoundary, nbCell, nbLocalEdgeIdx) = this->m_mesh->get_face_neighbor(c, e);

      const std::vector<int>& faceNodes = e == 0 ? this->F0_Nodes : (e == 1 ? this->F1_Nodes : this->F2_Nodes);
      const std::vector<int>& nbFaceNodes =
        nbLocalEdgeIdx == 0 ? this->F0_Nodes : (nbLocalEdgeIdx == 1 ? this->F1_Nodes : this->F2_Nodes);

      int eOffset = e * numFaceNodes;
      point_type n = cell.outward_normal(e);
      T faceJ = mapping::face_J(cell, e);
      for (int i = 0; i < numFaceNodes; ++i)
      {
        if(isBoundary)
        {
          T interiorVal = *(it + c * numCellNodes + faceNodes[i]);
          du = bc.is_dirichlet() ? const_val<T, 2> * interiorVal : const_val<T, 0>;
        }
        else
          du = (*(it + c * numCellNodes + faceNodes[i]) -
                *(it + nbCell * numCellNodes + nbFaceNodes[numFaceNodes - i - 1]));
        m_du[c * 3 * numFaceNodes + eOffset + i] = du; // store du

        // note: strong form and central flux lead to using du for integration;
        // note: in week form one would use the numerical flux instead
        toLiftX[eOffset + i] = n.x() * du * faceJ / const_val<T, 2>;
        toLiftY[eOffset + i] = n.y() * du * faceJ / const_val<T, 2>;
      }
    }

    // lift
    T cellJ = mapping::J(cell);
    m_L.gemv(- const_val<T, 1> / cellJ, toLiftX.cbegin(), const_val<T, 1>, qxItr + cOffset);
    m_L.gemv(- const_val<T, 1> / cellJ, toLiftY.cbegin(), const_val<T, 1>, qyItr + cOffset);
  }

  // second pass to get the laplacian
  for (int c = 0; c < this->m_mesh->num_cells(); ++c)
  {
    dense_matrix_t Dx = m_Dr * this->Inv_Jacobians[c * 4] + m_Ds * this->Inv_Jacobians[c * 4 + 2];
    dense_matrix_t Dy = m_Dr * this->Inv_Jacobians[c * 4 + 1] + m_Ds * this->Inv_Jacobians[c * 4 + 3];

    int cOffset = c * numCellNodes;
    Dx.gemv(const_val<T, 1>, qxItr + cOffset, const_val<T, 0>, it + cOffset);
    Dy.gemv(const_val<T, 1>, qyItr + cOffset, const_val<T, 1>, it + cOffset);

    // numerical flux of (qx, qy) (p. 275, using tau = 1)
    // note: eqns. 59 and 60 in the 2017 paper give determination of tau per element
    T dqN;
    const auto cell = this->m_mesh->get_cell(c);
    for (int e = 0; e < 3; ++e)
    {
      bool isBoundary;
      int nbCell, nbLocalEdgeIdx;
      std::tie(isBoundary, nbCell, nbLocalEdgeIdx) = this->m_mesh->get_face_neighbor(c, e);

      const std::vector<int>& faceNodes = e == 0 ? this->F0_Nodes : (e == 1 ? this->F1_Nodes : this->F2_Nodes);
      const std::vector<int>& nbFaceNodes =
        nbLocalEdgeIdx == 0 ? this->F0_Nodes : (nbLocalEdgeIdx == 1 ? this->F1_Nodes : this->F2_Nodes);

      int eOffset = e * numFaceNodes;
      point_type n = cell.outward_normal(e);
      T faceJ = mapping::face_J(cell, e);
      for (int i = 0; i < numFaceNodes; ++i)
      {
        if(isBoundary)
        {
          T interiorGradN = *(qxItr + c * numCellNodes + faceNodes[i]) * n.x() +
                            *(qyItr + c * numCellNodes + faceNodes[i]) * n.y();
          dqN = bc.is_dirichlet() ? const_val<T, 0> : const_val<T, 2> * interiorGradN;
        }
        else
        {
          dqN = (*(qxItr + c * numCellNodes + faceNodes[i]) -
                 *(qxItr + nbCell * numCellNodes + nbFaceNodes[numFaceNodes - i - 1])) * n.x() +
                (*(qyItr + c * numCellNodes + faceNodes[i]) -
                 *(qyItr + nbCell * numCellNodes + nbFaceNodes[numFaceNodes - i - 1])) * n.y();
        }

        // similarly, the integration below is due to strong form and the flux used for (qx, qy)
        toLiftX[eOffset + i] = (dqN + const_val<T, 2> * m_du[c * 3 * numFaceNodes + eOffset + i]) * faceJ / const_val<T, 2>;
      }
    }

    // lift
    m_L.gemv(- const_val<T, 1> / mapping::J(cell), toLiftX.cbegin(), const_val<T, 1>, it + cOffset);
  }
}

template<typename T, typename M> template<typename RandAccItr, typename BC2D>
void laplace_2d<T, M>::apply_inhomogeneous_as_rhs(RandAccItr it, const BC2D& bc) const
{
  using point_type = point_2d<T>;

  reference_element refElem;
  int numCellNodes = refElem.num_nodes(this->m_order);
  int numFaceNodes = refElem.num_face_nodes(this->m_order);

  std::vector<std::pair<T, T>> pos;
  refElem.node_positions(this->m_order, std::back_inserter(pos));

  std::vector<T> toLiftX(3 * numFaceNodes);
  std::vector<T> toLiftY(3 * numFaceNodes);

  // first pass to get (qx, qy)
  auto qxItr = m_qx.begin();
  auto qyItr = m_qy.begin();
  std::fill(m_du.begin(), m_du.end(), const_val<T, 0>);
  for (int c = 0; c < this->m_mesh->num_cells(); ++c)
    if(is_boundary_cell(c))
    {
      std::fill(toLiftX.begin(), toLiftX.end(), const_val<T, 0>);
      std::fill(toLiftY.begin(), toLiftY.end(), const_val<T, 0>);
      const auto cell = this->m_mesh->get_cell(c);

      T gu;
      int cOffset = c * numCellNodes;
      for (int e = 0; e < 3; ++e)
      {
        bool isBoundary;
        int nbCell, nbLocalEdgeIdx;
        std::tie(isBoundary, nbCell, nbLocalEdgeIdx) = this->m_mesh->get_face_neighbor(c, e);
        if(isBoundary)
        {
          const std::vector<int>& faceNodes = e == 0 ? this->F0_Nodes : (e == 1 ? this->F1_Nodes : this->F2_Nodes);
          const std::vector<int>& nbFaceNodes =
            nbLocalEdgeIdx == 0 ? this->F0_Nodes : (nbLocalEdgeIdx == 1 ? this->F1_Nodes : this->F2_Nodes);

          int eOffset = e * numFaceNodes;
          point_type n = cell.outward_normal(e);
          T faceJ = mapping::face_J(cell, e);
          for (int i = 0; i < numFaceNodes; ++i)
          {
            point_type xyPos = mapping::rs_to_xy(cell, point_type(pos[faceNodes[i]].first, pos[faceNodes[i]].second));
            gu = bc.is_dirichlet() ? bc.exterior_val(xyPos.x(), xyPos.y(), const_val<T, 0>) : const_val<T, 0>;
            m_du[c * 3 * numFaceNodes + eOffset + i] = gu; // store for the second pass to use

            toLiftX[eOffset + i] = n.x() * gu * faceJ / const_val<T, 2>;
            toLiftY[eOffset + i] = n.y() * gu * faceJ / const_val<T, 2>;
          }
        }
      }

      T cellJ = mapping::J(cell);
      m_L.gemv(- const_val<T, 1> / cellJ, toLiftX.cbegin(), const_val<T, 0>, qxItr + cOffset);
      m_L.gemv(- const_val<T, 1> / cellJ, toLiftY.cbegin(), const_val<T, 0>, qyItr + cOffset);
    }

  // second pass to get the laplacian
  for (int c = 0; c < this->m_mesh->num_cells(); ++c)
    if(is_boundary_cell(c))
    {
      dense_matrix_t Dx = m_Dr * this->Inv_Jacobians[c * 4] + m_Ds * this->Inv_Jacobians[c * 4 + 2];
      dense_matrix_t Dy = m_Dr * this->Inv_Jacobians[c * 4 + 1] + m_Ds * this->Inv_Jacobians[c * 4 + 3];

      int cOffset = c * numCellNodes;
      Dx.gemv(const_val<T, 1>, qxItr + cOffset, const_val<T, 1>, it + cOffset);
      Dy.gemv(const_val<T, 1>, qyItr + cOffset, const_val<T, 1>, it + cOffset);

      std::fill(toLiftX.begin(), toLiftX.end(), const_val<T, 0>);
      const auto cell = this->m_mesh->get_cell(c);

      T hq;
      for (int e = 0; e < 3; ++e)
      {
        bool isBoundary;
        int nbCell, nbLocalEdgeIdx;
        std::tie(isBoundary, nbCell, nbLocalEdgeIdx) = this->m_mesh->get_face_neighbor(c, e);
        if (isBoundary)
        {
          const std::vector<int>& faceNodes = e == 0 ? this->F0_Nodes : (e == 1 ? this->F1_Nodes : this->F2_Nodes);
          const std::vector<int>& nbFaceNodes =
            nbLocalEdgeIdx == 0 ? this->F0_Nodes : (nbLocalEdgeIdx == 1 ? this->F1_Nodes : this->F2_Nodes);

          int eOffset = e * numFaceNodes;
          point_type n = cell.outward_normal(e);
          T faceJ = mapping::face_J(cell, e);
          for (int i = 0; i < numFaceNodes; ++i)
          {
            point_type xyPos = mapping::rs_to_xy(cell, point_type(pos[faceNodes[i]].first, pos[faceNodes[i]].second));
            hq = bc.is_dirichlet() ? const_val<T, 0> : bc.exterior_grad_n(xyPos.x(), xyPos.y(), const_val<T, 0>);

            toLiftX[eOffset + i] = (hq + const_val<T, 2> * m_du[c * 3 * numFaceNodes + eOffset + i]) * faceJ / const_val<T, 2>;
          }
        }
      }

      m_L.gemv(- const_val<T, 1> / mapping::J(cell), toLiftX.cbegin(), const_val<T, 1>, it + cOffset);
    }
}

template<typename T, typename M>
bool laplace_2d<T, M>::is_boundary_cell(typename M::index_type c) const
{
  for (int e = 0; e < 3; ++e)
  {
    bool isBoundary;
    int nbCell, nbLocalEdgeIdx;
    std::tie(isBoundary, nbCell, nbLocalEdgeIdx) = this->m_mesh->get_face_neighbor(c, e);
    if (isBoundary) return true;
  }
  return false;
}

END_NAMESPACE

#endif
