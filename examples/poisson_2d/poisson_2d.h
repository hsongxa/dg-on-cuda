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

#ifndef POISSON_2D_H
#define POISSON_2D_H 

#include <utility>
#include <vector>
#include <math.h>

#include "simple_discretization_2d.h"

// host code that represent the problem of Poisson's equation in two dimensional space
// using triangle mesh
template<typename T, typename M> // T - number type, M - mesh type
class poisson_2d : public dgc::simple_discretization_2d<T, M>
{
public:
  using index_type = typename dgc::simple_discretization_2d<T, M>::index_type;

  poisson_2d(const M& mesh, int order); 
  ~poisson_2d(){}
  
  // note that the layout of DOFs in memory are different for CPU execution and GPU execution
  template<typename OutputItr>
  void exact_solution(OutputItr it) const;

  template<typename OutputItr>
  void rhs(OutputItr it) const;

  // CPU execution 
  template<typename RandAccItr>
  void operator()(RandAccItr it) const;

  // need to easily copy these data to device so make them public
  using typename dgc::simple_discretization_2d<T, M>::dense_matrix_t;
  dense_matrix_t m_Dr;
  dense_matrix_t m_Ds;
  dense_matrix_t m_L;

private:
  using typename dgc::simple_discretization_2d<T, M>::reference_element;
  using typename dgc::simple_discretization_2d<T, M>::mapping;

  // work spaces
  mutable std::vector<T> m_qx;
  mutable std::vector<T> m_qy;
  mutable std::vector<T> m_du;
};

template<typename T, typename M>
poisson_2d<T, M>::poisson_2d(const M& mesh, int order)
  : dgc::simple_discretization_2d<T, M>(mesh, order)
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
  dense_matrix_t mE(numCellNodes, 3 * numFaceNodes, dgc::const_val<T, 0>);
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

template<typename T, typename M> template<typename OutputItr>
void poisson_2d<T, M>::exact_solution(OutputItr it) const
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
      *it++ = std::sin(pnt.x() * M_PI) * std::sin(pnt.y() * M_PI);
    }
}

template<typename T, typename M> template<typename OutputItr>
void poisson_2d<T, M>::rhs(OutputItr it) const
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
      *it++ = - dgc::const_val<T, 2> * M_PI * M_PI * std::sin(pnt.x() * M_PI) * std::sin(pnt.y() * M_PI);
    }
}

// this is the discrete Laplace operator
template<typename T, typename M> template<typename RandAccItr>
void poisson_2d<T, M>::operator()(RandAccItr it) const
{
  using point_type = dgc::point_2d<T>;

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
    Dx.gemv(dgc::const_val<T, 1>, it + cOffset, dgc::const_val<T, 0>, qxItr + cOffset);
    Dy.gemv(dgc::const_val<T, 1>, it + cOffset, dgc::const_val<T, 0>, qyItr + cOffset);

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
          du = *(it + c * numCellNodes + faceNodes[i]); // u = 0 at boundary
        else
          du = (*(it + c * numCellNodes + faceNodes[i]) -
                *(it + nbCell * numCellNodes + nbFaceNodes[numFaceNodes - i - 1]));
        m_du[c * 3 * numFaceNodes + eOffset + i] = du; // store du

        toLiftX[eOffset + i] = n.x() * du * faceJ / dgc::const_val<T, 2>;
        toLiftY[eOffset + i] = n.y() * du * faceJ / dgc::const_val<T, 2>;
      }
    }

    // lift
    T cellJ = mapping::J(cell);
    m_L.gemv(- dgc::const_val<T, 1> / cellJ, toLiftX.cbegin(), dgc::const_val<T, 1>, qxItr + cOffset);
    m_L.gemv(- dgc::const_val<T, 1> / cellJ, toLiftY.cbegin(), dgc::const_val<T, 1>, qyItr + cOffset);
  }

  // second pass to get residuals
  for (int c = 0; c < this->m_mesh->num_cells(); ++c)
  {
    dense_matrix_t Dx = m_Dr * this->Inv_Jacobians[c * 4] + m_Ds * this->Inv_Jacobians[c * 4 + 2];
    dense_matrix_t Dy = m_Dr * this->Inv_Jacobians[c * 4 + 1] + m_Ds * this->Inv_Jacobians[c * 4 + 3];

    int cOffset = c * numCellNodes;
    Dx.gemv(dgc::const_val<T, 1>, qxItr + cOffset, dgc::const_val<T, 0>, it + cOffset);
    Dy.gemv(dgc::const_val<T, 1>, qyItr + cOffset, dgc::const_val<T, 1>, it + cOffset);

    // numerical flux of (qx, qy) (p. 275, using tau = 1)
    T dqx, dqy;
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
          point_type xyPos = mapping::rs_to_xy(cell, point_type(pos[faceNodes[i]].first, pos[faceNodes[i]].second));
          dqx = *(qxItr + c * numCellNodes + faceNodes[i]) -
                M_PI * std::cos(xyPos.x() * M_PI) * std::sin(xyPos.y() * M_PI); // boundary condition of grad(u)
          dqy = *(qyItr + c * numCellNodes + faceNodes[i]) -
                M_PI * std::sin(xyPos.x() * M_PI) * std::cos(xyPos.y() * M_PI); // boundary condition of grad(u)
        }
        else
        {
          dqx = (*(qxItr + c * numCellNodes + faceNodes[i]) -
                 *(qxItr + nbCell * numCellNodes + nbFaceNodes[numFaceNodes - i - 1]));
          dqy = (*(qyItr + c * numCellNodes + faceNodes[i]) -
                 *(qyItr + nbCell * numCellNodes + nbFaceNodes[numFaceNodes - i - 1]));
        }

        toLiftX[eOffset + i] = (n.x() * dqx + n.y() * dqy + dgc::const_val<T, 2> * m_du[c * 3 * numFaceNodes + eOffset + i]) *
                               faceJ / dgc::const_val<T, 2>;
      }
    }

    // lift
    m_L.gemv(- dgc::const_val<T, 1> / mapping::J(cell), toLiftX.cbegin(), dgc::const_val<T, 1>, it + cOffset);
  }
}

#endif
