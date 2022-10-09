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

#include "laplace_2d.h"
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

  template<typename OutputItr>
  void dof_positions(OutputItr it_x, OutputItr it_y) const;
  
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

  dgc::laplace_2d<T, M> m_laplace_op;

private:
  struct dirichlet_bc
  {
    bool is_dirichlet() const { return true; }
    // see pp. 248-249, also Table 1 of the 2017 paper by N. Fehn et al.
    T exterior_val(T x, T y, T interior_val) const { return - interior_val; }
    T exterior_grad_n(T x, T y, T interior_grad_n) const { return interior_grad_n; }
  };
};

template<typename T, typename M>
poisson_2d<T, M>::poisson_2d(const M& mesh, int order)
  : dgc::simple_discretization_2d<T, M>(mesh, order), m_laplace_op(mesh, order)
{
  // note: the process of constructing the matrices below has already been done
  // note: when constructing the laplace operator above, but we repeat here
  // note: for the sake of exposing them to device code

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
}

template<typename T, typename M> template<typename OutputItr>
void poisson_2d<T, M>::dof_positions(OutputItr it_x, OutputItr it_y) const
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
      *it_x++ = pnt.x();
      *it_y++ = pnt.y();
    }
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
  // NOTE: The reason that we can directly call this version of the laplace operator,
  // NOTE: rather than the version that separates the homogeneous and inhomogeneous
  // NOTE: parts, is that this particular problem uses homogeneous Dirichlet BC.
  m_laplace_op(it, dirichlet_bc());
}

#endif
