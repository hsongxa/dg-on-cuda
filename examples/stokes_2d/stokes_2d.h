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

#ifndef STOKES_2D_H
#define STOKES_2D_H 

#include <utility>
#include <vector>
#include <cmath>

#include <thrust/tuple.h>

#include "laplace_2d.h"
#include "simple_discretization_2d.h"

// host code that represent the problem of Stokes equation in two dimensional space
// using triangle mesh
template<typename T, typename M> // T - number type, M - mesh type
class stokes_2d : public dgc::simple_discretization_2d<T, M>
{
public:
  stokes_2d(const M& mesh, int order); 
  ~stokes_2d(){}
  
  // the layout of DOFs in memory are different for CPU execution and GPU execution;
  // the first iterator sets the velocity DOF positions and the second iterator sets
  // the initial values of the velocity DOFs
  template<typename PosItr, typename DofItr>
  void initialize_dofs(PosItr posItr, DofItr dofItr) const;

  // the layout of DOFs in memory are different for CPU execution and GPU execution
  template<typename ZipItr>
  void exact_solution(T t, ZipItr it) const;

  // CPU execution - second order in time
  template<typename ConstZipItr, typename ZipItr>
  void advance_timestep(ConstZipItr in0, ConstZipItr in1, std::size_t size, T t, T dt, ZipItr out) const;

private:
  using typename dgc::simple_discretization_2d<T, M>::reference_element;
  using typename dgc::simple_discretization_2d<T, M>::mapping;

  // (m_u_tilde_x, m_u_tilde_y) => m_p
  void solve_pressure() const;

  dgc::laplace_2d<T, M> m_laplace_2d;

  // work space
  mutable std::vector<T> m_u_tilde_x;
  mutable std::vector<T> m_u_tilde_y;
  mutable std::vector<T> m_div_u;
  mutable std::vector<T> m_p;

  static constexpr T a = 2.883356L;
  static constexpr T lamda = dgc::const_val<T, 1> + a * a;

private:
  struct pressure_bc
  {
    // Table 1 of the 2017 paper by N. Fehn et al.
    T exterior_val(T x, T y, T interior_val) { return - interior_val; }
    T exterior_grad_n(T x, T y, T interior_grad_n) { return interior_grad_n; }
  };
};

template<typename T, typename M>
stokes_2d<T, M>::stokes_2d(const M& mesh, int order)
  : dgc::simple_discretization_2d<T, M>(mesh, order), m_laplace_2d(mesh, order)
{
  // space allocation for intermediate variables
  auto numCells = this->m_mesh->num_cells();
  auto numCellNodes = reference_element().num_nodes(this->m_order);
  m_u_tilde_x.resize(numCells * numCellNodes);
  m_u_tilde_y.resize(numCells * numCellNodes);
  m_div_u.resize(numCells * numCellNodes);
  m_p.resize(numCells * numCellNodes);
}

template<typename T, typename M> template<typename PosItr, typename DofItr>
void stokes_2d<T, M>::initialize_dofs(PosItr posItr, DofItr dofItr) const
{
  using point_type = dgc::point_2d<T>;
  using cell_type = typename M::cell_type;

  std::vector<std::pair<T, T>> pos;
  reference_element().node_positions(this->m_order, std::back_inserter(pos));

  auto itTuple = dofItr.get_iterator_tuple();
  auto itUx = thrust::get<0>(itTuple);
  auto itUy = thrust::get<1>(itTuple);

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
      *posItr++ = pnt;
      *itUx++ = std::sin(pnt.x()) * (a * std::sin(a * pnt.y()) - std::cos(a) * sinh(pnt.y()));
      *itUy++ = std::cos(pnt.x()) * (std::cos(a * pnt.y()) + std::cos(a) * std::cosh(pnt.y()));
    }
}

template<typename T, typename M> template<typename ZipItr>
void stokes_2d<T, M>::exact_solution(T t, ZipItr it) const
{
  using point_type = dgc::point_2d<T>;
  using cell_type = typename M::cell_type;

  std::vector<std::pair<T, T>> pos;
  reference_element().node_positions(this->m_order, std::back_inserter(pos));

  auto itTuple = it.get_iterator_tuple();
  auto itUx = thrust::get<0>(itTuple);
  auto itUy = thrust::get<1>(itTuple);
  auto itP = thrust::get<2>(itTuple);

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
      *itUx++ = std::sin(pnt.x()) * (a * std::sin(a * pnt.y()) - std::cos(a) * std::sinh(pnt.y())) * std::exp(- lamda * t);
      *itUy++ = std::cos(pnt.x()) * (std::cos(a * pnt.y()) + std::cos(a) * std::cosh(pnt.y())) * std::exp(- lamda * t) ;
      *itP++ = lamda * std::cos(a) * std::cos(pnt.x()) * std::sinh(pnt.y()) * std::exp(- lamda * t);
    }
}

template<typename T, typename M>
void stokes_2d<T, M>::solve_pressure() const
{
  m_laplace_2d(m_div_u, pressure_bc());
}

// t => t + dt
template<typename T, typename M> template<typename ConstZipItr, typename ZipItr>
void stokes_2d<T, M>::advance_timestep(ConstZipItr in0, ConstZipItr in1, std::size_t size, T t, T dt, ZipItr out) const
{
  // stage 1: get u_tilde - no advection term in stokes equation
  const auto in0_Tuple = in0.get_iterator_tuple();
  const auto in0_Ux = thrust::get<0>(in0_Tuple);
  const auto in0_Uy = thrust::get<1>(in0_Tuple);
  const auto in1_Tuple = in1.get_iterator_tuple();
  const auto in1_Ux = thrust::get<0>(in1_Tuple);
  const auto in1_Uy = thrust::get<1>(in1_Tuple);
  for (std::size_t i = 0; i < size; ++i)
  {
    m_u_tilde_x[i] = dt * (*(in1_Ux + i) * dgc::const_val<T, 4> - *(in0_Ux + i)) / dgc::const_val<T, 3>;
    m_u_tilde_y[i] = dt * (*(in1_Uy + i) * dgc::const_val<T, 4> - *(in0_Uy + i)) / dgc::const_val<T, 3>;
  }

  // stage 2: pressure step
  solve_pressure();
}

#endif
