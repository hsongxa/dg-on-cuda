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
#include <tuple>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>

#include <thrust/tuple.h>

#include <cusparse.h>
#include <cusolverSp.h>

#include "div_2d.h"
#include "grad_2d.h"
#include "laplace_2d.h"
#include "simple_discretization_2d.h"

// host code that represent the problem of Stokes equation in two dimensional space
// using triangle mesh
template<typename M> // mesh type
class stokes_2d : public dgc::simple_discretization_2d<double, M>
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
  void exact_solution(double t, ZipItr it) const;

  // CPU execution - second order in time
  template<typename ConstZipItr, typename ZipItr>
  void advance_timestep(ConstZipItr in0, double t_prev, ConstZipItr in1, double t, std::size_t size, double dt, ZipItr out);

private:
  using typename dgc::simple_discretization_2d<double, M>::dense_matrix_t;
  using typename dgc::simple_discretization_2d<double, M>::reference_element;
  using typename dgc::simple_discretization_2d<double, M>::mapping;

  // first (m_u_tilde_x, m_u_tilde_y) => m_p
  void pressure_step(double t, double t_prev, double dt,
                     const double* u_t_x, const double* u_t_y, const double* u_t_prev_x, const double* u_t_prev_y);

  // first (m_u_tilde_x, m_u_tilde_y), m_p => second (m_u_tilde_x, m_u_tilde_y)
  void projection_step(double dt);
  
  // second (m_u_tilde_x, m_u_tilde_y) => final (u_x, u_y) stored in (m_u_tilde_x, m_u_tilde_y)
  void viscous_step(double t, double dt);

  // linear system building and solving - use CUDA linear solvers which need the matrix A explicitly
  // TODO: implement an implicit version of the discrete laplace operator which returns the matrix
  // TODO: and the rhs of the inhomogeneous BC to replace the two functions below
  template<typename BC>
  std::vector<std::tuple<int, int, double>> build_poisson_sys_matrix(std::vector<double>& temp, const BC& bc) const;
  template<typename BC>
  std::vector<std::tuple<int, int, double>> build_helmholtz_sys_matrix(double dt, std::vector<double>& temp, const BC& bc) const;
  // TODO: this function can be moved out of the class as it is independent to the template parameter M
  static void solve_sys(std::vector<std::tuple<int, int, double>>& matrix, const std::vector<double>& b, std::vector<double>& x);

  dgc::div_2d<double, M> m_div_2d;
  dgc::grad_2d<double, M> m_grad_2d;
  dgc::laplace_2d<double, M> m_laplace_2d;

  // work space
  std::vector<double> m_u_tilde_x;
  std::vector<double> m_u_tilde_y;
  std::vector<double> m_workspace;
  std::vector<double> m_p;

  static constexpr double a = 2.883356;
  static constexpr double lamda = 1 + a * a;

private:
  // boundary conditins - see Table 1 of the 2017 paper by N. Fehn et al.
  struct u_tilde_bc
  {
    u_tilde_bc(double t, double t_prev) : m_t(t), m_t_prev(t_prev) {}

    // eq. 70 of the 2017 paper by N. Fehn et al.
    std::pair<double, double> exterior_val(double x, double y, std::pair<double, double> interior_val) const
    {
      double gUTildeX = 4. * std::sin(x) * (a * std::sin(a * y) - std::cos(a) * std::sinh(y)) * std::exp(- lamda * m_t) -
                   std::sin(x) * (a * std::sin(a * y) - std::cos(a) * std::sinh(y)) * std::exp(- lamda * m_t_prev);
      double gUTildeY = 4. * std::cos(x) * (std::cos(a * y) + std::cos(a) * std::cosh(y)) * std::exp(- lamda * m_t) -
                   std::cos(x) * (std::cos(a * y) + std::cos(a) * std::cosh(y)) * std::exp(- lamda * m_t_prev);
      return std::make_pair(2. * gUTildeX / 3. - interior_val.first, 2. * gUTildeY / 3. - interior_val.second);
    }

  private:
    double m_t;      // t(n)
    double m_t_prev; // t(n-1)
  };

  // used by the pressure step
  struct pressure_bc
  {
    // one-time calculation of the hp's (eq. 18) during construction
    pressure_bc(double t_next, int order, const double* inv_jacobians, const M* mesh,
                const int* face_0_nodes, const int* face_1_nodes, const int* face_2_nodes,
                const double* u_t_x, const double* u_t_y, const double* u_t_prev_x, const double* u_t_prev_y);

    double exterior_val(double x, double y, double interior_val) const { return interior_val; }

    double exterior_grad_n(double x, double y, double interior_grad_n) const
    {
      // NOTE: Below is a hack that assumes this function is called in the same order as those stored
      // in m_hps and only called once per node!!! Otherwise we need to use (x, y) to find the hp,
      // maybe via a map, i.e., extending m_hps to be (x, y) => hps, or something.
      auto hp = m_hps[m_offset_to_hps % m_hps.size()];
      ++m_offset_to_hps;

      return 2 * hp - interior_grad_n;
    }

  private:
    std::vector<double> m_hps;
    mutable std::size_t m_offset_to_hps;
  };

  // used by the projection step
  struct pressure_bc_proj
  { double exterior_val(double x, double y, double interior_val) const { return interior_val; } };

  // Dirichlet boundary conditions for velocity components
  struct u_x_bc
  {
    explicit u_x_bc(double t) : m_t(t) {}

    double exterior_val(double x, double y, double interior_val) const
    { return 2. * std::sin(x) * (a * std::sin(a * y) - std::cos(a) * std::sinh(y)) * std::exp(- lamda * m_t) - interior_val; }

    double exterior_grad_n(double x, double y, double interior_grad_n) const { return interior_grad_n; }

  private:
    double m_t; // t(n + 1)
  };

  struct u_y_bc
  {
    explicit u_y_bc(double t) : m_t(t) {}

    double exterior_val(double x, double y, double interior_val) const
    { return 2. * std::cos(x) * (std::cos(a * y) + std::cos(a) * std::cosh(y)) * std::exp(- lamda * m_t) - interior_val; }

    double exterior_grad_n(double x, double y, double interior_grad_n) const { return interior_grad_n; }

  private:
    double m_t; // t(n + 1)
  };
};

template<typename M>
stokes_2d<M>::pressure_bc::pressure_bc(double t_next, int order, const double* inv_jacobians, const M* mesh,
                                       const int* face_0_nodes, const int* face_1_nodes, const int* face_2_nodes,
                                       const double* u_t_x, const double* u_t_y, const double* u_t_prev_x, const double* u_t_prev_y)
: m_hps(), m_offset_to_hps(0)
{
  using point_type = dgc::point_2d<double>;

  reference_element refElem;
  std::vector<std::pair<double, double>> pos;
  refElem.node_positions(order, std::back_inserter(pos));
  const int* faceNodes[] = {face_0_nodes, face_1_nodes, face_2_nodes};

  dense_matrix_t v = refElem.vandermonde_matrix(order);
  dense_matrix_t vInv = v.inverse();
  auto vGrad = refElem.grad_vandermonde_matrix(order);
  dense_matrix_t Dr = vGrad.first * vInv;
  dense_matrix_t Ds = vGrad.second * vInv;

  int numCellNodes = refElem.num_nodes(order);
  int numFaceNodes = refElem.num_face_nodes(order);
  std::vector<double> curl(numCellNodes);
  std::vector<double> curlCurlX(numCellNodes);
  std::vector<double> curlCurlY(numCellNodes);
  for (int c = 0; c < mesh->num_cells(); ++c)
  {
    const auto cell = mesh->get_cell(c);

    bool isFaceBoundary[3];
    int nbCell, nbLocalEdgeIdx;
    for (int e = 0; e < 3; ++e)
      std::tie(isFaceBoundary[e], nbCell, nbLocalEdgeIdx) = mesh->get_face_neighbor(c, e);

    if (isFaceBoundary[0] || isFaceBoundary[1] || isFaceBoundary[2])
    {
      // compute the (curl curl u) term
      dense_matrix_t Dx = Dr * inv_jacobians[c * 4] + Ds * inv_jacobians[c * 4 + 2];
      dense_matrix_t Dy = Dr * inv_jacobians[c * 4 + 1] + Ds * inv_jacobians[c * 4 + 3];

      int cOffset = c * numCellNodes;
      // t(n)
      Dx.gemv(1, u_t_y + cOffset, 0, curl.begin());
      Dy.gemv(-1, u_t_x + cOffset, 1, curl.begin());

      Dy.gemv(1, curl.begin(), 0, curlCurlX.begin());
      Dx.gemv(-1, curl.begin(), 0, curlCurlY.begin());
      // t(n - 1)
      Dx.gemv(1, u_t_prev_y + cOffset, 0, curl.begin());
      Dy.gemv(-1, u_t_prev_x + cOffset, 1, curl.begin());

      Dy.gemv(1, curl.begin(), -2, curlCurlX.begin());
      Dx.gemv(-1, curl.begin(), -2, curlCurlY.begin());

      // store hp's
      for (int e = 0; e < 3; ++e)
        if (isFaceBoundary[e])
        {
          point_type n = cell.outward_normal(e);
          double nx = n.x();
          double ny = n.y();

          for(int i = 0; i < numFaceNodes; ++i)
          {
            int nodeIndex = faceNodes[e][i];
            double curlTerm = curlCurlX[nodeIndex] * nx + curlCurlY[nodeIndex] * ny;

            point_type xyPos = mapping::rs_to_xy(cell, point_type(pos[nodeIndex].first, pos[nodeIndex].second));
            double x = xyPos.x();
            double y = xyPos.y();
            double dgdtX = lamda * std::sin(x) * (a * std::sin(a * y) - std::cos(a) * std::sinh(y)) * std::exp(- lamda * t_next);
            double dgdtY = lamda * std::cos(x) * (std::cos(a * y) + std::cos(a) * std::cosh(y)) * std::exp(- lamda * t_next);
            double dgdtTerm = dgdtX * nx + dgdtY * ny;

            m_hps.push_back(dgdtTerm + curlTerm);
          }
        }
    }
  }
}

template<typename M>
stokes_2d<M>::stokes_2d(const M& mesh, int order)
: dgc::simple_discretization_2d<double, M>(mesh, order),
  m_div_2d(mesh, order), m_grad_2d(mesh, order), m_laplace_2d(mesh, order)
{
  // space allocation for intermediate variables
  auto numCells = this->m_mesh->num_cells();
  auto numCellNodes = reference_element().num_nodes(this->m_order);
  m_u_tilde_x.resize(numCells * numCellNodes);
  m_u_tilde_y.resize(numCells * numCellNodes);
  m_workspace.resize(numCells * numCellNodes);
  m_p.resize(numCells * numCellNodes);
}

template<typename M> template<typename PosItr, typename DofItr>
void stokes_2d<M>::initialize_dofs(PosItr posItr, DofItr dofItr) const
{
  using point_type = dgc::point_2d<double>;
  using cell_type = typename M::cell_type;

  std::vector<std::pair<double, double>> pos;
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

template<typename M> template<typename ZipItr>
void stokes_2d<M>::exact_solution(double t, ZipItr it) const
{
  using point_type = dgc::point_2d<double>;
  using cell_type = typename M::cell_type;

  std::vector<std::pair<double, double>> pos;
  reference_element().node_positions(this->m_order, std::back_inserter(pos));

  auto itTuple = it.get_iterator_tuple();
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
      *itUx++ = std::sin(pnt.x()) * (a * std::sin(a * pnt.y()) - std::cos(a) * std::sinh(pnt.y())) * std::exp(- lamda * t);
      *itUy++ = std::cos(pnt.x()) * (std::cos(a * pnt.y()) + std::cos(a) * std::cosh(pnt.y())) * std::exp(- lamda * t) ;
    }
}

template<typename M>
void stokes_2d<M>::pressure_step(double t, double t_prev, double dt,
                                 const double* u_t_x, const double* u_t_y, const double* u_t_prev_x, const double* u_t_prev_y)
{
  auto aEntries = build_poisson_sys_matrix(m_p, pressure_bc(t + dt, this->m_order, this->Inv_Jacobians.data(),
                                                            this->m_mesh, this->F0_Nodes.data(), this->F1_Nodes.data(),
                                                            this->F2_Nodes.data(), u_t_x, u_t_y, u_t_prev_x, u_t_prev_y));

  // right-hand-side
  std::copy(m_u_tilde_x.begin(), m_u_tilde_x.end(), m_workspace.begin());
  m_div_2d(m_workspace.begin(), m_u_tilde_y.begin(), u_tilde_bc(t, t_prev));
  for (std::size_t i = 0; i < m_workspace.size(); ++i) m_workspace[i] *= 1.5 / dt;

  solve_sys(aEntries, m_workspace, m_p);
}

template<typename M>
void stokes_2d<M>::projection_step(double dt)
{
  m_grad_2d(m_p.begin(), m_workspace.begin(), pressure_bc_proj());
  for (std::size_t i = 0; i < m_u_tilde_x.size(); ++i)
  {
    m_u_tilde_x[i] -= 2. * dt * m_p[i] / 3.;
    m_u_tilde_y[i] -= 2. * dt * m_workspace[i] / 3.;
  }
}

template<typename M>
void stokes_2d<M>::viscous_step(double t, double dt)
{
  // NOTE: if the inhomogeneous part of the BC was separated out,
  // NOTE: the same matrix could be used to solve both components

  // x component
  auto aEntries = build_helmholtz_sys_matrix(dt, m_workspace, u_x_bc(t + dt));
  std::copy(m_u_tilde_x.begin(), m_u_tilde_x.end(), m_workspace.begin());
  solve_sys(aEntries, m_workspace, m_u_tilde_x);

  // y component
  aEntries = build_helmholtz_sys_matrix(dt, m_workspace, u_y_bc(t + dt));
  std::copy(m_u_tilde_y.begin(), m_u_tilde_y.end(), m_workspace.begin());
  solve_sys(aEntries, m_workspace, m_u_tilde_y);
}

// (t_prev, t) => t + dt
template<typename M> template<typename ConstZipItr, typename ZipItr>
void stokes_2d<M>::advance_timestep(ConstZipItr in0, double t_prev, ConstZipItr in1, double t,
                                    std::size_t size, double dt, ZipItr out)
{
  // stage 1: get u_tilde - no convection term in stokes equation
  const auto in0_Tuple = in0.get_iterator_tuple();
  const auto in0_Ux = thrust::get<0>(in0_Tuple);
  const auto in0_Uy = thrust::get<1>(in0_Tuple);
  const auto in1_Tuple = in1.get_iterator_tuple();
  const auto in1_Ux = thrust::get<0>(in1_Tuple);
  const auto in1_Uy = thrust::get<1>(in1_Tuple);
  for (std::size_t i = 0; i < size; ++i)
  {
    m_u_tilde_x[i] = (*(in1_Ux + i) * 4. - *(in0_Ux + i)) / 3.;
    m_u_tilde_y[i] = (*(in1_Uy + i) * 4. - *(in0_Uy + i)) / 3.;
  }

  // stage 2
  pressure_step(t, t_prev, dt, &(*in1_Ux), &(*in1_Uy), &(*in0_Ux), &(*in0_Uy));
  projection_step(dt);

  // stage 3
  viscous_step(t, dt);

  // output
  const auto out_Tuple = out.get_iterator_tuple();
  const auto out_Ux = thrust::get<0>(out_Tuple);
  const auto out_Uy = thrust::get<1>(out_Tuple);
  for (std::size_t i = 0; i < size; ++i)
  {
    out_Ux[i] = m_u_tilde_x[i];
    out_Uy[i] = m_u_tilde_y[i];
  }
}

template<typename M> template<typename BC>
std::vector<std::tuple<int, int, double>> stokes_2d<M>::build_poisson_sys_matrix(std::vector<double>& temp, const BC& bc) const
{
  std::vector<std::tuple<int, int, double>> aEntries;
  for (size_t i = 0; i < temp.size(); ++i)
  {
    std::fill(temp.begin(), temp.end(), 0.0);
    temp[i] = 1.0;

    m_laplace_2d(temp.begin(), bc);

    for (std::size_t j = 0; j < temp.size(); ++j)
      if (std::abs(temp[j]) > 1e-10) aEntries.push_back(std::make_tuple(j, i, temp[j]));
  }
  return aEntries;
}

template<typename M> template<typename BC>
std::vector<std::tuple<int, int, double>> stokes_2d<M>::build_helmholtz_sys_matrix(double dt, std::vector<double>& temp, const BC& bc) const
{
  std::vector<std::tuple<int, int, double>> aEntries;
  for (size_t i = 0; i < temp.size(); ++i)
  {
    std::fill(temp.begin(), temp.end(), 0.0);
    temp[i] = 1.0;

    m_laplace_2d(temp.begin(), bc);

    for (std::size_t j = 0; j < temp.size(); ++j)
    {
      temp[j] *= (- 2. * dt / 3.);
      if (j == i) temp[j] += 1.0;
      if (std::abs(temp[j]) > 1e-10) aEntries.push_back(std::make_tuple(j, i, temp[j]));
    }
  }
  return aEntries;
}

template<typename M>
void stokes_2d<M>::solve_sys(std::vector<std::tuple<int, int, double>>& matrix, const std::vector<double>& b,
                             std::vector<double>& x)
{
  // convert to csr format
  std::sort(matrix.begin(), matrix.end(),
       [](const std::tuple<int, int, double>& a, const std::tuple<int, int, double>& b)
       { return std::get<0>(a) == std::get<0>(b) ? std::get<1>(a) < std::get<1>(b) : std::get<0>(a) < std::get<0>(b); });

  std::vector<int> csrRowPtrA;
  std::vector<int> csrColIndA;
  std::vector<double> csrValA;
  int prev_row = -1;
  for(int entryIdx = 0; entryIdx < matrix.size(); ++entryIdx)
  {
    auto entry = matrix[entryIdx];
    int row = std::get<0>(entry);
    if (row == prev_row)
    {
      csrColIndA.push_back(std::get<1>(entry));
      csrValA.push_back(std::get<2>(entry));
    }
    else
    {
      assert(row > prev_row);
      for (int i = prev_row; i < row; ++i)
        csrRowPtrA.push_back(csrValA.size());
      prev_row = row;
      entryIdx--; // must use signed int for this index as it may become negative when first enter here
    }
  }
  for (int i = csrRowPtrA.size(); i <= x.size(); ++i)
    csrRowPtrA.push_back(csrValA.size());
  assert(csrValA.size() == matrix.size());

  // initialize cuSPARSE
  cusparseHandle_t handle;
  cusparseStatus_t cusparse_status = cusparseCreate(&handle);
  assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

  // descriptor for sparse matrix A
  cusparseMatDescr_t descrA;
  cusparse_status = cusparseCreateMatDescr(&descrA);
  assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

  // cuSPARSE linear solver
  cusolverSpHandle_t solver_handle;
  cusolverStatus_t cusolver_status = cusolverSpCreate(&solver_handle);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

  int singularity;
  cusolver_status = cusolverSpDcsrlsvluHost(solver_handle, x.size(), csrValA.size(),
                                            descrA, csrValA.data(), csrRowPtrA.data(), csrColIndA.data(),
                                            b.data(), 1e-10, 0, x.data(), &singularity);

  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
  assert(singularity < 0);

  cusolver_status = cusolverSpDestroy(solver_handle);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
}

#endif
