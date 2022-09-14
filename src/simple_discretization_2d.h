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

#ifndef SIMPLE_DISCRETIZATION_2D_H
#define SIMPLE_DISCRETIZATION_2D_H 

#include <utility>
#include <vector>

#include "config.h"
#include "basic_geom_2d.h"
#include "dense_matrix.h"
#include "reference_triangle.h"
#include "mapping_2d.h"

BEGIN_NAMESPACE

// Simple discretization on host - "simple" means conformal, single cell shape (i.e.,
// triangle) mesh with same approximation order across all cells, and no adaptation.
// This class is intended to be used by specific problems of 2D.
template<typename T, typename M> // T - number type, M - mesh type
class simple_discretization_2d
{
public:
  using index_type = typename M::index_type;

  simple_discretization_2d(const M& mesh, int order); 
  ~simple_discretization_2d(){}
  
  index_type total_num_nodes() const; 

  // prepare for GPU execution
  template<typename OutputIterator1, typename OutputIterator2, typename OutputIterator3>
  void fill_cell_mappings(OutputIterator1 it1, OutputIterator2 it2, OutputIterator3 it3) const;
  template<typename OutputIterator1, typename OutputIterator2>
  void fill_cell_interfaces(OutputIterator1 it1, OutputIterator2 it2) const;
  template<typename OutputIterator1, typename OutputIterator2>
  index_type fill_boundary_nodes(OutputIterator1 it1, OutputIterator2 it2) const; // return the number of boundary nodes
  template<typename OutputIterator1, typename OutputIterator2>
  void fill_outward_normals(OutputIterator1 it1, OutputIterator2 it2) const;

  // these are accessed frequently so pre-compute them and store them
  // as public members
  std::vector<int> F0_Nodes;
  std::vector<int> F1_Nodes;
  std::vector<int> F2_Nodes;

  // mappings of inverse Jacobian matrix for each cell is relatively expensive
  // to compute so pre-compute them, but for J and face Js which are relatively
  // cheap to compute, they will be computed on-the-fly
  std::vector<T> Inv_Jacobians;

protected:
  using dense_matrix_t = dense_matrix<T, false>; // row major
  using reference_element = reference_triangle<T>;
  using mapping = mapping_2d<T>;

  const M*  m_mesh; // no mesh adaptation here so use a constant pointer
  int       m_order;
};

template<typename T, typename M>
simple_discretization_2d<T, M>::simple_discretization_2d(const M& mesh, int order)
  : m_mesh(std::addressof(mesh)), m_order(order)
{
  reference_element refElem;
  refElem.face_nodes(m_order, 0, std::back_inserter(F0_Nodes));
  refElem.face_nodes(m_order, 1, std::back_inserter(F1_Nodes));
  refElem.face_nodes(m_order, 2, std::back_inserter(F2_Nodes));

  Inv_Jacobians.resize(m_mesh->num_cells() * 4);
  for (int c = 0; c < m_mesh->num_cells(); ++c)
  {
    dense_matrix_t jInv = mapping::jacobian_matrix(m_mesh->get_cell(c));
    jInv = jInv.inverse();
    Inv_Jacobians[c * 4] = jInv(0, 0);
    Inv_Jacobians[c * 4 + 1] = jInv(0, 1);
    Inv_Jacobians[c * 4 + 2] = jInv(1, 0);
    Inv_Jacobians[c * 4 + 3] = jInv(1, 1);
  }
}

template<typename T, typename M>
typename simple_discretization_2d<T, M>::index_type
simple_discretization_2d<T, M>::total_num_nodes() const
{
  reference_element refElem;
  return m_mesh->num_cells() * refElem.num_nodes(m_order);
}

template<typename T, typename M> template<typename OutputIterator1, typename OutputIterator2, typename OutputIterator3>
void simple_discretization_2d<T, M>::fill_cell_mappings(OutputIterator1 it1, OutputIterator2 it2, OutputIterator3 it3) const
{
  for (int c = 0; c < m_mesh->num_cells(); ++c)
  {
    *it1++ = Inv_Jacobians[c * 4];
    *it1++ = Inv_Jacobians[c * 4 + 1];
    *it1++ = Inv_Jacobians[c * 4 + 2];
    *it1++ = Inv_Jacobians[c * 4 + 3];

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
void simple_discretization_2d<T, M>::fill_cell_interfaces(OutputIterator1 it1, OutputIterator2 it2) const
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
typename simple_discretization_2d<T, M>::index_type
simple_discretization_2d<T, M>::fill_boundary_nodes(OutputIterator1 it1, OutputIterator2 it2) const
{
  reference_element refElem;
  const int numFaceNodes = refElem.num_face_nodes(m_order);
  index_type numBoundaryNodes = 0;

  using point_type = point_2d<T>;
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
        const std::vector<int>& faceNodes = e == 0 ? F0_Nodes : (e == 1 ? F1_Nodes : F2_Nodes);
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
void simple_discretization_2d<T, M>::fill_outward_normals(OutputIterator1 it1, OutputIterator2 it2) const
{
  for (int c = 0; c < m_mesh->num_cells(); ++c)
  {
    const auto cell = m_mesh->get_cell(c);
    for (int e = 0; e < 3; ++e)
    {
      point_2d<T> n = cell.outward_normal(e);
      *it1++ = n.x();
      *it2++ = n.y();
    }
  }
}

END_NAMESPACE

#endif
