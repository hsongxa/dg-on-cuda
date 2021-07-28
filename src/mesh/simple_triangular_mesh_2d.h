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

#ifndef SIMPLE_TRIANGULAR_MESH_2D_H
#define SIMPLE_TRIANGULAR_MESH_2D_H

#include <vector>
#include <map>
#include <utility>
#include <tuple>
#include <cassert>

#include "basic_geom_2d.h"
#include "encoded_integer.h"

BEGIN_NAMESPACE

// "Simple" means conformal, non-adaptive (no refinement or coarsening), and serial (no partitioning),
// but the public typedefs and functions of this class define the minimum API needed for the DG method.
//
// GPU execution may choose to map cells to threads, hence the general concept of cell iterator/handle
// has to reduce to cell index in order to match thread index. However, the user should be able to
// specify the integer type (int32, int64, etc.) to use as the index depending on the size of the mesh.
//
// Note that the specified index type must leave an extra 2 bits of room to encode the half-facet
// handle, which is the cell index encoded with the local edge index. In the 2D mesh, we use the terms
// "face" and "edge" interchangeably so "facet" actually means "half-edge" here.
template<typename CT, typename IT> // CT - vertex coordinate type, IT - index type for vertices and cells
class simple_triangular_mesh_2d
{
public:
  using vertex_type = point_2d<CT>;
  using face_type = segment_2d<vertex_type>;
  using cell_type = triangle_2d<vertex_type>;

  // build the mesh

  template<class InputIt>
  void fill_vertices(InputIt first, InputIt last)
  {
    m_vertices.clear();
    while (first != last)
      m_vertices.emplace_back(*first++, *first++, hf_handle());
  }

  template<class InputIt>
  void fill_connectivity(InputIt first, InputIt last)
  {
    m_cells.clear();
    cell_storage c;
    while (first != last)
    {
      c.connectivity[0] = *first++;
      c.connectivity[1] = *first++;
      c.connectivity[2] = *first++;
      m_cells.push_back(c);
    }
  }

  void build_topology(); // build the twin half-facet and vertex-to-hf maps

  // queries - both geometry and topology

  IT num_vertices() const { return m_vertices.size(); }

  vertex_type get_vertex(IT vi) const { return vertex_type(std::get<0>(m_vertices[vi]), std::get<1>(m_vertices[vi])); }

  IT num_cells() const { return m_cells.size(); }

  cell_type get_cell(IT ci) const
  {
    const IT* v = m_cells[ci].connectivity;
    return cell_type(vertex_type(std::get<0>(m_vertices[v[0]]), std::get<1>(m_vertices[v[0]])),
                     vertex_type(std::get<0>(m_vertices[v[1]]), std::get<1>(m_vertices[v[1]])),
                     vertex_type(std::get<0>(m_vertices[v[2]]), std::get<1>(m_vertices[v[2]])));
  }

  // this is only needed for mesh export
  std::tuple<IT, IT, IT> get_cell_connectivity(IT ci) const
  {
    const IT* v = m_cells[ci].connectivity;
    return std::make_tuple(v[0], v[1], v[2]);
  }

  face_type get_face(IT ci, int face) const
  {
    assert(face >= 0 && face < 3);
    IT* v = m_cells[ci].connectivity;
    return face_type(vertex_type(std::get<0>(m_vertices[v[face]]), std::get<1>(m_vertices[v[face]])),
                     vertex_type(std::get<0>(m_vertices[v[(face + 1) % 3]]), std::get<1>(m_vertices[v[(face + 1) % 3]])));
  }

  std::tuple<bool, IT, int> get_face_neighbor(IT ci, int face) const
  {
    assert(face >= 0 && face < 3);
    hf_handle hf = m_cells[ci].twinhf[face];
    return std::make_tuple(hf.integer() != ci, hf.integer(), hf.code());
  }

  // output all the cell interfaces - needed for GPU execution
  template<typename OutputIt>
  void get_face_mapping(OutputIt it) const
  {
    for (std::size_t c = 0; c < m_cells.size(); ++c)
    {
      *it++ = m_cells[c].twinhf[0].integer_representation();
      *it++ = m_cells[c].twinhf[1].integer_representation();
      *it++ = m_cells[c].twinhf[2].integer_representation();
    }
  }

  // TODO: patches - same concept as in Gmsh and OpenFOAM

private:
  // "hf" stands for half-facet
  using hf_handle = encoded_integer<IT, 2>; // two bits to encode the local edge index [0, 1, 2]

  struct cell_storage
  {
    IT connectivity[3];
    hf_handle twinhf[3];
  };

  std::vector<std::tuple<CT, CT, hf_handle>> m_vertices; // priority is given to boundary half-facets
  std::vector<cell_storage>                  m_cells;
};

template<typename CT, typename IT>
void simple_triangular_mesh_2d<CT, IT>::build_topology()
{
  // the twin half-facet map
  std::map<std::pair<IT, IT>, hf_handle> dict;
  for (std::size_t c = 0; c < m_cells.size(); ++c)
  {
    IT v[3] = {m_cells[c].connectivity[0], m_cells[c].connectivity[1], m_cells[c].connectivity[2]};
    for (int e = 0; e < 3; ++e)
    {
      IT v0 = v[e];
      IT v1 = v[(e + 1) % 3];
      assert(v0 != v1);
      std::pair<IT, IT> edge = v0 < v1 ? std::make_pair(v0, v1) : std::make_pair(v1, v0);

      // if already exists a hf with the same key, then they are twin hfs
      auto it_twin = dict.find(edge);
      if (it_twin != dict.end())
      {
        hf_handle hf = it_twin->second;
        m_cells[hf.integer()].twinhf[hf.code()] = hf_handle(c, e);
        m_cells[c].twinhf[e] = hf;
        dict.erase(it_twin);
      }
      // otherwise store self
      else
      {
        hf_handle hf(c, e);
        m_cells[c].twinhf[e] = hf;
        dict.emplace(edge, hf);
      }
    }
  }

  // the vertex-to-hf map (this part may not be used by DG but we follow the half-facet data structure anyway)
  std::vector<bool> populated(m_vertices.size(), false);
  for (IT c = 0; c < m_cells.size(); ++c)
  {
    // populate vertices with the first incident facet if they are not populated yet
    for (int v = 0; v < 3; ++v)
    {
      IT vtx = m_cells[c].connectivity[v];
      if (populated[vtx] == false)
      {
        std::get<2>(m_vertices[vtx]) = hf_handle(c, v); // use v for the local edge index
        populated[vtx] = true;
      }
    }

    // give priority to boundary facet
    for (int e = 0; e < 3; ++e)
    {
      if (m_cells[c].twinhf[e].integer() == c) // boundary hf's twin is itself
      {
        assert(m_cells[c].twinhf[e].code() == e);

        IT vtx = m_cells[c].connectivity[e]; // use e for the first local vertex
        assert(populated[vtx]); 
        std::get<2>(m_vertices[vtx]) = hf_handle(c, e); // overwritten by this boundary facet

        vtx = m_cells[c].connectivity[(e + 1) % 3]; // similar idea to find the second local vertex
        assert(populated[vtx]);
        std::get<2>(m_vertices[vtx]) = hf_handle(c, e); // overwritten by this boundary facet
      }
    }
  }
}

END_NAMESPACE

#endif
