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

#ifndef GMSH_IMPORTER_EXPORTER_H
#define GMSH_IMPORTER_EXPORTER_H

#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <type_traits>
#include <cassert>

#include "simple_triangular_mesh_2d.h"

BEGIN_NAMESPACE

// quick-and-dirty implementation of import and export of the MSH file format (version 4) of Gmsh
template<typename CT, typename IT>
class gmsh_importer_exporter
{
public:
  static std::unique_ptr<simple_triangular_mesh_2d<CT, IT>> import_triangle_mesh_2d(const std::string& file_name);  

  static void export_triangle_mesh_2d(const simple_triangular_mesh_2d<CT, IT>& mesh, const std::string& file_name);  
};

template<typename CT, typename IT>
std::unique_ptr<simple_triangular_mesh_2d<CT, IT>> gmsh_importer_exporter<CT, IT>::import_triangle_mesh_2d(const std::string& file_name)
{
  auto mesh_ptr = std::make_unique<simple_triangular_mesh_2d<CT, IT>>();
  std::vector<CT> coords;
  std::vector<IT> conns;

  std::ifstream file(file_name);
  if (file.is_open())
  {
    std::string line;
    while (std::getline(file, line))
      if (line.find("$MeshFormat") != std::string::npos)
      {
        std::getline(file, line);
        assert(std::stof(line) > 4.0); // version requirement
        break;
      }

    IT num_blocks, num_entries, min_index, max_index;

    // section of nodes
    IT num_nodes = 0;
    while (std::getline(file, line))
      if (line.find("$Nodes") != std::string::npos)
      {
        std::getline(file, line); // this line shows the number of blocks
        if constexpr(std::is_signed<IT>::value) num_blocks = std::stoll(line);
        else num_blocks = std::stoull(line);

        line.erase(0, line.find_first_of(' ') + 1);
        line.erase(0, line.find_first_of(' ') + 1);
        if constexpr(std::is_signed<IT>::value) min_index = std::stoll(line);
        else min_index = std::stoull(line);

        line.erase(0, line.find_first_of(' ') + 1);
        if constexpr(std::is_signed<IT>::value) max_index = std::stoll(line);
        else max_index = std::stoull(line);

        assert(num_blocks > 0);
        for (IT b = 0; b < num_blocks; ++b)
        {
          std::getline(file, line); // this line shows the number of nodes in this block
          line = line.substr(line.find_last_of(' ') + 1);
          if constexpr(std::is_signed<IT>::value) num_entries = std::stoll(line);
          else num_entries = std::stoull(line);
          
          for (IT l = 0; l < num_entries; ++l)
            std::getline(file, line); // ignore node indices - assume they are continuous
          for (IT l = 0; l < num_entries; ++l)
          {
            std::getline(file, line);

            coords.push_back(std::stold(line)); 
            line.erase(0, line.find_first_of(' ') + 1);
            coords.push_back(std::stold(line)); 

            num_nodes++;
          }
        }

        break;
      }

    // assume the indices are continuous - if not, we will have to explicitly store indeices of nodes
    assert(num_nodes == (max_index - min_index + 1));
    assert((coords.size() % 2) == 0);
    mesh_ptr->fill_vertices(coords.begin(), coords.end());
    coords.clear();

    // section of cells 
    while (std::getline(file, line))
      if (line.find("$Elements") != std::string::npos)
      {
        std::getline(file, line); // this line shows the number of blocks
        if constexpr(std::is_signed<IT>::value) num_blocks = std::stoll(line);
        else num_blocks = std::stoull(line);

        assert(num_blocks > 0);
        for (IT b = 0; b < num_blocks; ++b)
        {
          std::getline(file, line); // this line shows element type and number in this block
          line.erase(0, line.find_first_of(' ') + 1);
          line.erase(0, line.find_first_of(' ') + 1);
          int type = std::stoi(line); // see the definition of types in the Gmsh documentation

          line.erase(0, line.find_first_of(' ') + 1);
          if constexpr(std::is_signed<IT>::value) num_entries = std::stoll(line);
          else num_entries = std::stoull(line);
 
          for (IT l = 0; l < num_entries; ++l)
          {
            std::getline(file, line); // assumption: regardless of type, each entry has only one line

            if (type == 2) // we only care about triangles here
            {
              line.erase(0, line.find_first_of(' ') + 1);

              if constexpr(std::is_signed<IT>::value) conns.push_back(std::stoll(line) - min_index);
              else conns.push_back(std::stoull(line) - min_index);

              line.erase(0, line.find_first_of(' ') + 1);
              if constexpr(std::is_signed<IT>::value) conns.push_back(std::stoll(line) - min_index);
              else conns.push_back(std::stoull(line) - min_index);

              line.erase(0, line.find_first_of(' ') + 1);
              if constexpr(std::is_signed<IT>::value) conns.push_back(std::stoll(line) - min_index);
              else conns.push_back(std::stoull(line) - min_index);
            }
          }
        }

        break;
      }

    assert((conns.size() % 3) == 0);
    mesh_ptr->fill_connectivity(conns.begin(), conns.end());
    conns.clear();

    file.close();
  }

  mesh_ptr->build_topology();
  return mesh_ptr;
}

template<typename CT, typename IT>
void gmsh_importer_exporter<CT, IT>::export_triangle_mesh_2d(const simple_triangular_mesh_2d<CT, IT>& mesh, const std::string& file_name)
{
  std::ofstream out(file_name);
  if (out.is_open())
  {
    out << "$MeshFormat" << std::endl;
    out << "4.1 0 8" << std::endl;
    out << "$EndMeshFormat" << std::endl;
    
    // section of nodes
    IT num_entries = mesh.num_vertices();
    out << "$Nodes" << std::endl;
    out << "1 " << num_entries << " 1 " << num_entries << std::endl;
    out << "2 1 0 " << num_entries << std::endl; // dummy entityDim and entityTag
    for (IT i = 1; i <= num_entries; ++i)
      out << i << std::endl;
    for (IT i = 1; i <= num_entries; ++i)
    {
      auto vertex = mesh.get_vertex(i - 1);
      out << vertex.x() << " " << vertex.y() << " 0" << std::endl;
    }
    out << "$EndNodes" << std::endl;

    // section of elements
    num_entries = mesh.num_cells();
    out << "$Elements" << std::endl;
    out << "1 " << num_entries << " 1 " << num_entries << std::endl;
    out << "2 1 2 " << num_entries << std::endl; // dummy entityDim and entityTag, elementType = 2 (triangle)
    for (IT i = 1; i <= num_entries; ++i)
    {
      auto conn = mesh.get_cell_connectivity(i - 1);
      out << i << " " << std::get<0>(conn) + 1 << " " << std::get<1>(conn) + 1 << " " << std::get<2>(conn) + 1 << std::endl;
    }
    out << "$EndElements" << std::endl;

    out.close();
  }
}

END_NAMESPACE

#endif
