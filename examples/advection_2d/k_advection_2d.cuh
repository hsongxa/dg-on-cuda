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

#ifndef K_ADVECTION_2D_CUH
#define K_ADVECTION_2D_CUH

#include "d_advection_2d.cuh"

// NOTE: See the note of the companion .cu file.

d_advection_2d<double, int>* create_device_object(int num_cells, int order, double* dr, double* ds, double* l,
                                                  double* inv_jacobians, double* Js, double* face_Js,
                                                  int* interface_cells, int* interface_faces, int num_boundary_nodes,
                                                  double* boundary_node_Xs, double* boundary_node_Ys,
                                                  double** d_inv_jacobians, double** d_Js, double** d_face_Js,
                                                  int** d_interface_cells, int** d_interface_faces,
                                                  double** d_boundary_node_Xs, double** d_boundary_node_Ys);

void rk4_on_device(int gridSize, int blockSize, double* inout, std::size_t size, double t, double dt,
                   d_advection_2d<double, int>* d_op, double* wk0, double* wk1, double* wk2, double* wk3, double* wk4);

void destroy_device_object(d_advection_2d<double, int>* device_obj, double* d_inv_jacobians, double* d_Js, double* d_face_Js,
                           int* d_interface_cells, int* d_interface_faces, double* d_boundary_node_Xs, double* d_boundary_node_Ys);

#endif
