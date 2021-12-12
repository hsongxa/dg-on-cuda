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

#ifndef K_MAXWELL_2D_CUH
#define K_MAXWELL_2D_CUH

#include <thrust/device_vector.h>

#include "d_maxwell_2d.cuh"

// NOTE: See the note of the companion .cu file.

// "D" stands for device
using DIntVector = thrust::device_vector<int>;
using DDblVector = thrust::device_vector<double>;
using DDblIterator = DDblVector::iterator;
using DIteratorTuple = thrust::tuple<DDblIterator, DDblIterator, DDblIterator>;
using DZipIterator = thrust::zip_iterator<DIteratorTuple>;

d_maxwell_2d<double, int>* create_device_object(int num_cells, int order, double* dr, double* ds, double* l,
                                                const DIntVector& face_0_nodes,
                                                const DIntVector& face_1_nodes,
                                                const DIntVector& face_2_nodes,
                                                const DDblVector& inv_jacobians,
                                                const DDblVector& Js,
                                                const DDblVector& face_Js,
                                                const DIntVector& interface_cells,
                                                const DIntVector& interface_faces,
                                                int num_boundary_nodes,
                                                const DDblVector& boundary_node_Xs,
                                                const DDblVector& boundary_node_Ys,
                                                const DDblVector& outward_normal_Xs,
                                                const DDblVector& outward_normal_Ys);

void rk4_on_device(int gridSize, int blockSize, DZipIterator inout, std::size_t size, double t, double dt,
                   d_maxwell_2d<double, int>* d_op, DZipIterator wk0, DZipIterator wk1, DZipIterator wk2,
                   DZipIterator wk3, DZipIterator wk4);

void destroy_device_object(d_maxwell_2d<double, int>* device_obj);

#endif
