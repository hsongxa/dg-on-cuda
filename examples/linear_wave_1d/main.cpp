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

#include <vector>
#include <iterator>
#include <iostream>
#include <fstream>
#include <limits>

#include <cuda_runtime.h>

#include "linear_wave_1d.h"
#include "explicit_runge_kutta.h"

using real = typename linear_wave_1d::real;

real compute_error_norm(std::vector<real>& x, real* v, real t)
{
  real err = 0.0;
  for(int i = 0; i < x.size(); ++i)
    err += (std::sin(x[i] - 2. * M_PI * t) - v[i]) * (std::sin(x[i] - 2. * M_PI * t) - v[i]);
  return err / x.size();
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

  const int numCells = 100;
  const int order = 1;
  linear_wave_1d op(numCells, order); // op(numCells, order, false);

  std::vector<real> x;
  op.dof_positions(std::back_inserter(x));

  // initial condition
  int numDOFs = numCells * (order +  1);
  real* v;
  cudaMallocManaged(&v, numDOFs * sizeof(real)); // unified memory
  for (int i = 0; i < numDOFs; ++i) v[i] = std::sin(x[i]);

  // allocate work space for Runge-Kutta loop
  real* v1;
  real* v2;
  real* v3;
  real* v4;
  real* v5;
  cudaMallocManaged(&v1, numDOFs * sizeof(real));
  cudaMallocManaged(&v2, numDOFs * sizeof(real));
  cudaMallocManaged(&v3, numDOFs * sizeof(real));
  cudaMallocManaged(&v4, numDOFs * sizeof(real));
  cudaMallocManaged(&v5, numDOFs * sizeof(real));
  
  // time advancing loop
  int totalTSs = 3000;
  real t = 0.0;
real dt = 0.15 * 0.75 * op.min_elem_size() / op.wave_speed();
for (int i = 0; i < totalTSs; ++i)
{
#if defined USE_CPU_ONLY
  dgc::rk4(v, numDOFs, t, dt, op, v1, v2, v3, v4, v5);
#else
  dgc::d_rk4(v, numDOFs, t, dt, op, v1, v2, v3, v4, v5);
  cudaDeviceSynchronize();
#endif
  t += dt;
  real errNorm = compute_error_norm(x, v, t);
  std::cout << "t = " << t << ", error norm = " << errNorm << std::endl;
}

// output to visualize
std::ofstream file;
file.open("LinearWave1DDataFile.txt");
file.precision(std::numeric_limits<real>::digits10);
file << "#         x         y" << std::endl;
for(int i = 0; i < numDOFs; ++i)
  file << x[i] << "  " << v[i] << std::endl;
file << std::endl;
file << "#         x         reference solution" << std::endl;
for(int i = 0; i < numDOFs; ++i)
  file << x[i] << " " << std::sin(x[i] - op.wave_speed() * t) << std::endl;

cudaFree(v);
cudaFree(v1);
cudaFree(v2);
cudaFree(v3);
cudaFree(v4);
cudaFree(v5);
return 0;
}
