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

#include "linear_wave_1d.h"
#include "explicit_runge_kutta.h"

using real = typename linear_wave_1d::real;

void print_vector(std::vector<real>& v)
{
  std::cout << "vector size:" << v.size() << std::endl;
  for (int i = 0; i < v.size(); ++i) std::cout << v[i] << std::endl;
}

real compute_error_norm(std::vector<real>& x, std::vector<real>& v, real t)
{
  real err = 0.0;
  for(int i = 0; i < v.size(); ++i)
    err += (std::sin(x[i] - 2. * M_PI * t) - v[i]) * (std::sin(x[i] - 2. * M_PI * t) - v[i]);
  return err / v.size();
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
  std::vector<real> v(numDOFs);
  for (int i = 0; i < v.size(); ++i) v[i] = std::sin(x[i]);

  // allocate work space for runge-kutta loop
  std::vector<real> v1(numDOFs);	
  std::vector<real> v2(numDOFs);
  std::vector<real> v3(numDOFs);
  std::vector<real> v4(numDOFs);
  std::vector<real> v5(numDOFs);
  
  // time advancing loop
  int totalTSs = 3000;
  real t = 0.0;
  real dt = 0.15 * 0.75 * op.min_elem_size() / op.wave_speed();
  for (int i = 0; i < totalTSs; ++i)
  {
#if defined USE_CPU_ONLY
    dgc::rk4(dgc::cpu, v.begin(), v.size(), t, dt, op, v1.begin(), v2.begin(), v3.begin(), v4.begin(), v5.begin());
#else
    dgc::rk4(dgc::gpu, v.data(), v.size(), t, dt, op, v1.data(), v2.data(), v3.data(), v4.data(), v5.data());
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

  return 0;
}
