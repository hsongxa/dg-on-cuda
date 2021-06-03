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

#ifndef EXPLICIT_RUNGE_KUTTA_H
#define EXPLICIT_RUNGE_KUTTA_H

#include <cstddef>
#include <cassert>

#include "execution_policy.h"

BEGIN_NAMESPACE

// Vector type could be composite type, i.e., containing multiple vectors and it must support
// add(double, const Vector& in, Vector& out), add(const Vector& in), and scale(double s);
// and DiscreteOp type must overload operator() (const Vector& in, double, Vector& out)
/*template <typename Vector, typename DiscreteOp>
void rk4(Vector& inout, double t, double dt, DiscreteOp op, Vector& wk0, Vector& wk1, Vector& wk2, Vector& wk3, Vector& wk4)
{
	op(inout, t, wk1);

	inout.add(0.5 * dt, wk1, wk0);
	op(wk0, t + 0.5 * dt, wk2);

	inout.add(0.5 * dt, wk2, wk0);
	op(wk0, t + 0.5 * dt, wk3);

	inout.add(dt, wk3, wk0);
	op(wk0, t + dt, wk4);

	wk1.add(2.0, wk2, wk0);
	wk4.add(2.0, wk3, wk1);
	wk0.add(wk1);
	wk0.scale(dt / 6.0);
	inout.add(wk0);
}*/

// Vector type could be composite type, i.e., containing multiple vectors and it must support
// add(double, const Vector& in, Vector& out), add(const Vector& in), and scale(double s);
// DiscreteOp type must overload operator() (const Vector& in, double, Vector& out); and
// Limiter type must overload operator() (Vector& inOut)
template <typename Vector, typename DiscreteOp, typename Limiter>
void optimal_rk2_with_limiter(Vector& inout, double t, double dt, DiscreteOp op, Limiter limiter, Vector& wk0, Vector& wk1)
{
	op(inout, t, wk0);
	inout.add(dt, wk0, wk1);
	limiter(wk1);

	op(wk1, t + dt, wk0);
	wk0.scale(dt);
	wk1.add(wk0);
	inout.add(wk1);
	inout.scale(0.5);
	limiter(inout);
}

// Vector type could be composite type, i.e., containing multiple vectors, and it must support
// add(double, const Vector& in, Vector& out), add(const Vector& in), and scale(double s);
// DiscreteOp type must overload operator() (const Vector& in, double, Vector& out); and
// Limiter type must overload operator() (Vector& inOut)
template <typename Vector, typename DiscreteOp, typename Limiter>
void optimal_rk3_with_limiter(Vector& inout, double t, double dt, DiscreteOp op, Limiter limiter, Vector& wk0, Vector& wk1, Vector& wk2)
{
	op(inout, t, wk0);
	inout.add(dt, wk0, wk1);
	limiter(wk1);

	op(wk1, t + dt, wk0);
	wk1.add(dt, wk0, wk2);
	wk2.scale(1.0 / 3.0);
	wk2.add(inout);
	wk2.scale(0.75);
	limiter(wk2);

	op(wk2, t + 0.5 * dt, wk0);
	wk2.add(dt, wk0, wk1);
	wk1.scale(2.0);
	inout.add(wk1);
	inout.scale(1.0 / 3.0);
	limiter(inout);
}

// forward euler
// Vector type could be composite type, i.e., containing multiple vectors and it must support
// add(const Vector& in), and scale(double s); and DiscreteOp type must overload operator()
// (const Vector& in, double, Vector& out)
template <typename Vector, typename DiscreteOp>
void fe1(Vector& inout, double t, double dt, DiscreteOp op, Vector& wk)
{
	op(inout, t, wk);
	wk.scale(dt);
	inout.add(wk);
}

template <typename ConstItr, typename T, typename Itr>
void axpy_n(T a, ConstItr x_cbegin, std::size_t x_size, ConstItr y_cbegin, Itr out_begin)
{
  assert(out_begin != x_cbegin && out_begin != y_cbegin);

  for (std::size_t i = 0; i < x_size; ++i)
    *out_begin++ = a * (*x_cbegin++) + (*y_cbegin++);
}

// fourth-order explicit Runge-Kutta method for scalar variable
template <typename Itr, typename T, typename DiscreteOp>
void rk4(cpu_policy policy, Itr inout, std::size_t size, T t, T dt, const DiscreteOp& op, Itr wk0, Itr wk1, Itr wk2, Itr wk3, Itr wk4)
{
  Itr inout_begin = inout;
  Itr wk1_begin = wk1;
  op(policy, inout_begin, size, t, wk1_begin);

  Itr wk0_begin = wk0;
  inout_begin = inout;
  wk1_begin = wk1;
  axpy_n((T)(0.5L) * dt, wk1_begin, size, inout_begin, wk0_begin);

  wk0_begin = wk0;
  Itr wk2_begin = wk2;
  op(policy, wk0_begin, size, t + (T)(0.5L) * dt, wk2_begin);

  wk0_begin = wk0;
  inout_begin = inout;
  wk2_begin = wk2;
  axpy_n((T)(0.5L) * dt, wk2_begin, size, inout_begin, wk0_begin);

  wk0_begin = wk0;
  Itr wk3_begin = wk3;
  op(policy, wk0_begin, size, t + (T)(0.5L) * dt, wk3_begin);

  wk0_begin = wk0;
  inout_begin = inout;
  wk3_begin = wk3;
  axpy_n(dt, wk3_begin, size, inout_begin, wk0_begin);

  wk0_begin = wk0;
  Itr wk4_begin = wk4;
  op(policy, wk0_begin, size, t + dt, wk4_begin);

  wk0_begin = wk0;
  wk1_begin = wk1;
  wk2_begin = wk2;
  axpy_n((T)(2.0L), wk2_begin, size, wk1_begin, wk0_begin);

  wk1_begin = wk1;
  wk4_begin = wk4;
  wk3_begin = wk3;
  axpy_n((T)(2.0L), wk3_begin, size, wk4_begin, wk1_begin);

  wk2_begin = wk2;
  inout_begin = inout;
  wk0_begin = wk0;
  axpy_n(dt / (T)(6.0L), wk0_begin, size, inout_begin, wk2_begin);

  inout_begin = inout;
  wk2_begin = wk2;
  wk1_begin = wk1;
  axpy_n(dt / (T)(6.0L), wk1_begin, size, wk2_begin, inout_begin);
}

template <typename T, typename DiscreteOp>
void rk4(gpu_policy policy, T* inout, std::size_t size, T t, T dt, const DiscreteOp& op, T* wk0, T* wk1, T* wk2, T* wk3, T* wk4)
{
  op(policy, inout, size, t, wk1);

}

END_NAMESPACE

#endif
