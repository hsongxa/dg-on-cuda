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

BEGIN_NAMESPACE

// fourth-order explicit Runge-Kutta method
template <typename Itr, typename T, typename DiscreteOp, typename Axpy>
void rk4(Itr inout, std::size_t size, T t, T dt, const DiscreteOp& op, const Axpy& axpy, Itr wk0, Itr wk1, Itr wk2, Itr wk3, Itr wk4)
{
  op(inout, size, t, wk1);

  axpy((T)(0.5L) * dt, wk1, size, inout, wk0);
  op(wk0, size, t + (T)(0.5L) * dt, wk2);

  axpy((T)(0.5L) * dt, wk2, size, inout, wk0);
  op(wk0, size, t + (T)(0.5L) * dt, wk3);

  axpy(dt, wk3, size, inout, wk0);
  op(wk0, size, t + dt, wk4);

  axpy((T)(2.0L), wk2, size, wk1, wk0);
  axpy((T)(2.0L), wk3, size, wk4, wk1);
  axpy(dt / (T)(6.0L), wk0, size, inout, wk2);
  axpy(dt / (T)(6.0L), wk1, size, wk2, inout);
}

END_NAMESPACE

#endif
