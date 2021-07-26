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

#ifndef ENCODED_INTEGER_H
#define ENCODED_INTEGER_H

#include <type_traits>
#include <cassert>

#include "config.h"

BEGIN_NAMESPACE

// encode an integer type I (signed or unsigned) with N least significant bits to come up with
// a "combined" integer
//
// NOTE: this essentially reduces the valid range of I by N bits, e.g., the original value in I
// NOTE: before the endcoding must < 2^(sizeof(I) * 8 - N), if I is an unsigned integer type
template<typename I, int N>
struct encoded_integer
{
  static_assert(std::is_integral<I>::value, "I must be an integer type");
  static_assert(N > 0, "N must be positive");
  static_assert(N < sizeof(I) * 8 - 1, "N too large");

  encoded_integer() : _int_rep(0) {}

  encoded_integer(I integer, I code) : _int_rep(integer << N | code)
  {
    assert(code >= 0 && code < (static_cast<I>(1) << N));
    if constexpr(std::is_signed<I>::value)
      assert(integer < (static_cast<I>(1) << (sizeof(I) * 8 - N - 1)) &&
             integer >= -(static_cast<I>(1) << (sizeof(I) * 8 - N - 1)));
    else
      assert(integer < (static_cast<I>(1) << (sizeof(I) * 8 - N)));
  }

  I integer() const { return _int_rep >> N; }

  I code() const { return _int_rep & ((static_cast<I>(1) << N) - 1); }

  I integer_representation() const { return _int_rep; }

private:
  I _int_rep;
};

END_NAMESPACE

#endif
