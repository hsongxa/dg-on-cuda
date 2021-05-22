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

#ifndef QUADRATURE_RULES_H
#define QUADRATURE_RULES_H

#include <iterator>
#include <memory>

#include <cmath>
#include <cassert>

#include "config.h"

BEGIN_NAMESPACE

// see https://stackoverflow.com/questions/29065760/traits-class-to-extract-containers-value-type-from-a-back-insert-iterator
template<class T>
struct output_iterator_traits : std::iterator_traits<T> {};

template< class OutputIt, class T>
struct output_iterator_traits<std::raw_storage_iterator<OutputIt, T>>
: std::iterator<std::output_iterator_tag, T> {};

template<class Container>
struct output_iterator_traits<std::back_insert_iterator<Container>>
: std::iterator<std::output_iterator_tag, typename Container::value_type> {};

template<class Container>
struct output_iterator_traits<std::front_insert_iterator<Container>>
: std::iterator<std::output_iterator_tag, typename Container::value_type> {};

template<class Container>
struct output_iterator_traits<std::insert_iterator<Container>>
: std::iterator<std::output_iterator_tag, typename Container::value_type> {};

template <class T, class charT, class traits>
struct output_iterator_traits<std::ostream_iterator<T, charT, traits>>
: std::iterator<std::output_iterator_tag, T> {}; 

template <class charT, class traits>
struct output_iterator_traits<std::ostreambuf_iterator<charT, traits>>
: std::iterator<std::output_iterator_tag, charT> {};




// TODO: implement the Golub-Welsch algorithm to cover arbitrary order
template<typename OutputIteratorP, typename OutputIteratorW>
void gauss_lobatto_quadrature(std::size_t npts, OutputIteratorP it_p, OutputIteratorW it_w)
{
  assert(npts >= 2);

  using P = typename output_iterator_traits<OutputIteratorP>::value_type;
  using W = typename output_iterator_traits<OutputIteratorW>::value_type;

  switch (npts)
  {
    case 2:
      it_p = - P(1);
      it_w = W(1);
      it_p = P(1);
      it_w = W(1);
      return;
    case 3:
      it_p = - P(1);
      it_w = W(1) / W(3);
      it_p = P{}; // zero
      it_w = W(4) / W(3);
      it_p = P(1);
      it_w = W(1) / W(3);
      return;
    case 4:
      it_p = - P(1);
      it_w = W(1) / W(6);
      it_p = - std::sqrt(P(1) / P(5));
      it_w = W(5) / W(6);
      it_p = std::sqrt(P(1) / P(5));
      it_w = W(5) / W(6);
      it_p = P(1);
      it_w = W(1) / W(6);
      return;
    case 5:
      it_p = - P(1);
      it_w = W(1) / W(10);
      it_p = - std::sqrt(P(3) / P(7));
      it_w = W(49) / W(90);
      it_p = P{}; // zero
      it_w = W(32) / W(45);
      it_p = std::sqrt(P(3) / P(7));
      it_w = W(49) / W(90);
      it_p = P(1);
      it_w = W(1) / W(10);
      return;
    case 6:
      it_p = - P(1);
      it_w = W(1) / W(15);
      it_p = - std::sqrt(P(1) / P(3) + P(2) * std::sqrt(P(7)) / P(21));
      it_w = (W(14) - std::sqrt(W(7))) / W(30);
      it_p = - std::sqrt(P(1) / P(3) - P(2) * std::sqrt(P(7)) / P(21));
      it_w = (W(14) + std::sqrt(W(7))) / W(30);
      it_p = std::sqrt(P(1) / P(3) - P(2) * std::sqrt(P(7)) / P(21));
      it_w = (W(14) + std::sqrt(W(7))) / W(30);
      it_p = std::sqrt(P(1) / P(3) + P(2) * std::sqrt(P(7)) / P(21));
      it_w = (W(14) - std::sqrt(W(7))) / W(30);
      it_p = P(1);
      it_w = W(1) / W(15);
      return;
    case 7:
      it_p = - P(1);
      it_w = W(1) / W(21);
      it_p = - std::sqrt(P(5) / P(11) + P(2) * std::sqrt(P(5) / P(3)) / P(11));
      it_w = (W(124) - W(7) * std::sqrt(W(15))) / W(350);
      it_p = - std::sqrt(P(5) / P(11) - P(2) * std::sqrt(P(5) / P(3)) / P(11));
      it_w = (W(124) + W(7) * std::sqrt(W(15))) / W(350);
      it_p = P{}; // zero
      it_w = W(256) / W(525);
      it_p = std::sqrt(P(5) / P(11) - P(2) * std::sqrt(P(5) / P(3)) / P(11));
      it_w = (W(124) + W(7) * std::sqrt(W(15))) / W(350);
      it_p = std::sqrt(P(5) / P(11) + P(2) * std::sqrt(P(5) / P(3)) / P(11));
      it_w = (W(124) - W(7) * std::sqrt(W(15))) / W(350);
      it_p = P(1);
      it_w = W(1) / W(21);
      return;
    default:
      throw "arbitrary order gauss-lobatto quadrature is not implemented yet!";
  }
}

END_NAMESPACE

#endif
