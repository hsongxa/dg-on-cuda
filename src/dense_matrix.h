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

#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

#include <memory>
#include <cstddef>
#include <type_traits>
#include <initializer_list>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <iostream>

#include "config.h"

BEGIN_NAMESPACE

// This serves as the base class of dense_matrix. It handles allocation/deallocation and
// ownership of the continuous memory of size n * sizeof(T). Similar to std::vector_base,
// this class uses EBCO to minimize its size and RAII to make exception handling easier
// for derived classes. But different from std::vector_base which manages memory of
// n-capacity-but-m-storage (m <= n), this class manages memory of n-capacity-and-n-storage.
//
// Note that this class can also be used as the base class of other types that have a same
// memory management pattern.
template<typename T, typename Alloc>
class n_storage
{
protected:
  using allocator_type   = Alloc;
  using allocator_traits = std::allocator_traits<allocator_type>;
  using pointer          = typename allocator_traits::pointer;
  using const_pointer    = typename allocator_traits::const_pointer;
  using size_type        = std::size_t;

private:
  struct allocatorage : public allocator_type
  {  
    pointer m_start;
    size_type m_size;

    allocatorage() noexcept(std::is_nothrow_default_constructible_v<allocator_type>)
    : allocator_type(), m_start(), m_size() {}

    explicit allocatorage(const allocator_type& a) noexcept(std::is_nothrow_copy_constructible_v<allocator_type>)
    : allocator_type(a), m_start(), m_size() {}

    allocatorage(allocatorage&& x) noexcept(std::is_nothrow_move_constructible_v<allocator_type>)
    : allocator_type(std::move(x)), m_start(x.m_start), m_size(x.m_size)
    { x.m_start = pointer(); x.m_size = size_type(); }

    allocatorage(allocator_type&& a) noexcept(std::is_nothrow_move_constructible_v<allocator_type>)
    : allocator_type(std::move(a)) {}

    allocatorage(allocator_type& a, allocatorage&& x) noexcept(std::is_nothrow_copy_constructible_v<allocator_type>)
    : allocator_type(a), m_start(x.m_start), m_size(x.m_size)
    { x.m_start = pointer(); x.m_size = size_type(); }

    allocator_type& allocator()
    { return static_cast<allocator_type&>(*this); }

    const allocator_type& allocator() const
    { return static_cast<const allocator_type&>(*this); }

    void swap(allocatorage& x) noexcept(std::is_nothrow_swappable_v<allocator_type>)
    {
      std::swap(allocator(), x.allocator());
      std::swap(m_start, x.m_start);
      std::swap(m_size, x.m_size);
    }
  };

protected:
  // constructors
  n_storage() noexcept(std::is_nothrow_default_constructible_v<allocator_type>)
  : m_allocatorage() {}

  explicit n_storage(const allocator_type& a) noexcept(std::is_nothrow_copy_constructible_v<allocator_type>)
  : m_allocatorage(a) {}

  explicit n_storage(size_type n)
  : m_allocatorage() { create_storage(n); }

  n_storage(size_type n, const allocator_type& a)
  : m_allocatorage(a) { create_storage(n); }

  // move constructors
  n_storage(n_storage&& x) noexcept(std::is_nothrow_move_constructible_v<allocator_type>)
  : m_allocatorage(std::move(x.m_allocatorage)) {}

  n_storage(allocator_type&& a) noexcept(std::is_nothrow_move_constructible_v<allocator_type>)
  : m_allocatorage(std::move(a)) {}

  n_storage(n_storage&& x, const allocator_type& a) noexcept(std::is_nothrow_copy_constructible_v<allocator_type>)
  : m_allocatorage(a, std::move(x.m_allocatorage)) {}

  // copy assignment
  void copy_assign(const n_storage& other)
  {
    bool reallocate = (get_allocator() != other.get_allocator() || m_allocatorage.m_size != other.m_allocatorage.m_size);
    if (reallocate) destroy_storage();
    copy_assign_allocator(other, std::bool_constant<allocator_traits::propagate_on_container_copy_assignment::value>());
    if (reallocate) create_storage(other.m_allocatorage.m_size);  
  }

  // move assignment
  void move_assign(n_storage& other) noexcept(std::is_nothrow_move_assignable_v<allocator_type>)
  { move_assign_impl(other, std::bool_constant<allocator_traits::propagate_on_container_move_assignment::value>()); }

  // swap
  void swap(n_storage& other) noexcept(std::is_nothrow_swappable_v<allocator_type> ||
                                       !allocator_traits::propagate_on_container_swap::value)
  { swap_impl(other, std::bool_constant<allocator_traits::propagate_on_container_swap::value>()); }

  // destructor
  ~n_storage() noexcept { destroy_storage(); }

  // get the allocator
  const allocator_type& get_allocator() const noexcept { return m_allocatorage.allocator(); }

  pointer start() { return m_allocatorage.m_start; }

  const_pointer start() const { return m_allocatorage.m_start; }

  size_type size() const { return m_allocatorage.m_size; }

private:
  pointer allocate(size_type n)
  { return n != 0 ? allocator_traits::allocate(m_allocatorage, n) : pointer(); }

  void deallocate(pointer p, size_type n)
  { if(p) allocator_traits::deallocate(m_allocatorage, p, n); }

  void create_storage(size_type n)
  { m_allocatorage.m_start = allocate(n); m_allocatorage.m_size = n; }

  void destroy_storage()
  { deallocate(m_allocatorage.m_start, m_allocatorage.m_size); }

  void copy_assign_allocator(const n_storage& other, std::true_type) noexcept(std::is_nothrow_copy_assignable_v<allocator_type>)
  {
    // copy assignment of allocator
    m_allocatorage.allocator() = other.m_allocatorage.allocator();
  }

  void copy_assign_allocator(const n_storage& other, std::false_type) noexcept
  {}

  void move_assign_impl(n_storage& other, std::true_type) noexcept(std::is_nothrow_move_assignable_v<allocator_type>)
  {
    destroy_storage();

    // move assignment of allocator
    m_allocatorage.allocator() = std::move(other.m_allocatorage.allocator());

    m_allocatorage.m_start = other.m_allocatorage.m_start;
    m_allocatorage.m_size = other.m_allocatorage.m_size;
    other.m_allocatorage.m_start = pointer();
    other.m_allocatorage.m_size = size_type();
  }

  void move_assign_impl(n_storage& other, std::false_type) noexcept(std::is_nothrow_move_assignable_v<allocator_type>)
  {
    if (get_allocator() != other.get_allocator())
    {
      // the allocator cannot be moved
      destroy_storage();
      m_allocatorage.m_start = other.m_allocatorage.m_start;
      m_allocatorage.m_size = other.m_allocatorage.m_size;
      other.m_allocatorage.m_start = pointer();
      other.m_allocatorage.m_size = size_type();
    }
    else
      move_assign_impl(other, std::true_type());
  }

  void swap_impl(n_storage& other, std::true_type) noexcept(std::is_nothrow_swappable_v<allocator_type>)
  { m_allocatorage.swap(other.m_allocatorage); }

  void swap_impl(n_storage& other, std::false_type) noexcept
  {
    if (get_allocator() == other.get_allocator())
    {
      std::swap(m_allocatorage.m_start, other.m_allocatorage.m_start);
      std::swap(m_allocatorage.m_size, other.m_allocatorage.m_size);
    }
    // otherwise, undefined behavior
  }

private:
  allocatorage m_allocatorage;
};

// T: the number type; CM: column major storage when true, otherwise row major.
//
// Note that the resize() function will always result in memory re-allocation
// if the new size is different from the current size, due to the n_storage
// memory management used.
template<typename T, bool CM = false, typename Alloc = std::allocator<T>>
class dense_matrix : private n_storage<T, Alloc>
{
public:
  using Base             = n_storage<T, Alloc>;
  using allocator_type   = typename Base::allocator_type;
  using allocator_traits = typename Base::allocator_traits;

  using value_type      = T;
  using pointer         = typename Base::pointer;
  using const_pointer   = typename Base::const_pointer;
  using reference       = value_type&;
  using const_reference = const value_type&;
  using size_type       = typename Base::size_type;

  using init_list = std::initializer_list<std::initializer_list<value_type>>;

public:
  // constructors
  dense_matrix() noexcept(std::is_nothrow_default_constructible_v<allocator_type>)
  : Base(), m_stride() {}

  explicit dense_matrix(const allocator_type& a) noexcept(std::is_nothrow_copy_constructible_v<allocator_type>)
  : Base(a), m_stride() {}
  
  dense_matrix(size_type size_row, size_type size_col, const allocator_type& a = allocator_type())
  : Base(size_row * size_col, a)
  { set_stride(size_row, size_col); std::uninitialized_default_construct_n(Base::start(), Base::size()); }

  dense_matrix(size_type size_row, size_type size_col, const value_type& val, const allocator_type& a = allocator_type())
  : Base(size_row * size_col, a)
  { set_stride(size_row, size_col); std::uninitialized_fill_n(Base::start(), Base::size(), val); }

  dense_matrix(init_list initl)
  : Base(initl.size() * initl.begin()->size())
  {
    auto list_itr = initl.begin();
    m_stride = list_itr->size();
    for(size_type i = 0; i < initl.size(); ++i, ++list_itr)
    {
      assert(list_itr->size() == m_stride);
      std::uninitialized_copy_n(list_itr->begin(), m_stride, Base::start() + i * m_stride);
    }
  }
  
  // copy constructors
  dense_matrix(const dense_matrix& other)
  : Base(other.size(), allocator_traits::select_on_container_copy_construction(other.get_allocator()))
  {
    std::uninitialized_copy_n(other.start(), Base::size(), Base::start());
    m_stride = other.m_stride;
  }

  dense_matrix(const dense_matrix& other, const allocator_type& a)
  : Base(other.m_allocatorage.m_size, a)
  {
    std::uninitialized_copy_n(other.start(), Base::size(), Base::start());
    m_stride = other.stride;
  }

  // move constructors
  dense_matrix(dense_matrix&& other) noexcept(std::is_nothrow_move_constructible_v<allocator_type>)
  : Base(std::move(other))
  { m_stride = other.m_stride; other.m_stride = size_type(); }

  dense_matrix(dense_matrix&& other, const allocator_type& a) noexcept(std::is_nothrow_copy_constructible_v<allocator_type> &&
                                                                       std::is_nothrow_move_constructible_v<allocator_type>)
  {
    if (other.get_allocator() == a) Base(std::move(other));
    else Base(std::move(other), a);
    m_stride = other.m_stride;
    other.m_stride = size_type();
  }

  // copy assignment
  dense_matrix& operator=(const dense_matrix& other)
  {
    if (this != &other)
    {
      Base::copy_assign(other);
      std::uninitialized_copy_n(other.start(), Base::size(), Base::start());
      m_stride = other.m_stride;
    }

    return *this;
  }

  // move assignment
  dense_matrix& operator=(dense_matrix&& other) noexcept(std::is_nothrow_move_assignable_v<allocator_type>)
  {
    Base::move_assign(other);
    m_stride = other.m_stride;
    other.m_stride = size_type();
    return *this;
  }
  
  // assignment from initialization list 
  dense_matrix& operator=(init_list initl)
  {
    dense_matrix tmp(initl);
    return this->operator=(std::move(tmp));
  }
  
  // swap
  void swap(dense_matrix& other) noexcept(std::is_nothrow_swappable_v<allocator_type> ||
                                          !allocator_traits::propagate_on_container_swap::value)
  { Base::swap(other); std::swap(m_stride, other.m_stride); }

  // destructor
  ~dense_matrix() = default;

  T* data() { return to_address(Base::start()); } // TODO: replace with std::to_address(Base::start()) in c++20

  const T* data() const { return to_address(Base::start()); } // TODO: replace with std::to_address(Base::start()) in c++20

  size_type size_row() const
  {
    if constexpr(CM) return m_stride;
    else return m_stride == 0 ? 0 : Base::size() / m_stride;
  }
  
  size_type size_col() const
  {
    if constexpr(CM) return m_stride == 0 ? 0 : Base::size() / m_stride;
    else return m_stride;
  }
  
  void resize(size_type row, size_type col)
  {
    if (row * col != Base::size())
    {
      dense_matrix tmp(row, col);
      this->operator=(std::move(tmp));
    }
  }

  reference operator()(size_type i, size_type j)
  {
    if constexpr(CM) return *(Base::start() + j * m_stride + i);
    else return *(Base::start() + i * m_stride + j);
  }

  const_reference operator()(size_type i, size_type j) const
  {
    if constexpr(CM) return *(Base::start() + j * m_stride + i);
    else return *(Base::start() + i * m_stride + j);
  }

  dense_matrix transpose() const;
  
  dense_matrix inverse() const;

  value_type determinant() const { return determinant(*this); }

  template<typename InputItr, typename InOutItr>
  void gemv(value_type alpha, InputItr in_first, value_type beta, InOutItr inout_first) const;

  // TODO: implement expression template
  // NOTE: these friend functions rely on ADL to be found as they are non-template non-member functions
  friend dense_matrix operator*(value_type scalar, const dense_matrix& matrix)
  {
    dense_matrix result(matrix.size_row(), matrix.size_col());
    std::transform(matrix.start(), matrix.start() + matrix.size(), result.start(),
                   [scalar](const_reference v) { return v * scalar; });
    return result;
  }

  friend dense_matrix operator*(const dense_matrix& matrix, value_type scalar)
  { return scalar * matrix; }

  friend dense_matrix operator*(const dense_matrix& m1, const dense_matrix& m2)
  {
    assert(m1.size_col() == m2.size_row());

    dense_matrix prod(m1.size_row(), m2.size_col());
    pointer d = prod.start();

    if constexpr(CM)
    {
      dense_matrix tmp = m1.transpose();
      for (const_pointer p = tmp.start(); p < tmp.start() + tmp.size(); p += tmp.m_stride)
        for (const_pointer q = m2.start(); q < m2.start() + m2.size(); q += m2.m_stride)
          *d++ = std::inner_product(p, p + tmp.m_stride, q, value_type());
    }
    else
    {
      dense_matrix tmp = m2.transpose();
      for (const_pointer p = m1.start(); p < m1.start() + m1.size(); p += m1.m_stride)
        for (const_pointer q = tmp.start(); q < tmp.start() + tmp.size(); q += tmp.m_stride)
          *d++ = std::inner_product(p, p + m1.m_stride, q, value_type());
    }

    return prod;
  }

  friend dense_matrix operator+(const dense_matrix& m1, const dense_matrix& m2)
  {
    assert(m1.size_row() == m2.size_row() && m1.size_col() == m2.size_col());

    dense_matrix result(m1.size_row(), m1.size_col());
    std::transform(m1.start(), m1.start() + m1.size(), m2.start(), result.start(),
                   [](const_reference v1, const_reference v2) { return v1 + v2; });
    return result;
  }

private:
  void set_stride(size_type size_row, size_type size_col)
  {
    if (size_row == 0 || size_col == 0)
    {
      assert(Base::size() == 0);
      m_stride = 0;
      return;
    }

    if constexpr(CM) m_stride = size_row;
    else m_stride = size_col;
  }

  // TODO: remove the following 4 functions in c++20
  template<typename Ptr>
  value_type* to_address(Ptr pointer) { return pointer.operator->(); }

  template<typename Ptr>
  const value_type* to_address(Ptr pointer) const { return pointer.operator->(); }

  template<typename P>
  P* to_address(P* p) { return p; }

  template<typename P>
  const P* to_address(P* p) const { return p; }

  static void fill_cofactors(const dense_matrix& m, dense_matrix& cofactor, size_type i, size_type j);

  static value_type determinant(const dense_matrix& m);

private:
  size_type m_stride;
};

template<typename T, bool CM, typename Alloc>
dense_matrix<T, CM, Alloc> dense_matrix<T, CM, Alloc>::transpose() const
{
  dense_matrix trans(size_col(), size_row());
  pointer d = trans.start();
  for (const_pointer p = Base::start(); p < Base::start() + m_stride; ++p)
    for (const_pointer q = p; q < Base::start() + Base::size(); q += m_stride)
      *d++ = *q;
  return trans;
}
  
template<typename T, bool CM, typename Alloc>
dense_matrix<T, CM, Alloc> dense_matrix<T, CM, Alloc>::inverse() const
{
  assert(size_row() == size_col());

  size_type size = size_row();
  dense_matrix inv(size, size);
  if (size == 0) return inv;

  // find adjoint
  value_type one = static_cast<value_type>(1.0L); // TODO: properly retrieve one from type
  if (size == 1)
    inv(0, 0) = one;
  else
  {
    dense_matrix tmp(size - 1, size - 1);
    for (size_type i = 0; i < size; ++i)
      for (size_type j = 0; j < size; ++j)
      {
        fill_cofactors(*this, tmp, i, j);
        if ((i + j) % 2 == 0) inv(j, i) = determinant(tmp);
        else inv(j, i) = - determinant(tmp);
      }
  }

  return (one / determinant(*this)) * inv;
}

template<typename T, bool CM, typename Alloc> template<typename InputItr, typename InOutItr>
void dense_matrix<T, CM, Alloc>::gemv(value_type alpha, InputItr in_first, value_type beta, InOutItr inout_first) const
{
  assert(in_first != inout_first);
  if constexpr(CM)
  {
    for (const_pointer p = Base::start(); p < Base::start() + m_stride; ++p)  
    {
      value_type y = beta * (*inout_first);
      InputItr x = in_first; 
      for (const_pointer q = p; q < Base::start() + Base::size(); ++x, q += m_stride)
        y += alpha * (*q) * (*x);
      *inout_first++ = y;
    }
  }
  else
  {
    for (const_pointer p = Base::start(); p < Base::start() + Base::size(); p += m_stride)  
    {
      value_type y = beta * (*inout_first);
      InputItr x = in_first; 
      for (const_pointer q = p; q < p + m_stride; ++x, ++q)
        y += alpha * (*q) * (*x);
      *inout_first++ = y;
    }
  }
}

template<typename T, bool CM, typename Alloc>
void dense_matrix<T, CM, Alloc>::fill_cofactors(const dense_matrix& m, dense_matrix& cofactor, size_type i, size_type j)
{
  assert(cofactor.size_row() == m.size_row() - 1);
  assert(cofactor.size_col() == m.size_col() - 1);

  if constexpr(CM)
  {
    pointer d = cofactor.start();
    size_type col = 0;
    for (const_pointer p = m.start(); p < m.start() + m.size(); ++col, p += m.m_stride)  
    {
      if (col != j)
      {
        size_type row = 0;
        for (const_pointer q = p; q < p + m.m_stride; ++row, ++q)
          if (row != i) *d++ = *q;
      }
    }
  }
  else
  {
    pointer d = cofactor.start();
    size_type row = 0;
    for (const_pointer p = m.start(); p < m.start() + m.size(); ++row, p += m.m_stride)  
    {
      if (row != i)
      {
        size_type col = 0;
        for (const_pointer q = p; q < p + m.m_stride; ++col, ++q)
          if (col != j) *d++ = *q;
      }
    }
  }
}

template<typename T, bool CM, typename Alloc>
typename dense_matrix<T, CM, Alloc>::value_type dense_matrix<T, CM, Alloc>::determinant(const dense_matrix& m)
{
  assert(m.size_row() == m.size_col());

  value_type d{}; // zero 
  if (m.size_row() == 0) return d;
  if (m.size_row() == 1) return m(0, 0);

  size_type size = m.size_row();
  dense_matrix tmp(size - 1, size - 1);

  bool sign = true;
  for (size_type f = 0; f < size; ++f)
  {
    fill_cofactors(m, tmp, 0, f);
    if (sign) d += m(0, f) * determinant(tmp);
    else d -= m(0, f) * determinant(tmp);
    sign = (!sign);
  }

  return d;
}

template<typename T, bool CM, typename Alloc>
std::ostream& operator<<(std::ostream& out, const dense_matrix<T, CM, Alloc>& m)
{
  using size_type = typename dense_matrix<T, CM, Alloc>::size_type;

  out << '{';
  for (size_type i = 0; i < m.size_row(); ++i)
  {
    out << '{';
    for (size_type j = 0; j < m.size_col(); ++j)
    {
      out << m(i, j);
      if (j < m.size_col() - 1) out << ", ";
    }
    out << '}';
    if (i < m.size_row() - 1) out << ',' << std::endl << ' ';
  }
  return out << '}' << std::endl;
}

END_NAMESPACE

#endif
