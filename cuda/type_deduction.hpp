// Copyright (c)      2018 NVIDIA Corporation
// Copyright (c) 2013-2018 Eric Niebler (`_RETURNS*`)
// Copyright (c) 2016-2018 Casey Carter (`_RETURNS*`)
//
// Reply-To: Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#if !defined(CUDA_TYPE_DEDUCTION_HPP)
#define CUDA_TYPE_DEDUCTION_HPP

#include <utility>

#include <cuda/preprocessor.hpp>

///////////////////////////////////////////////////////////////////////////////

/// \def _FWD(x)
/// \brief Performs universal forwarding of a universal reference.
///
#define _FWD(x) ::std::forward<decltype(x)>(x)

/// \def _MV(x)
/// \brief Moves `x`.
///
#define _MV(x) ::std::move(x)

/// \def _MVCAP(x)
/// \brief Capture `x` into a lambda by moving.
///
#define _MVCAP(x) x = _MV(x)

/// \def _RETOF(invocable, ...)
/// \brief Expands to the type returned by invoking an instance of the invocable
///        type \a invocable with parameters of type <tt>__VA_ARGS__</tt>.
///
#define _RETOF(...)   _DISPATCH(_RETOF, __VA_ARGS__)
#define _RETOF1(C)    decltype(::std::declval<C>()())
#define _RETOF2(C, V) decltype(::std::declval<C>()(::std::declval<V>()))

/// \def _RETURNS(...)
/// \brief Expands to a function definition that returns the expression
///        <tt>__VA_ARGS__</tt>.
///
#define _RETURNS(...)                                                         \
  noexcept(noexcept(decltype(__VA_ARGS__)(__VA_ARGS__)))                      \
  { return (__VA_ARGS__); }                                                   \
  /**/

/// \def _AUTORETURNS(...)
/// \brief Expands to a function definition, including a trailing returning
///        type, that returns the expression <tt>__VA_ARGS__</tt>.
///
#define _AUTORETURNS(...)                                                     \
  noexcept(noexcept(decltype(__VA_ARGS__)(__VA_ARGS__)))                      \
  -> decltype(__VA_ARGS__)                                                    \
  { return (__VA_ARGS__); }                                                   \
  /**/

/// \def _AUTOQUALRETURNS(qual...)
/// \brief Expands to a function definition, including a trailing returning
///        type and a qualifier, that returns the expression
///        <tt>__VA_ARGS__</tt>.
///
#define _AUTOQUALRETURNS(qualifier, ...)                                      \
  qualifier                                                                   \
  noexcept(noexcept(decltype(__VA_ARGS__)(__VA_ARGS__)))                      \
  -> decltype(__VA_ARGS__)                                                    \
  { return (__VA_ARGS__); }                                                   \
  /**/

///////////////////////////////////////////////////////////////////////////////

#endif // CUDA_TYPE_DEDUCTION_HPP

