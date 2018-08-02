// Copyright (c) 2018 NVIDIA Corporation
// Reply-To: Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#if !defined(CUDA_CONFIG_HPP)
#define CUDA_CONFIG_HPP

#include <cuda/preprocessor.hpp>

///////////////////////////////////////////////////////////////////////////////

#if 201103L > __cplusplus
  // /SO/ last decade, friend. Sorry.
  #error The CUDA C++ core library requires C++11.
#endif

///////////////////////////////////////////////////////////////////////////////

#define _CUDA_CPP_LIB_ABI_VERSION 0

#define _CUDA_CPP_LIB_ABI_NAMESPACE _CAT2(__, _CUDA_CPP_LIB_ABI_VERSION)

#define _BEGIN_CUDA_NAMESPACE                                               \
  namespace cuda { inline namespace _CUDA_CPP_LIB_ABI_NAMESPACE {           \
  /**/

#define _END_CUDA_NAMESPACE }}

#define _CUDA cuda::_CUDA_CPP_LIB_ABI_NAMESPACE

///////////////////////////////////////////////////////////////////////////////

#if !defined(__CUDACC__)
  #define _HOST
  #define _DEVICE
#else
  #define _HOST   __host__
  #define _DEVICE __device__
#endif

///////////////////////////////////////////////////////////////////////////////

#endif // CUDA_CONFIG_HPP
