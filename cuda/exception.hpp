// Copyright (c) 2018 NVIDIA Corporation
// Reply-To: Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

// TODO: This should probably be split into two headers so that driver users
// don't have to include `<cuda_runtime.h>` and runtime users don't have to
// include `<cuda.h>`.

#if !defined(CUDA_EXCEPTION_HPP)
#define CUDA_EXCEPTION_HPP

#include <cuda/config.hpp>

#include <string>
#include <exception>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda/preprocessor.hpp>

_BEGIN_CUDA_NAMESPACE

///////////////////////////////////////////////////////////////////////////////

struct exception : std::exception
{
  _HOST
  virtual ~exception() noexcept {}

  _HOST
  virtual char const* what() const noexcept = 0;
};

struct runtime_exception : _CUDA::exception
{
  _HOST
  runtime_exception(cudaError_t error, char const* message)
    : error_(error)
    , message_(
        std::string(cudaGetErrorName(error)) + ": "
      + cudaGetErrorString(error) + ": "
      + message
      )
  {}

  _HOST
  virtual ~runtime_exception() noexcept {}

  _HOST
  cudaError_t code() const noexcept
  {
    return error_;
  }

  _HOST
  virtual char const* what() const noexcept
  {
    return message_.c_str();
  }

private:
  cudaError_t error_;
  std::string message_;
};

struct driver_exception : _CUDA::exception 
{
  _HOST
  driver_exception(CUresult error, char const* message)
    : error_(error)
  {
    char const* str = nullptr;
    cuGetErrorName(error, &str);
    message_ = str;
    message_ += ": ";
    cuGetErrorString(error, &str);
    message_ += str;
    message_ += ": ";
    message_ += message;
  }

  _HOST
  virtual ~driver_exception() noexcept {}

  _HOST
  CUresult code() const noexcept
  {
    return error_;
  }

  _HOST
  virtual char const* what() const noexcept
  {
    return message_.c_str();
  }

private:
  CUresult    error_;
  std::string message_;
};

namespace detail {

_HOST
inline void throw_on_error(cudaError_t error, char const* message)
{
  if (cudaSuccess != error)
    throw _CUDA::runtime_exception(error, message);
}

_HOST
inline void throw_on_error(CUresult error, char const* message)
{
  if (CUDA_SUCCESS != error)
    throw _CUDA::driver_exception(error, message);
}

} // namespace detail

#define CUDA_THROW_ON_ERROR(error)                                            \
  _CUDA::detail::throw_on_error(error, _CURRENT_FUNCTION)                     \
  /**/

///////////////////////////////////////////////////////////////////////////////

_END_CUDA_NAMESPACE

#endif // CUDA_EXCEPTION_HPP

