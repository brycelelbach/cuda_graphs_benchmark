// Copyright 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

// WARNING: Not for the faint of heart.

// TODO: Convert to ranges.
// TODO: Refactor CUDA smart pointer handlers.

/*

///////////////////////////////////////////////////////////////////////////////
Hypothesis
///////////////////////////////////////////////////////////////////////////////

~x denotes base 10 order of magnitude. Ex: ~1 means between 1 and 9, ~10 means
between 10 and 99.

This study is limited to sets of linearly dependent kernels for the time being.

- Graph launch has an upfront cost called graph compilation. This cost can be
  amortized by reusing a compiled graph.

- Compiling a graph of N kernels has:
  - O(N) time complexity if the kernels are linearly dependent.
  - At worst O(N^2) time complexity otherwise.

- Launching N kernels has O(N) time complexity for both traditional launches
  and graph launches.

- Thus, compiling and launching a graph of N linearly dependent kernels has O(N)
  time complexity.

- Conceptually, graph launch should reduce the time complexity of launching N
  kernels from O(N) to O(1). In practice, graph launch should at least reduce
  the time complexity of launching N kernels from O(CN) to O(cN), where c << C.

- Launching AmortizeLaunch or more linearly dependent kernels once takes less
  time with graph launch than traditional launch, where AmortizeLaunch is ~1.

- Compiling and launching AmortizeCompileLaunch or more linearly dependent
  kernels once takes less time with graph launch than traditional launch, where
  AmortizeCompileLaunch is ~1000.

- Compiling and launching N linearly dependent kernels AmortizeReuse or more
  times takes less time with graph launch than traditional launch, where
  AmortizeReuse is decreases as N increases and AmortizeReuse is ~100 when
  N is ~1.

///////////////////////////////////////////////////////////////////////////////
Predictions
///////////////////////////////////////////////////////////////////////////////

Launch x linearly dependent kernels traditionally once:

  y = Launch * x

Compile x linearly dependent kernels:

  y = CompileGraph * x + CompileGraphFixed

Launch x linearly dependent kernels with graphs once:

  y = LaunchGraph * x + LaunchGraphFixed

Compile and launch x linearly dependent kernels with graphs once:

  y = CompileGraph * x + LaunchGraph * x + CompileGraphFixed + LaunchGraphFixed
    = (CompileGraph + LaunchGraph) * x + (CompileGraphFixed + LaunchGraphFixed)

Launch N linearly dependent kernels traditionally x times:

  y = Launch * N * x
    = (Launch * N) * x

Compile and launch N linearly dependent kernels with graphs x times:

  y = LaunchGraph * N * x + LaunchGraphFixed * x + CompileGraph * N + CompileGraphFixed
    = (LaunchGraph * N + LaunchGraphFixed) * x + (CompileGraph * N + CompileGraphFixed)

Amortization of launch:

  Launch * AmortizeLaunch = LaunchGraph * AmortizeLaunch + LaunchGraphFixed
  AmortizeLaunch \in [10^0, 10^1)

Amortization of graph compilation and launch:

  Launch * AmortizeCompileLaunch = (CompileGraph + LaunchGraph) * AmortizeCompileLaunch + (CompileGraphFixed + LaunchGraphFixed)
  AmortizeCompileLaunch \in [10^3, 10^4)

Amortization of graph reuse:

  (Launch * N) * AmortizeReuse(N) = (LaunchGraph * N + LaunchGraphFixed) * AmortizeReuse(N) + (CompileGraph * N + CompileGraphFixed)
  AmortizeReuse(2N) < AmortizeReuse(N)
  AmortizeReuse(1) \in [10^2, 10^3)

///////////////////////////////////////////////////////////////////////////////
Testing
///////////////////////////////////////////////////////////////////////////////

* Measure execution time of traditional launch of linearly dependent kernels.

* Measure execution time of graph launch of linearly dependent kernels while varying:
  * # of kernels per graph.
  * # of launches per graph.

* Measure compilation time of graphs of linearly dependent kernels while varying:
  * # of kernels per graph.

*/

#include <utility>
#include <memory>
#include <functional>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <vector>
#include <map>
#include <string>
#include <exception>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cassert>

#include <cuda.h>

///////////////////////////////////////////////////////////////////////////////

/// \def PP_CAT2(a, b)
/// \brief Concatenates the tokens \a a and \b b.
///
/// \par <b>Example</b>:
///
/// \code
/// #include <iostream>
/// 
/// int main()
/// {
///   std::cout << PP_CAT2(1, PP_CAT2(2, 3)) << std::endl;
/// }
/// \endcode
///
/// The above code expands to:
///
/// \code
/// #include <iostream>
/// 
/// int main()
/// {
///   std::cout << 123 << std::endl;
/// }
/// \endcode
///
#define PP_CAT2(a, b) PP_CAT2_IMPL(a, b)

#if    defined(_MSC_VER)                                                      \
    && (defined(__EDG__) || defined(__EDG_VERSION__))                         \
    && (defined(__INTELLISENSE__) || __EDG_VERSION__ >= 308)
    #define PP_CAT2_IMPL(a, b) PP_CAT2_IMPL2(~, a ## b)
    #define PP_CAT2_IMPL2(p, res) res
#else
    #define PP_CAT2_IMPL(a, b) a ## b
#endif

///////////////////////////////////////////////////////////////////////////////

/// \def PP_EXPAND(x)
/// \brief Performs macro expansion on \a x. 
///
/// \par <b>Example</b>:
///
/// \code
/// #include <iostream>
///
/// #define FOO_BAR() "foo_bar"
/// #define BUZZ()     PP_EXPAND(PP_CAT2(FOO_, BAR)())
///
/// int main()
/// {
///   std::cout << BUZZ() << std::endl;
/// }
/// \endcode
///
/// The above code expands to:
///
/// \code
/// #include <iostream>
/// 
/// int main()
/// {
///   std::cout << "foo_bar" << std::endl;
/// }
/// \endcode
///
#define PP_EXPAND(x) x

///////////////////////////////////////////////////////////////////////////////

/// \def PP_ARITY(...)
/// \brief Returns the number of arguments that \a PP_ARITY was called with.
///
/// \par <b>Example</b>:
///
/// \code
/// #include <iostream>
/// 
/// int main()
/// {
///   std::cout << PP_ARITY()        << std::endl
///             << PP_ARITY(x)       << std::endl
///             << PP_ARITY(x, y)    << std::endl
///             << PP_ARITY(x, y, z) << std::endl;
/// }
/// \endcode
///
/// The above code expands to:
///
/// \code
/// #include <iostream>
/// 
/// int main()
/// {
///   std::cout << 0 << std::endl
///             << 1 << std::endl
///             << 2 << std::endl
///             << 3 << std::endl;
/// }
/// \endcode
///
#define PP_ARITY(...)                                                         \
  PP_EXPAND(PP_ARITY_IMPL(__VA_ARGS__,                                        \
  63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,                            \
  47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,                            \
  31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,                            \
  15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))                           \
  /**/

#define PP_ARITY_IMPL(                                                        \
   _1, _2, _3, _4, _5, _6, _7, _8, _9,_10,_11,_12,_13,_14,_15,_16,            \
  _17,_18,_19,_20,_21,_22,_23,_24,_25,_26,_27,_28,_29,_30,_31,_32,            \
  _33,_34,_35,_36,_37,_38,_39,_40,_41,_42,_43,_44,_45,_46,_47,_48,            \
  _49,_50,_51,_52,_53,_54,_55,_56,_57,_58,_59,_60,_61,_62,_63,  N,...) N      \
  /**/

/// \def PP_DISPATCH(basename, ...)
/// \brief Expands to <tt>basenameN(...)</tt>, where <tt>N</tt> is the number
///        of variadic arguments that \a PP_DISPATCH was called with. This macro
///        can be used to implement "macro overloading".
///
/// \par <b>Example</b>:
///
/// \code
/// #include <iostream>
/// 
/// #define PLUS(...) PP_DISPATCH(PLUS, __VA_ARGS__)
/// #define PLUS1(x)       x
/// #define PLUS2(x, y)    x + y
/// #define PLUS3(x, y, z) x + y + z
/// 
/// int main()
/// {
///   std::cout << PLUS(1)       << std::endl
///             << PLUS(1, 2)    << std::endl
///             << PLUS(1, 2, 3) << std::endl; 
/// }
/// \endcode
///
/// The above code expands to:
///
/// \code
/// #include <iostream>
/// 
/// #define PLUS(...) PP_DISPATCH(PLUS, __VA_ARGS__)
/// #define PLUS1(x)       x
/// #define PLUS2(x, y)    x + y
/// #define PLUS3(x, y, z) x + y + z
/// 
/// int main()
/// {
///   std::cout << 1         << std::endl
///             << 1 + 2     << std::endl
///             << 1 + 2 + 3 << std::endl; 
/// }
/// \endcode
///
#define PP_DISPATCH(basename, ...)                                            \
  PP_EXPAND(PP_CAT2(basename, PP_ARITY(__VA_ARGS__))(__VA_ARGS__))            \
  /**/

///////////////////////////////////////////////////////////////////////////////

/// \def CURRENT_FUNCTION
/// \brief The name of the current function as a string.
///
#if    defined(__GNUC__)                                                      \
    || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000))                        \
    || (defined(__ICC) && (__ICC >= 600)) || defined(__ghs__)
  #define CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__DMC__) && (__DMC__ >= 0x810)
  #define CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__FUNCSIG__)
  #define CURRENT_FUNCTION __FUNCSIG__
#elif    (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600))             \
      || (defined(__IBMCPP__) && (__IBMCPP__ >= 500))
  #define CURRENT_FUNCTION __FUNCTION__
#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)
  #define CURRENT_FUNCTION __FUNC__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
  #define CURRENT_FUNCTION __func__
#elif defined(__cplusplus) && (__cplusplus >= 201103)
  #define CURRENT_FUNCTION __func__
#else
  #define CURRENT_FUNCTION "(unknown)"
#endif

///////////////////////////////////////////////////////////////////////////////

/// \def FWD(x)
/// \brief Performs universal forwarding of a universal reference.
///
#define FWD(x) ::std::forward<decltype(x)>(x)

/// \def MV(x)
/// \brief Moves `x`.
///
#define MV(x) ::std::move(x)

/// \def RETOF(invocable, ...)
/// \brief Expands to the type returned by invoking an instance of the invocable
///        type \a invocable with parameters of type <tt>__VA_ARGS__</tt>.
///
#define RETOF(...)   PP_DISPATCH(RETOF, __VA_ARGS__)
#define RETOF1(C)    decltype(::std::declval<C>()())
#define RETOF2(C, V) decltype(::std::declval<C>()(::std::declval<V>()))

/// \def RETURNS(...)
/// \brief Expands to a function definition that returns the expression
///        <tt>__VA_ARGS__</tt>.
///
#define RETURNS(...)                                                          \
  noexcept(noexcept(decltype(__VA_ARGS__)(__VA_ARGS__)))                      \
  { return (__VA_ARGS__); }                                                   \
  /**/

/// \def AUTORETURNS(...)
/// \brief Expands to a function definition, including a trailing returning
///        type, that returns the expression <tt>__VA_ARGS__</tt>.
///
#define AUTORETURNS(...)                                                      \
  noexcept(noexcept(decltype(__VA_ARGS__)(__VA_ARGS__)))                      \
  -> decltype(__VA_ARGS__)                                                    \
  { return (__VA_ARGS__); }                                                   \
  /**/

/// \def AUTOQUALRETURNS(qual...)
/// \brief Expands to a function definition, including a trailing returning
///        type and a qualifier, that returns the expression
///        <tt>__VA_ARGS__</tt>.
///
#define AUTOQUALRETURNS(qualifier, ...)                                       \
  qualifier                                                                   \
  noexcept(noexcept(decltype(__VA_ARGS__)(__VA_ARGS__)))                      \
  -> decltype(__VA_ARGS__)                                                    \
  { return (__VA_ARGS__); }                                                   \
  /**/

///////////////////////////////////////////////////////////////////////////////

struct cuda_drv_exception : std::exception
{
  cuda_drv_exception(CUresult error_, char const* message_)
    : error(error_)
  {
    char const* str = nullptr;
    cuGetErrorName(error_, &str);
    message = str;
    message += ": ";
    cuGetErrorString(error_, &str);
    message += str;
    message += ": ";
    message += message_;
  }

  CUresult code() const
  {
    return error;
  } 

  virtual char const* what() const noexcept
  { 
    return message.c_str();
  } 

private:
  CUresult error;
  std::string message;
};

inline void throw_on_cuda_drv_error(CUresult error, char const* message)
{
  if (CUDA_SUCCESS != error)
    throw cuda_drv_exception(error, message);
}

#define THROW_ON_CUDA_DRV_ERROR(error)                                        \
  throw_on_cuda_drv_error(error, CURRENT_FUNCTION)                            \
  /**/

///////////////////////////////////////////////////////////////////////////////

struct cuda_context_deleter final
{
  void operator()(CUctx_st* context) const
  {
    if (nullptr != context)
      THROW_ON_CUDA_DRV_ERROR(cuCtxDestroy(context));
  }
};

struct cuda_context final
{
private:
  std::unique_ptr<CUctx_st, cuda_context_deleter> ptr_;

public:
  cuda_context(int32_t device_ordinal = 0)
  {
    CUdevice device;
    THROW_ON_CUDA_DRV_ERROR(cuDeviceGet(&device, device_ordinal));

    CUctx_st* context;
    THROW_ON_CUDA_DRV_ERROR(cuCtxCreate(&context, 0, device));

    ptr_.reset(context);
  }

  cuda_context(cuda_context const&)     = delete;
  cuda_context(cuda_context&&) noexcept = default;

  CUctx_st& operator*()  const RETURNS(*ptr_.get());
  CUctx_st* operator->() const RETURNS(ptr_.get());
  CUctx_st* get()        const RETURNS(ptr_.get());
};

///////////////////////////////////////////////////////////////////////////////

struct cuda_module_deleter final
{
  void operator()(CUmod_st* module) const
  {
    if (nullptr != module)
      THROW_ON_CUDA_DRV_ERROR(cuModuleUnload(module));
  }
};

struct cuda_module final
{
private:
  std::unique_ptr<CUmod_st, cuda_module_deleter> ptr_;

public:
  cuda_module(char const* filename)
  {
    CUmod_st* module;
    THROW_ON_CUDA_DRV_ERROR(cuModuleLoad(&module, filename));
    ptr_.reset(module);
  }

  cuda_module(cuda_module const&)     = delete;
  cuda_module(cuda_module&&) noexcept = default;

  CUmod_st& operator*()  const RETURNS(*ptr_.get());
  CUmod_st* operator->() const RETURNS(ptr_.get());
  CUmod_st* get()        const RETURNS(ptr_.get());
};

///////////////////////////////////////////////////////////////////////////////

struct cuda_function_deleter final
{
  void operator()(CUfunc_st* function) const {}
};

struct cuda_function final
{
private:
  std::unique_ptr<CUfunc_st, cuda_function_deleter> ptr_;

public:
  cuda_function(cuda_module& module, char const* function_name)
  {
    CUfunc_st* function;
    THROW_ON_CUDA_DRV_ERROR(
      cuModuleGetFunction(&function, module.get(), function_name)
    );

    ptr_.reset(function);
  }

  cuda_function(cuda_function const&)     = delete;
  cuda_function(cuda_function&&) noexcept = default;

  CUfunc_st& operator*()  const RETURNS(*ptr_.get());
  CUfunc_st* operator->() const RETURNS(ptr_.get());
  CUfunc_st* get()        const RETURNS(ptr_.get());
};

///////////////////////////////////////////////////////////////////////////////

struct cuda_stream_deleter final
{
  void operator()(CUstream_st* stream) const
  {
    if (nullptr != stream)
      THROW_ON_CUDA_DRV_ERROR(cuStreamDestroy(stream));
  }
};

struct cuda_stream final
{
private:
  std::unique_ptr<CUstream_st, cuda_stream_deleter> ptr_;

public:
  cuda_stream()
  {

    CUstream_st* stream;
    THROW_ON_CUDA_DRV_ERROR(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    ptr_.reset(stream);
  }

  cuda_stream(cuda_stream const&)     = delete;
  cuda_stream(cuda_stream&&) noexcept = default;

  CUstream_st& operator*()  const RETURNS(*ptr_.get());
  CUstream_st* operator->() const RETURNS(ptr_.get());
  CUstream_st* get()        const RETURNS(ptr_.get());

  void wait()
  {
    THROW_ON_CUDA_DRV_ERROR(cuStreamSynchronize(get()));
  }
};

///////////////////////////////////////////////////////////////////////////////

struct cuda_graph_deleter final
{
  void operator()(CUgraph_st* graph) const
  {
    if (nullptr != graph)
      THROW_ON_CUDA_DRV_ERROR(cuGraphDestroy(graph));
  }
};

struct cuda_graph final
{
private:
  std::unique_ptr<CUgraph_st, cuda_graph_deleter> ptr_;

public:
  cuda_graph()
  {
    CUgraph_st* graph;
    THROW_ON_CUDA_DRV_ERROR(cuGraphCreate(&graph, 0));
    ptr_.reset(graph);
  }

  cuda_graph(cuda_graph const&)     = delete;
  cuda_graph(cuda_graph&&) noexcept = default;

  CUgraph_st& operator*()  const RETURNS(*ptr_.get());
  CUgraph_st* operator->() const RETURNS(ptr_.get());
  CUgraph_st* get()        const RETURNS(ptr_.get());
};

struct cuda_compiled_graph_deleter final
{
  void operator()(CUgraphExec_st* graph) const
  {
    if (nullptr != graph)
      THROW_ON_CUDA_DRV_ERROR(cuGraphExecDestroy(graph));
  }
};

struct cuda_compiled_graph final
{
private:
  std::unique_ptr<CUgraphExec_st, cuda_compiled_graph_deleter> ptr_;

public:
  cuda_compiled_graph() = default;

  cuda_compiled_graph(cuda_graph& template_)
  {
    CUgraphExec_st* graph;
    CUgraphNode n;
    THROW_ON_CUDA_DRV_ERROR(
      cuGraphInstantiate(&graph, template_.get(), nullptr, nullptr, 0)
    );
    ptr_.reset(graph);
  }

  cuda_compiled_graph(cuda_compiled_graph const&)     = delete;
  cuda_compiled_graph(cuda_compiled_graph&&) noexcept = default;

  cuda_compiled_graph& operator=(cuda_compiled_graph const&)     = delete;
  cuda_compiled_graph& operator=(cuda_compiled_graph&&) noexcept = default;

  CUgraphExec_st& operator*()  const RETURNS(*ptr_.get());
  CUgraphExec_st* operator->() const RETURNS(ptr_.get());
  CUgraphExec_st* get()        const RETURNS(ptr_.get());

  void reset() { ptr_.reset(); }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct index_iterator final : std::iterator<
  // iterator_category
  std::random_access_iterator_tag
  // value_type
, std::remove_cv_t<std::remove_reference_t<T>>
  // difference_type
, decltype(std::declval<T>() - std::declval<T>())
  // pointer
, std::add_pointer_t<std::remove_cv_t<std::remove_reference_t<T>>>
  // reference
, std::remove_cv_t<std::remove_reference_t<T>>
>
{
  using iterator_category = std::random_access_iterator_tag;
  using value_type        = std::remove_cv_t<std::remove_reference_t<T>>;
  using difference_type   = decltype(std::declval<T>() - std::declval<T>());
  using pointer           = std::add_pointer_t<value_type>;
  using reference         = value_type;

private:
  T current_;

public:
  struct sentinel
  {
    T const last;

    constexpr sentinel(T last_) : last(MV(last_)) {}
  }; 

  constexpr index_iterator(T current) : current_(MV(current)) {}

  // InputIterator requirements

  constexpr auto operator*() RETURNS(current_);

  friend constexpr bool
  operator==(index_iterator const& l, index_iterator const& r)
  RETURNS(l.current_ == r.current_);
  friend constexpr bool
  operator==(sentinel const& l,       index_iterator const& r)
  RETURNS(l.last     == r.current_);
  friend constexpr bool
  operator==(index_iterator const& l, sentinel const& r)
  RETURNS(l.current_ == r.last);

  friend constexpr bool
  operator!=(index_iterator const& l, index_iterator const& r)
  RETURNS(!(l == r));
  friend constexpr bool
  operator!=(sentinel const& l,       index_iterator const& r)
  RETURNS(!(l == r));
  friend constexpr bool
  operator!=(index_iterator const& l, sentinel const& r)
  RETURNS(!(l == r));

  // ForwardIterator requirements

  constexpr index_iterator& operator++()
  {
    ++current_;
    return *this;
  }

  constexpr index_iterator& operator++(int)
  {
    index_iterator tmp(*this);
    ++tmp;
    return MV(tmp);
  }

  // BidirectionIterator requirements

  constexpr index_iterator& operator--()
  {
    --current_;
    return *this;
  }

  constexpr index_iterator& operator--(int)
  {
    index_iterator tmp(*this);
    --tmp;
    return MV(tmp);
  }

  // RandomAccessIterator requirements

  friend constexpr index_iterator& operator+=(index_iterator& l, T const& r) 
  {
    l.current_ += r;
    return l;
  }

  friend constexpr index_iterator operator+(T const& l, index_iterator r) 
  RETURNS(r += l);
  friend constexpr index_iterator operator+(index_iterator l, T const& r) 
  RETURNS(l += r);

  friend constexpr index_iterator& operator-=(index_iterator& l, T const& r) 
  RETURNS(l += -r);

  friend constexpr index_iterator operator-(index_iterator l, T const& r) 
  RETURNS(l -= r);

  friend constexpr auto
  operator-(index_iterator const& l, index_iterator const& r) 
  RETURNS(l.current_ - r.current_);
  friend constexpr auto
  operator-(sentinel const& l,       index_iterator const& r) 
  RETURNS(l.last     - r.current_);
  friend constexpr auto
  operator-(index_iterator const& l, sentinel const& r) 
  RETURNS(l.current_ - r.last);

  constexpr auto operator[](T n) RETURNS(current_ + MV(n));

  friend constexpr bool
  operator< (index_iterator const& l, index_iterator const& r)
  RETURNS(MV(l.current_) < MV(r.current_));
  friend constexpr bool
  operator> (index_iterator const& l, index_iterator const& r)
  RETURNS(r < l);
  friend constexpr bool
  operator<=(index_iterator const& l, index_iterator const& r)
  RETURNS(!(l > r));
  friend constexpr bool
  operator>=(index_iterator const& l, index_iterator const& r)
  RETURNS(!(l < r));

  friend constexpr bool
  operator< (sentinel const& l, index_iterator const& r)
  RETURNS(MV(l.last) < MV(r.current_));
  friend constexpr bool
  operator> (sentinel const& l, index_iterator const& r)
  RETURNS(r < l);
  friend constexpr bool
  operator<=(sentinel const& l, index_iterator const& r)
  RETURNS(!(l > r));
  friend constexpr bool
  operator>=(sentinel const& l, index_iterator const& r)
  RETURNS(!(l < r));

  friend constexpr bool
  operator< (index_iterator const& l, sentinel const& r)
  RETURNS(MV(l.current_) < MV(r.last));
  friend constexpr bool
  operator> (index_iterator const& l, sentinel const& r)
  RETURNS(r < l);
  friend constexpr bool
  operator<=(index_iterator const& l, sentinel const& r)
  RETURNS(!(l > r));
  friend constexpr bool
  operator>=(index_iterator const& l, sentinel const& r)
  RETURNS(!(l < r));
};

template <typename Iterator, typename Sentinel> 
struct iterator_sentinel_range final
{
private:
  Iterator first_;
  Sentinel last_;

public:
  constexpr iterator_sentinel_range(Iterator first, Sentinel last) noexcept
    : first_(MV(first)), last_(MV(last))
  {}

  Iterator constexpr begin() const RETURNS(first_);
  Sentinel constexpr end()   const RETURNS(last_);

  auto size() const RETURNS(last_ - first_);
};

// Python style `xrange`.
template <typename Integral>
auto constexpr xrange(Integral last)
{
  using iterator = index_iterator<Integral>;
  using sentinel = typename index_iterator<Integral>::sentinel;
  return iterator_sentinel_range<iterator, sentinel>(
    iterator(Integral(0)), sentinel(last)
  );
}

template <typename... Sizes>
auto constexpr make_index_array(Sizes&&... sizes)
RETURNS(std::array<int64_t, sizeof...(Sizes)>{FWD(sizes)...});

///////////////////////////////////////////////////////////////////////////////
// `std::transform_reduce` and `std::reduce` (Iterator and Range Interfaces).

template <
  typename InputItBegin, typename InputItEnd
, typename T
, typename ReduceOperation, typename TransformOperation
>
T constexpr transform_reduce(
  InputItBegin first, InputItEnd last
, T init
, ReduceOperation reduce_op
, TransformOperation transform_op
)
{
  while (first != last)
    init = reduce_op(MV(init), transform_op(*first++));
  return init;
}
template <
  typename Range
, typename T
, typename ReduceOperation, typename TransformOperation
>
auto constexpr transform_reduce_r(
  Range&& r
, T&& init
, ReduceOperation&& reduce_op
, TransformOperation&& transform_op
)
{
  return transform_reduce(
    r.begin(), r.end(), FWD(init), FWD(reduce_op), FWD(transform_op)
  ); 
}

template <
  typename InputItBegin0, typename InputItEnd0
, typename InputIt1
, typename T
, typename ReduceOperation, typename TransformOperation
>
T constexpr transform_reduce(
  InputItBegin0 first0, InputItEnd0 last0
, InputIt1 first1
, T init
, ReduceOperation reduce_op
, TransformOperation transform_op
)
{
  while (first0 != last0)
    init = reduce_op(MV(init), transform_op(*first0++, *first1++));
  return init;
}
template <
  typename Range0, typename Range1
, typename T
, typename ReduceOperation, typename TransformOperation
>
auto constexpr transform_reduce_r(
  Range0&& r0, Range1&& r1
, T&& init
, ReduceOperation&& reduce_op
, TransformOperation&& transform_op
)
{
  return transform_reduce(
    r0.begin(), r0.end(), r1.begin(), FWD(init), FWD(reduce_op), FWD(transform_op)
  ); 
}

template <
  typename InputItBegin, typename InputItEnd
, typename T
, typename ReduceOperation
>
T constexpr reduce(
  InputItBegin first, InputItEnd last
, T init
, ReduceOperation reduce_op
)
{
  for (; first != last; ++first) 
    init = reduce_op(MV(init), *first);
  return init;
}
template <typename Range, typename T, typename ReduceOperation>
auto constexpr reduce_r(Range&& r, T&& init, ReduceOperation&& reduce_op)
{
  return reduce(r.begin(), r.end(), FWD(init), FWD(reduce_op));
}

template <typename InputItBegin, typename InputItEnd, typename T>
auto constexpr reduce(InputItBegin first, InputItEnd last, T init)
{
  return reduce(first, last, FWD(init), std::plus<>());
}
template <typename Range, typename T>
auto constexpr reduce_r(Range&& r, T&& init)
{
  return reduce(r.begin(), r.end(), FWD(init));
}

template <typename InputItBegin, typename InputItEnd>
auto constexpr reduce(InputItBegin first, InputItEnd last)
{
  using T = typename std::iterator_traits<InputItBegin>::value_type;
  return reduce(first, last, T{}, std::plus<>());
}
template <typename Range>
auto constexpr reduce_r(Range&& r)
{
  return reduce(r.begin(), r.end());
}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
bool constexpr floating_point_equal(
  T x, T y, T epsilon = std::numeric_limits<T>::epsilon()
)
{
  if ((x + epsilon >= y) && (x - epsilon <= y)) return true;
  else return false;
}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct squared_difference final
{
private:
  T const average_;

public:
  constexpr squared_difference(T average) 
    : average_(average) {}

  constexpr squared_difference(squared_difference const& rhs) = default;
  constexpr squared_difference(squared_difference&& rhs)      = default;

  T constexpr operator()(T x) const RETURNS((x - average_) * (x - average_));
};

template <typename T>
struct value_and_count final
{
  T       value;
  int64_t count;

  constexpr value_and_count(T value_) 
    : value(MV(value_)), count(1)
  {}

  constexpr value_and_count(T value_, int64_t count_)
    : value(MV(value_)), count(count_)
  {}

  constexpr value_and_count(value_and_count const& rhs) = default;
  constexpr value_and_count(value_and_count&& rhs)      = default;

  constexpr value_and_count& operator=(value_and_count const& rhs) = default;
  constexpr value_and_count& operator=(value_and_count&& rhs)      = default;

  constexpr value_and_count& operator=(T value_)
  {
    value = MV(value_);
    count = 1;
    return *this;
  }
};

template <typename T, typename ReduceOp>
struct counting_op final
{
private:
  ReduceOp reduce_;

public:
  constexpr counting_op() = default;

  constexpr counting_op(counting_op const& rhs) = default;
  constexpr counting_op(counting_op&& rhs)      = default;

  constexpr counting_op(ReduceOp reduce) : reduce_(MV(reduce)) {}

  constexpr value_and_count<T>
  operator()(value_and_count<T> x, T y) const
  {
    return value_and_count<T>(reduce_(MV(x.value), MV(y)), MV(x.count) + 1);
  }

  constexpr value_and_count<T> 
  operator()(value_and_count<T> x, value_and_count<T> y) const
  {
    return value_and_count<T>(
      reduce_(MV(x.value), MV(y.value))
    , MV(x.count) + MV(y.count)
    );
  }
};

template <typename InputIt, typename T>
T constexpr arithmetic_mean(InputIt first, InputIt last, T init)
{
  value_and_count<T> init_vc(init, 0);

  counting_op<T, std::plus<T>> reduce_vc;

  value_and_count<T> vc = std::accumulate(first, last, init_vc, reduce_vc);

  return vc.value / vc.count;
}

template <typename InputIt>
auto constexpr arithmetic_mean(InputIt first, InputIt last)
{
  using T = typename std::iterator_traits<InputIt>::value_type;
  return arithmetic_mean(first, last, T{});
}

template <typename InputIt, typename T>
T constexpr sample_standard_deviation(InputIt first, InputIt last, T average)
{
  value_and_count<T> init_vc(T{}, 0);

  counting_op<T, std::plus<T>> reduce_vc;

  squared_difference<T> transform(average);

  value_and_count<T> vc
    = transform_reduce(first, last, init_vc, reduce_vc, transform);

  return std::sqrt(vc.value / T(vc.count - 1));
}

///////////////////////////////////////////////////////////////////////////////
// Formulas for propagation of uncertainty from:
//
//   https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulas
//
// Even though it's Wikipedia, I trust it as I helped write that table.
//
// TODO: Replace with a proper reference.

// Compute the propagated uncertainty from the multiplication of two uncertain
// values, `A +/- A_unc` and `B +/- B_unc`. Given `f = AB` or `f = A/B`, where
// `A != 0` and `B != 0`, the relative uncertainty in `f` is approximately:
//
//   f_unc = abs(f) * sqrt((A_unc / A) ^ 2 + (B_unc / B) ^ 2)
//
template <typename T>
T constexpr uncertainty_multiplicative(
    T const& f
  , T const& A, T const& A_unc
  , T const& B, T const& B_unc
    )
{
  return std::abs(f)
       * std::sqrt((A_unc / A) * (A_unc / A) + (B_unc / B) * (B_unc / B));
}

// Compute the propagated uncertainty from addition of two uncertain values,
// `A +/- A_unc` and `B +/- B_unc`. Given `f = cA + dB` (where `c` and `d` are
// certain constants), the relative uncertainty in `f` is approximately:
//
//   f_unc = sqrt(c ^ 2 * A_unc ^ 2 + d ^ 2 * B_unc ^ 2)
//
template <typename T>
T constexpr uncertainty_additive(
    T const& c, T const& A_unc
  , T const& d, T const& B_unc
    )
{
  return std::sqrt((c * c * A_unc * A_unc) + (d * d * B_unc * B_unc));
}

// Given an uncertain value `A` and its absolute uncertainty `+/- A_abs_unc`,
// compute its relative uncertainty (a unitless value between 0 and 1):
//
//   A_rel_unc = A_abs_unc / A
//
// Precondition: `0 != A`.
//
template <typename T>
T constexpr uncertainty_absolute_to_relative(T const& A, T const& A_abs_unc)
RETURNS(A_abs_unc / A);

// Given an uncertain value `A` and its relative uncertainty `+/- A * A_rel_unc`,
// compute its absolute uncertainty:
//
//   A_abs_unc = A_rel_unc * A
//
template <typename T>
T constexpr uncertainty_relative_to_absolute(T const& A, T const& A_rel_unc)
RETURNS(A_rel_unc * A);

///////////////////////////////////////////////////////////////////////////////

// Return the significant digit of `x`. The result is the number of digits
// after the decimal place to round to (negative numbers indicate rounding
// before the decimal place)
template <typename T>
int32_t constexpr find_significant_digit(T x)
{
  if (x == T(0)) return T(0);
  return -int32_t(std::floor(std::log10(std::abs(x))));
}

// Round `x` to `ndigits` after the decimal place (Python-style).
template <typename T, typename N>
T constexpr round_to_precision(T x, N ndigits)
{
  double m = (x < 0.0) ? -1.0 : 1.0;
  double pwr = std::pow(T(10.0), ndigits);
  return (std::floor(x * m * pwr + 0.5) / pwr) * m;
}

// Round `x` to its significant digit.
template <typename T>
T constexpr round_to_precision(T x)
{
  return round_to_precision(x, find_significant_digit(x));
}

///////////////////////////////////////////////////////////////////////////////

/*
concept UncertainValue
{
  using value_type = ...;

  // Returns: The uncertain value rounded to its significant digit or the
  // significant digit of the absolute uncertainty if it is higher.
  value_type value() const;
  operator value_type() const { return value(); } 
 
  // Returns: The absolute uncertanity rounded to its significant digit or the
  // significant of the uncertain value if it is higher.
  value_type absolute_uncertainty() const;

  // Returns: The relative uncertanity rounded to its significant digit.
  value_type relative_uncertainty() const;

  // Returns: The unrounded uncertain value.
  value_type value_unrounded() const;

  // Returns: The unrounded absolute uncertainty.
  value_type absolute_uncertainty_unrounded() const;

  // Returns: The unrounded relative uncertainty.
  value_type relative_uncertainty_unrounded() const;

  friend std::ostream& operator<<(std::ostream& os, UncertainValue const& uv);
};
*/

template <typename T, typename Derived>
struct uncertain_value
{
  using value_type = T;

  constexpr Derived const& derived() const noexcept
  {
    return static_cast<Derived const&>(*this);
  }

  value_type constexpr value() const
  {
    int32_t const precision = std::max(
      find_significant_digit(derived().value_unrounded())
    , find_significant_digit(derived().absolute_uncertainty_unrounded())
    );
    return round_to_precision(derived().value_unrounded(), precision);
  }

  constexpr operator value_type() const { return derived().value(); } 

  value_type constexpr absolute_uncertainty() const
  {
    int32_t const precision = std::max(
      find_significant_digit(derived().value_unrounded())
    , find_significant_digit(derived().absolute_uncertainty_unrounded())
    );
    return round_to_precision(
      derived().absolute_uncertainty_unrounded(), precision
    );
  }

  value_type constexpr relative_uncertainty() const 
  {
    return round_to_precision(derived().relative_uncertainty_unrounded());
  }
};

template <typename T, typename Derived>
std::ostream&
operator<<(std::ostream& os, uncertain_value<T, Derived> const& uc)
RETURNS(
  os        << uc.value()
     << "," << uc.absolute_uncertainty()
     << "," << 100.0 * uc.relative_uncertainty()
);

///////////////////////////////////////////////////////////////////////////////

template <typename Time, typename SampleSize = int64_t>
struct experiment_results final
  : uncertain_value<Time, experiment_results<Time, SampleSize>>
{
  using base_type = uncertain_value<Time, experiment_results<Time, SampleSize>>;

  using typename base_type::value_type;

  SampleSize const warmup_size;
  SampleSize const sample_size;
  value_type const average_time;   // Arithmetic mean of trial times in seconds.
  value_type const stdev_time;     // Sample standard deviation of trial times.

  constexpr experiment_results(
      SampleSize  warmup_size_
    , SampleSize  sample_size_
    , value_type  average_time_
    , value_type  stdev_time_
    )
    : warmup_size(MV(warmup_size_))
    , sample_size(MV(sample_size_))
    , average_time(MV(average_time_))
    , stdev_time(MV(stdev_time_))
  {}

  value_type constexpr value_unrounded() const
  RETURNS(average_time);

  value_type constexpr absolute_uncertainty_unrounded() const
  RETURNS(stdev_time);

  value_type constexpr relative_uncertainty_unrounded() const noexcept
  {
    return uncertainty_absolute_to_relative(
      value_unrounded(), absolute_uncertainty_unrounded()
    );
  }
};

template <typename Time, typename SampleSize>
std::ostream&
operator<<(std::ostream& os, experiment_results<Time, SampleSize> const& e)
{
  os        << e.warmup_size
     << "," << e.sample_size
     << "," << static_cast<
                 typename experiment_results<Time, SampleSize>::base_type const&
               >(e);
  return os;
};

template <typename Time, typename SampleSize>
auto constexpr make_experiment_results(
  SampleSize warmup_size
, SampleSize sample_size
, Time average_time
, Time stdev_time
)
{
  return experiment_results<Time, SampleSize>(
      MV(warmup_size)
    , MV(sample_size)
    , MV(average_time)
    , MV(stdev_time)
  );
}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct linear_regression_no_intercept final
  : uncertain_value<T, linear_regression_no_intercept<T>>
{
  using base_type = uncertain_value<T, linear_regression_no_intercept<T>>;

  using typename base_type::value_type;

  value_type const slope;
  value_type const r_squared;

  // Precondition: `0 <= r_squared_ && 1 >= r_squared_`.
  constexpr linear_regression_no_intercept(T slope_, T r_squared_)
    : slope(MV(slope_))
    , r_squared(MV(r_squared_))
  {}

  value_type constexpr value_unrounded() const
  RETURNS(slope);

  value_type constexpr absolute_uncertainty_unrounded() const noexcept
  {
    return uncertainty_relative_to_absolute(
      value_unrounded(), relative_uncertainty_unrounded()
    );
  }

  value_type constexpr relative_uncertainty_unrounded() const
  RETURNS(1.0 - r_squared);
};

template <typename T>
auto constexpr make_linear_regression_no_intercept(T slope, T r_squared)
RETURNS(linear_regression_no_intercept<T>(MV(slope), MV(r_squared)));

///////////////////////////////////////////////////////////////////////////////

/// \brief Computes a linear regression of \a y vs \a x with an origin of 
///        \a (x_origin, y_origin) with no intercept term and no variance in
///        \a x via ordinary least squares.
///
/// Requires: `IndependentValues` and `DependentValues` are random access
/// ranges with arithmetic value types.
///
/// Preconditions: `x.size() == y.size()`.
///
template <typename IndependentValues, typename DependentValues, typename T>
auto constexpr ordinary_least_squares_estimator_no_intercept(
  IndependentValues&& x 
, DependentValues&&   y
, T x_origin
, T y_origin
)
{
  // TODO: Tests for non (0, 0) origins.

  assert(x.size() == y.size());

  if (0 == x.size())
    return make_linear_regression_no_intercept(T{}, T{});

  auto const n  = x.size();

  T const x_sum = reduce_r(x, T{});
  T const y_sum = reduce_r(y, T{});

  T const x_squared_sum = transform_reduce_r(
    x, T{}, std::plus<>{}
  , [&] (auto x) { return (x - x_origin) * (x - x_origin); }
  );

  T const y_squared_sum = transform_reduce_r(
    y, T{}, std::plus<>{}
  , [&] (auto y) { return (y - y_origin) * (y - y_origin); }
  );

  T const xy_sum = transform_reduce_r(
    x, y, T{}, std::plus<>{}
  , [&] (auto x, auto y) { return (x - x_origin) * (y - y_origin); }
  );

  T const x_avg         = x_sum / n;
  T const y_avg         = y_sum / n;
  T const x_squared_avg = x_squared_sum / n;
  T const y_squared_avg = y_squared_sum / n;
  T const xy_avg        = xy_sum / n;

  // Formula for simple linear regression modelling with ordinary least squares
  // with no intercept:
  //
  //   https://en.wikipedia.org/wiki/Simple_linear_regression#Simple_linear_regression_without_the_intercept_term_(single_regressor)
  // 
  // TODO: Replace with a proper reference.

  // slope = avg((x - x_origin) * (y - y_origin))
  //       / avg((x - x_origin) * (x - x_origin))

  T const slope = ( !floating_point_equal(T{}, x_squared_avg)
                  ? xy_avg / x_squared_avg
                  : T{});

  // Formula for the coefficient of determination for linear regression models:
  //
  //   http://www.statisticshowto.com/probability-and-statistics/coefficient-of-determination-r-squared/
  // 
  // TODO: Replace with a proper reference.

  // r = (sum(x * y) - avg(x) * avg(y))
  //   / sqrt((sum(x * x) - avg(x) ^ 2) * (sum(y * y) - avg(y) ^ 2))
  //
  // r_squared = r ^ 2

  auto const r_denom_squared = (x_squared_sum - x_avg * x_avg)
                             * (y_squared_sum - y_avg * y_avg);

  auto const r = ( !floating_point_equal(r_denom_squared, 0.0)
                 ? (xy_sum - x_avg * y_avg) / std::sqrt(r_denom_squared)
                 : 0.0);

  auto const r_squared = r * r;

  return make_linear_regression_no_intercept(slope, r_squared);
}

/// \brief Computes a linear regression of \a y vs \a x with an origin of 
///        \a (0.0, 0.0) with no intercept term and no variance in a x via
///        ordinary least squares.
///
/// Requires: `IndependentValues` and `DependentValues` are random access
/// ranges with arithmetic value types.
///
/// Preconditions: `x.size() == y.size()`.
///
template <typename IndependentValues, typename DependentValues>
auto constexpr ordinary_least_squares_estimator_no_intercept(
  IndependentValues&& x 
, DependentValues&&   y
)
{
  return ordinary_least_squares_estimator_no_intercept(
    FWD(x), FWD(y), double(0.0), double(0.0)
  );
}

///////////////////////////////////////////////////////////////////////////////

template <typename Size, typename Test, typename Setup>
auto experiment(
  Size warmup_sample_size
, Size sample_size
, Test&& test
, Setup&& setup
)
{
  // Warmup trials.
  for (auto w : xrange(warmup_sample_size))
  {
    setup();
    test();
  }

  std::vector<double> times;
  times.reserve(sample_size);

  for (auto s : xrange(sample_size))
  {
    setup();

    // Record trial.
    auto const start = std::chrono::high_resolution_clock::now();
    test();
    auto const end   = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> const time = end - start;
    times.push_back(time.count());
  }
  
  double const average_time
    = arithmetic_mean(times.begin(), times.end());

  double const stdev_time
    = sample_standard_deviation(times.begin(), times.end(), average_time);

  return make_experiment_results(
    warmup_sample_size, sample_size, average_time, stdev_time
  );
}

template <typename Size, typename Test>
auto experiment(Size warmup_sample_size, Size sample_size, Test&& test)
{
  return experiment(MV(warmup_sample_size), MV(sample_size), FWD(test), [] {});
}

///////////////////////////////////////////////////////////////////////////////

struct cuda_launch_shape final
{
  int32_t const grid_size;  // Blocks per grid.
  int32_t const block_size; // Threads per block.

  constexpr cuda_launch_shape(int32_t grid_size_, int32_t block_size_)
    : grid_size(grid_size_), block_size(block_size_)
  {}
};

cuda_launch_shape cuda_compute_occupancy(cuda_function& f)
{
  int32_t grid_size  = 0;
  int32_t block_size = 0;

  THROW_ON_CUDA_DRV_ERROR(
    cuOccupancyMaxPotentialBlockSize(
      &grid_size, &block_size, f.get(), 0, 0, 0
    )
  ); 

  return cuda_launch_shape(grid_size, block_size);
}

cuda_launch_shape cuda_compute_occupancy(cuda_function& f, int32_t input_size)
{
  int32_t min_grid_size = 0;
  int32_t block_size    = 0;

  THROW_ON_CUDA_DRV_ERROR(
    cuOccupancyMaxPotentialBlockSize(
      &min_grid_size, &block_size, f.get(), 0, 0, 0
    )
  ); 

  // Round up based on input size.
  int32_t const grid_size = (input_size + block_size - 1) / block_size;

  return cuda_launch_shape(grid_size, block_size);
}

///////////////////////////////////////////////////////////////////////////////

template <typename Size, typename... Args> 
auto compile_graph_linearly_dependent(
  cuda_function&    f
, cuda_launch_shape shape
, Size              size
, Args&&...         args
)
{
  cuda_graph graph;

  void* args_ptrs[] = { std::addressof(args)... };

  CUDA_KERNEL_NODE_PARAMS desc;
  desc.func = f.get();
  desc.gridDimX = shape.grid_size;
  desc.gridDimY = 1;
  desc.gridDimZ = 1;
  desc.blockDimX = shape.block_size;
  desc.blockDimY = 1;
  desc.blockDimZ = 1;
  desc.sharedMemBytes = 0;
  desc.kernelParams = args_ptrs;
  desc.extra = nullptr;

  CUgraphNode_st* next = nullptr;
  CUgraphNode_st* prev = nullptr;

  for (auto t : xrange(size))
  {
    THROW_ON_CUDA_DRV_ERROR(cuGraphAddKernelNode(
      &next, graph.get(), &prev, prev != nullptr, &desc
    ));

    std::swap(next, prev);
  }

  return cuda_compiled_graph(graph); 
}

///////////////////////////////////////////////////////////////////////////////

template <typename... Args> 
void traditional_launch(
  cuda_function& f, cuda_stream& stream, cuda_launch_shape shape, Args&&... args
)
{
  void* args_ptrs[] = { std::addressof(args)... };
  THROW_ON_CUDA_DRV_ERROR(cuLaunchKernel(
    f.get()
  , shape.grid_size,  1, 1
  , shape.block_size, 1, 1
  , 0, stream.get(), args_ptrs, 0
  ));
}

inline void graph_launch(cuda_compiled_graph& cgraph, cuda_stream& stream)
{
  THROW_ON_CUDA_DRV_ERROR(cuGraphLaunch(cgraph.get(), stream.get()));
}

///////////////////////////////////////////////////////////////////////////////

std::vector<std::string> split(std::string const& str, std::string const& delim)
{
  // TODO Use `std::string_view` when possible.
  std::vector<std::string> tokens;
  std::string::size_type prev = 0, pos = 0;
  do
  {
    pos = str.find(delim, prev);
    if (pos == std::string::npos) pos = str.length();
    std::string token = str.substr(prev, pos - prev);
    if (!token.empty()) tokens.push_back(token);
    prev = pos + delim.length();
  }
  while (pos < str.length() && prev < str.length());
  return tokens;
}

///////////////////////////////////////////////////////////////////////////////

struct command_line_option_error : std::exception
{
  virtual ~command_line_option_error() noexcept {}
  virtual const char* what() const noexcept = 0;
};

struct only_one_option_allowed : command_line_option_error
{
  // Construct a new `only_one_option_allowed` exception. `key` is the
  // option name and `[first, last)` is a sequence of
  // `std::pair<std::string const, std::string>`s (the values).
  template <typename InputIt>
  only_one_option_allowed(std::string const& key, InputIt first, InputIt last)
    : message()
  {
    message  = "Only one `--";
    message += key;
    message += "` option is allowed, but multiple were received: ";

    for (; first != last; ++first)
    {
      message += "`";
      message += (*first).second;
      message += "` ";
    }

    // Remove the trailing space added by the last iteration of the above loop.
    message.erase(message.size() - 1, 1);

    message += ".";
  }

  virtual ~only_one_option_allowed() noexcept {}

  virtual const char* what() const noexcept
  {
    return message.c_str();
  }

private:
  std::string message;
};

struct required_option_missing : command_line_option_error
{
  // Construct a new `requirement_option_missing` exception. `key` is the
  // option name.
  required_option_missing(std::string const& key)
    : message()
  {
    message  = "`--";
    message += key;
    message += "` option is required.";
  }

  virtual ~required_option_missing() noexcept {}

  virtual const char* what() const noexcept
  {
    return message.c_str();
  }

private:
  std::string message;
};

struct command_line_processor
{
  typedef std::vector<std::string> positional_options_type;

  typedef std::multimap<std::string, std::string> keyword_options_type;

  typedef std::pair<
    keyword_options_type::const_iterator
  , keyword_options_type::const_iterator
  > keyword_option_values;

  command_line_processor(int argc, char** argv)
    : pos_args(), kw_args()
  { // {{{
    for (int i = 1; i < argc; ++i)
    {
      std::string arg(argv[i]);

      // Look for --key or --key=value options.
      if (arg.substr(0, 2) == "--")
      {
        std::string::size_type n = arg.find('=', 2);

        keyword_options_type::value_type key_value;

        if (n == std::string::npos) // --key
          kw_args.insert(keyword_options_type::value_type(
            arg.substr(2), ""
          ));
        else                        // --key=value
          kw_args.insert(keyword_options_type::value_type(
            arg.substr(2, n - 2), arg.substr(n + 1)
          ));

        kw_args.insert(key_value);
      }
      else // Assume it's positional.
        pos_args.push_back(arg);
    }
  } // }}}

  // Return the value for option `key`.
  //
  // Throws:
  // * `only_one_option_allowed` if there is more than one value for `key`.
  // * `required_option_missing` if there is no value for `key`.
  std::string operator()(std::string const& key) const
  {
    keyword_option_values v = kw_args.equal_range(key);

    keyword_options_type::difference_type d = std::distance(v.first, v.second);

    if      (1 < d)  // Too many options.
      throw only_one_option_allowed(key, v.first, v.second);
    else if (0 == d) // No option.
      throw required_option_missing(key);

    return (*v.first).second;
  }

  // Return the value for option `key`, or `dflt` if `key` has no value.
  //
  // Throws: `only_one_option_allowed` if there is more than one value for `key`.
  std::string operator()(std::string const& key, std::string const& dflt) const
  {
    keyword_option_values v = kw_args.equal_range(key);

    keyword_options_type::difference_type d = std::distance(v.first, v.second);

    if (1 < d)  // Too many options.
      throw only_one_option_allowed(key, v.first, v.second);

    if (0 == d) // No option.
      return dflt;
    else        // 1 option.
      return (*v.first).second;
  }

  // Returns `true` if the option `key` was specified at least once.
  bool has(std::string const& key) const
  {
    return kw_args.count(key) > 0;
  }

private:
  positional_options_type pos_args;
  keyword_options_type    kw_args;
};

///////////////////////////////////////////////////////////////////////////////

int main()
{
  cuInit(0);
  
  cuda_context context;

  cuda_module module("kernels.cubin");

  cuda_function     hello_world(module, "hello_world_kernel");
  cuda_launch_shape hello_world_shape(cuda_compute_occupancy(hello_world));

  cuda_function     noop(module, "noop_kernel");
  cuda_launch_shape noop_shape(cuda_compute_occupancy(noop));

  cuda_function     hang(module, "hang_kernel");
  cuda_launch_shape hang_shape(cuda_compute_occupancy(hang));

  cuda_function     payload(module, "payload_kernel");
  cuda_launch_shape payload_shape(cuda_compute_occupancy(payload));

  // Print CSV header first row (variable names)
  std::cout << "Test"
     << "," << "Kernels per Graph"
     << "," << "Dependencies per Graph"
     << "," << "Warmup Sample Size" 
     << "," << "Sample Size"
     << "," << "Average Execution Time"
     << "," << "Execution Time Absolute Uncertainty"
     << "," << "Execution Time Relative Uncertainty"
     << std::endl;

  // Print CSV header second row (variable units)
  std::cout << ""
     << "," << "vertices"
     << "," << "edges"
     << "," << "samples"
     << "," << "samples"
     << "," << "seconds"
     << "," << "seconds"
     << "," << "%"
     << std::endl;

  auto traditional_result = [&] {
    cuda_stream stream;

    auto result = experiment(
      128, 4096, [&] { traditional_launch(noop, stream, noop_shape); }
    );

    std::cout << "traditional_launch_linearly_dependent"
       << "," << 1
       << "," << 1
       << "," << result
       << std::endl;

    stream.wait();

    return result;
  }();

  {
    auto constexpr sizes = make_index_array(
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    , 32,   48,   64,   80,   96,   112,  128 
    , 256,  384,  512,  640,  768,  896,  1024
    , 2048, 3072, 4096, 5120, 6144, 7168, 8192
    );

    std::vector<decltype(traditional_result)> compile_results;
    compile_results.reserve(sizes.size());

    std::vector<decltype(traditional_result)> launch_results;
    launch_results.reserve(sizes.size());

    for (auto size : sizes)
    {
      cuda_compiled_graph cg;

      compile_results.emplace_back(experiment(
        32, 256
      , [&] { cg = MV(compile_graph_linearly_dependent(noop, noop_shape, size)); }
      , [&] { cg.reset(); }
      ));

      std::cout << "graph_compile_linearly_dependent"
         << "," << size
         << "," << (size - 1)
         << "," << compile_results.back()
         << std::endl;
    }

    for (auto size : sizes)
    {
      cuda_stream stream;

      auto cg = compile_graph_linearly_dependent(noop, noop_shape, size);

      launch_results.emplace_back(experiment(
        32, 256, [&] { graph_launch(cg, stream); }
      ));

      std::cout << "graph_launch_linearly_dependent"
         << "," << size
         << "," << (size - 1)
         << "," << launch_results.back()
         << std::endl;

      stream.wait();
    }

    std::cout << std::endl;

    auto const compile_model
      = ordinary_least_squares_estimator_no_intercept(
          sizes, compile_results
        );

    auto const launch_model
      = ordinary_least_squares_estimator_no_intercept(
          sizes, launch_results
        );

    std::cout << "traditional_launch_linearly_dependent"
       << "," << traditional_result.value_unrounded()
       << "," << traditional_result.absolute_uncertainty_unrounded() 
       << "," << traditional_result.relative_uncertainty_unrounded()
       << std::endl;

    std::cout << "graph_compile_linearly_dependent"
       << "," << compile_model.value_unrounded()
       << "," << compile_model.absolute_uncertainty_unrounded() 
       << "," << compile_model.relative_uncertainty_unrounded()
       << std::endl;

    std::cout << "graph_launch_linearly_dependent"
       << "," << launch_model.value_unrounded()
       << "," << launch_model.absolute_uncertainty_unrounded() 
       << "," << launch_model.relative_uncertainty_unrounded()
       << std::endl;
  }

  THROW_ON_CUDA_DRV_ERROR(cuCtxSynchronize());
}

