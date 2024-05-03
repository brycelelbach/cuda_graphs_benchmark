// Copyright 2018 NVIDIA Corporation
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)
//
// Author: Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>

#pragma once

#include <chrono>
#include <cassert>

template <
  typename BaseClock = std::chrono::high_resolution_clock
, typename Period    = std::nano
>
struct std_chrono_clocksource
{
  using base_clock = BaseClock;
  using rep        = typename base_clock::rep;
  using period     = Period;
  using duration   = std::chrono::duration<rep, period>;

  static_assert(base_clock::is_steady == true, "base_clock is not steady");

  // Returns: Current time.
  // Postconditions: 0 <= t
  static rep timestamp()
  {
    duration const d = std::chrono::duration_cast<duration>(
      base_clock::now().time_since_epoch()
    );
    rep const t = d.count();
    assert(0 <= t);
    return t;
  }

  static double constexpr clock_uncertainty() noexcept
  {
    // For steady clocks, we use instrument uncertainty, ie:
    //   instrument_uncertainty = instrument_least_count / 2
    return 1.0 / 2.0;
  }
};

struct cuda_globaltimer_clocksource
{
  using rep      = uint64_t;
  using period   = std::nano;

  __host__ __device__
  static rep timestamp()
  {
    rep t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l" (t));
    return t;
  }

  static double constexpr clock_uncertainty() noexcept
  {
    // For steady clocks, we use instrument uncertainty, ie:
    //   instrument_uncertainty = instrument_least_count / 2
    return 1.0 / 2.0;
  }
};

// Performs approximately 'expected' duration of artificial work.
// Returns: The duration of the work performed.
template <typename Clocksource>
__host__ __device__
auto payload(typename Clocksource::rep expected)
{
  auto const start = Clocksource::timestamp();

  while (true)
  {
    auto const measured = Clocksource::timestamp() - start;

    if (measured >= expected)
      return measured;
  }
}

