// Copyright 2018 NVIDIA Corporation
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)
//
// Author: Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>

#include <cstdio>

#include "clocks.hpp"

extern "C" {

__global__ void hello_world_kernel() { printf("hello world from device\n"); }

__global__ void noop_kernel() {}

__global__ void hang_kernel() { while (true) {} }

__global__ void payload_kernel(uint64_t expected)
{
  payload<cuda_globaltimer_clocksource>(expected);
}

__global__ void noop_dynamic_linearly_dependent_kernel(
  int32_t kernels_per_graph
, int32_t grid_size
, int32_t block_size
) {
  if (threadIdx.x == 0 && kernels_per_graph > 0)
    noop_dynamic_linearly_dependent_kernel<<<
      grid_size, block_size, 0, cudaStreamTailLaunch
    >>>(
      kernels_per_graph - 1, grid_size, block_size
    );
}

}

