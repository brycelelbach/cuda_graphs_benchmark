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

}

