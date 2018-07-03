#! /bin/bash
#
# Copyright 2017 NVIDIA Corporation
#
# Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)
#
# Author: Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>

BUILD_DIR=./build

CUDATT_ROOT=/home/wash/development/nvidia/cuda_linux_p4
CUDATT_NVCC=${CUDATT_ROOT}/sw/gpgpu/bin/x86_64_Linux_release/nvcc
CUDATT_FLAGS="-I ${CUDATT_ROOT}/sw/gpgpu/thrust"

CUDA10_ROOT=/home/wash/install/nvidia/cuda-10.0.84
CUDA10_NVCC=${CUDA10_ROOT}/bin/nvcc
CUDA10_FLAGS=""

NVCC_CU_FLAGS="--std=c++14 -cubin -gencode arch=compute_61,code=sm_61"

NVCC_CPP_FLAGS="-ccbin g++-8 -Xcompiler -std=c++17 -Xcompiler -O3 -Xcompiler -msse -Xcompiler -fno-strict-aliasing -Xcompiler -frerun-loop-opt -Xcompiler -funroll-all-loops -lcuda"

mkdir -p ${BUILD_DIR}/TT

${CUDATT_NVCC} ${CUDATT_FLAGS} ${NVCC_CU_FLAGS} kernels.cu -o ${BUILD_DIR}/TT/kernels.cubin

echo "Building CUDA Top-of-Trunk Graphs Benchmark Suite"
${CUDATT_NVCC} ${CUDATT_FLAGS} ${NVCC_CPP_FLAGS} benchmark_cuda_graphs.cpp -o ${BUILD_DIR}/TT/benchmark_cuda_graphs

mkdir -p ${BUILD_DIR}/10

${CUDA10_NVCC} ${CUDA10_FLAGS} ${NVCC_CU_FLAGS} kernels.cu -o ${BUILD_DIR}/10/kernels.cubin

echo "Building CUDA 10.0 Graphs Benchmark Suite"
${CUDA10_NVCC} ${CUDA10_FLAGS} ${NVCC_CPP_FLAGS} benchmark_cuda_graphs.cpp -o ${BUILD_DIR}/10/benchmark_cuda_graphs

