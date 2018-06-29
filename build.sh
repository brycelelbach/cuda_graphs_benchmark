#! /bin/bash

CUDATT_ROOT=/home/wash/development/nvidia/cuda_linux_p4
CUDATT_NVCC=${CUDATT_ROOT}/sw/gpgpu/bin/x86_64_Linux_release/nvcc
CUDATT_FLAGS="-I ${CUDATT_ROOT}/sw/gpgpu/thrust"

NVCC_CU_FLAGS="--std=c++14 -cubin -gencode arch=compute_61,code=sm_61"

NVCC_CPP_FLAGS="-ccbin g++-8 -Xcompiler -std=c++17 -Xcompiler -O3 -Xcompiler -msse -Xcompiler -fno-strict-aliasing -Xcompiler -frerun-loop-opt -Xcompiler -funroll-all-loops -lcuda"

echo "Building CUDA Top-of-Trunk Graphs Benchmark Suite"
${CUDATT_NVCC} ${CUDATT_FLAGS} ${NVCC_CU_FLAGS}  noop.cu              -o noop.cubin
${CUDATT_NVCC} ${CUDATT_FLAGS} ${NVCC_CPP_FLAGS} graphs_benchmark.cpp -o graphs_benchmark

