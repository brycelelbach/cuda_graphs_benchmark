#! /bin/bash
#
# Copyright 2017 NVIDIA Corporation
#
# Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)
#
# Author: Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>

if [[ $# -gt 1 ]]; then
  if [[ ${1} = "-h" ]] || [[ ${1} = "-help" ]] || [[ ${1} = "--help" ]]; then
    echo "Usage: $0 [<variant>] [<process-runs>] [<gpu-model>] [<build-directory>]"
    exit 1
  fi
fi

if [[ $# -ge 1 ]]; then
  VARIANT="${1}"
else
  VARIANT=TT
fi

if [[ $# -ge 2 ]]; then
  PROCESS_RUNS="${2}"
else
  PROCESS_RUNS=20
fi

if [[ $# -ge 3 ]]; then
  GPU_MODEL="${3}"
else
  GPU_MODEL=gp102
fi

if [[ $# -ge 4 ]]; then
  BUILD_DIR="${4}"
else
  BUILD_DIR=./build
fi

KMD_VERSION="r`nvidia-smi -i 0 --format=csv,noheader,nounits --query-gpu=driver_version`"
TIME_DATE=`date "+%Y-%m-%d_%H-%M-%S"`

cd ${BUILD_DIR}/${VARIANT}

mkdir -p ../../results

../../run_benchmark.sh ./benchmark_cuda_graphs 0 ${PROCESS_RUNS} ../../results/benchmark_cuda_graphs__${VARIANT}__${KMD_VERSION}__${GPU_MODEL}__${TIME_DATE}

