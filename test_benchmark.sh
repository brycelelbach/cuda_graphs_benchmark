#! /bin/bash
#
# Copyright 2017 NVIDIA Corporation
#
# Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)
#
# Author: Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>

if [[ $# -gt 1 ]]; then
  if [[ ${1} = "-h" ]] || [[ ${1} = "-help" ]] || [[ ${1} = "--help" ]]; then
    echo "Usage: $0 [<variant>] [<build-directory>]"
    exit 1
  fi
fi

if [[ $# -ge 1 ]]; then
  VARIANT="${1}"
else
  VARIANT=TT
fi

if [[ $# -ge 2 ]]; then
  BUILD_DIR="${2}"
else
  BUILD_DIR=./build
fi

cd ${BUILD_DIR}/${VARIANT}

./benchmark_cuda_graphs

