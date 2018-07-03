#! /bin/bash
#
# Copyright 2017 NVIDIA Corporation
#
# Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)
#
# Author: Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <benchmark> <devices> <runs> <output-directory>"
  exit 1
fi

BENCHMARK="${1}"
DEVICES="${2}"
RUNS="${3}"
OUTPUT_DIR="${4}"

mkdir -p ${OUTPUT_DIR}

for device in ${DEVICES}
do
  UUID="`nvidia-smi -i ${device} --format=csv,noheader,nounits --query-gpu=gpu_uuid`"

  #############################################################################
  # Set application clocks.

  # https://devblogs.nvidia.com/parallelforall/increase-performance-gpu-boost-k80-autoboost/

  MAX_SM_CLOCK="`nvidia-smi -i ${device} --format=csv,noheader,nounits --query-gpu=clocks.max.sm`"
  MAX_MEMORY_CLOCK="`nvidia-smi -i ${device} --format=csv,noheader,nounits --query-gpu=clocks.max.memory`"

  sudo nvidia-smi -i ${device} -ac "${MAX_MEMORY_CLOCK},${MAX_SM_CLOCK}" > /dev/null

  for t in `seq -f "%03g" 0 $((${RUNS}-1))`
  do
    ###########################################################################
    # Display system information.

    echo "#,Time,`date`" | tee ${OUTPUT_DIR}/${t}.csv

    echo "#,P4 TOT Changelist,`p4 changes -m1 //sw/gpgpu/...#have | awk '{ print $2 }'`" | tee ${OUTPUT_DIR}/${t}.csv

    echo "#,OS Kernel,`uname -a`" | tee ${OUTPUT_DIR}/${t}.csv

    echo "#,CPU,`cat /proc/cpuinfo | grep "model name" | head -n 1 | awk -F ': ' '{ print $2 }'`" | tee ${OUTPUT_DIR}/${t}.csv

    echo "#,CPU Clock Frequency [MHz],`cat /proc/cpuinfo | grep "cpu MHz" | head -n 1 | awk -F ': ' '{ print $2 }'`" | tee ${OUTPUT_DIR}/${t}.csv

    echo "#,CPU Memory [MiB],$((`cat /proc/meminfo | grep MemTotal | sed 's/ \+/ /' | awk -F ' ' '{ print $2 }'` / 1024))" | tee ${OUTPUT_DIR}/${t}.csv

    ###########################################################################
    # Print environment information. 

    GPU_VARIABLES=()
    GPU_VARIABLE_NAMES=()

    GPU_VARIABLES[0]="gpu_name"
    GPU_VARIABLE_NAMES[0]="GPU ${device}"
    GPU_VARIABLES[1]="gpu_uuid"
    GPU_VARIABLE_NAMES[1]="GPU ${device} UUID"
    GPU_VARIABLES[2]="driver_version"
    GPU_VARIABLE_NAMES[2]="GPU ${device} KMD Version"
    GPU_VARIABLES[3]="vbios_version"
    GPU_VARIABLE_NAMES[3]="GPU ${device} VBIOS Version"
    GPU_VARIABLES[4]="memory.total"
    GPU_VARIABLE_NAMES[4]="GPU ${device} Memory [MiB]"
    GPU_VARIABLES[5]="pcie.link.gen.max"
    GPU_VARIABLE_NAMES[5]="GPU ${device} Max PCIe Link Generation"
    GPU_VARIABLES[6]="pcie.link.width.max"
    GPU_VARIABLE_NAMES[6]="GPU ${device} Max PCIe Link Width"
    GPU_VARIABLES[7]="persistence_mode"
    GPU_VARIABLE_NAMES[7]="GPU ${device} Persistence Mode"
    GPU_VARIABLES[8]="compute_mode"
    GPU_VARIABLE_NAMES[8]="GPU ${device} Compute Mode"
    GPU_VARIABLES[9]="gpu_operation_mode.current"
    GPU_VARIABLE_NAMES[9]="GPU ${device} Operation Mode"
    GPU_VARIABLES[10]="clocks.max.sm"
    GPU_VARIABLE_NAMES[10]="GPU ${device} Max SM Clock Frequency [MHz]"
    GPU_VARIABLES[11]="clocks.max.memory"
    GPU_VARIABLE_NAMES[11]="GPU ${device} Max Memory Clock Frequency [MHz]"
    GPU_VARIABLES[12]="clocks.applications.graphics"
    GPU_VARIABLE_NAMES[12]="GPU ${device} Application SM Clock Frequency [MHz]"
    GPU_VARIABLES[13]="clocks.applications.memory"
    GPU_VARIABLE_NAMES[13]="GPU ${device} Application Memory Clock Frequency [MHz]"

    for v in {0..12}
    do
      echo -n "#,${GPU_VARIABLE_NAMES[${v}]}," | tee ${OUTPUT_DIR}/${t}.csv
      nvidia-smi -i ${device} --format=csv,noheader,nounits --query-gpu="${GPU_VARIABLES[${v}]}"
    done

    #############################################################################
    # Run benchmark.

    CUDA_VISIBLE_DEVICES=${UUID} ${BENCHMARK} | tee ${OUTPUT_DIR}/${t}.csv
  done

  #############################################################################
  # Reset application clocks.

  sudo nvidia-smi -i ${device} -rac > /dev/null
done

