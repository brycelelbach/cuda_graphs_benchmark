###############################################################################
# Copyright (c) 2018   NVIDIA Corporation
# Copyright (c) 2015-6 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
#
# Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)
###############################################################################

# Target device architecture (list of `[0-9][0-9]`s or `all`).
DEVICE_ARCHS    ?= all
# Class of build (`debug` or `release`).
BUILD_TYPE      ?= release
# The build directory to create and use.
BUILD_DIRECTORY ?= $(ROOT)/build__$(BUILD_TYPE)/

# C++ version (`c++14`, `c++1z`, or `c++17`)
CXX_VERSION     ?= c++14

# Host C++ (`.cpp`) -> Host Object File (`.o`) -> Host Executable

# Host ISO C++ compiler (path to executable).
ISO_CXX                      ?= c++
# Flags passed to the host/device CUDA C++ compiler when compiling host ISO C++ code.
HOST_ISO_CXX_FLAGS           += -lcuda
# Flags passed to the host ISO C++ compiler when compiling host ISO C++ code.
HOST_ISO_CXX_XCOMPILER_FLAGS ?=

# Device CUDA C++ (`.cu`) -> Device Binary (`.cubin`)

# Host/device CUDA C++ compiler (path to executable).
CUDA_CXX              ?= nvcc
# Flags for compiling device CUDA C++ code.
DEVICE_CUDA_CXX_FLAGS ?=

# TODO: Print out all build flags and do version check

$(info ###############################################################################)
$(info # Settings)
$(info ###############################################################################)
$(info # DEVICE_ARCHS                 : `$(value DEVICE_ARCHS)`)
$(info # BUILD_TYPE                   : `$(value BUILD_TYPE)`)
$(info # BUILD_DIRECTORY              : `$(value BUILD_DIRECTORY)`)
$(info # CXX_VERSION                  : `$(value CXX_VERSION)`)
$(info # ISO_CXX                      : `$(value CXX)`)
$(info # HOST_ISO_CXX_FLAGS           : `$(value HOST_ISO_CXX_FLAGS)`)
$(info # HOST_ISO_CXX_XCOMPILER_FLAGS : `$(value HOST_ISO_CXX_XCOMPILER_FLAGS)`)
$(info # CUDA_CXX                     : `$(value CUDA_CXX)`)
$(info # DEVICE_CUDA_CXX_FLAGS        : `$(value DEVICE_CUDA_CXX_FLAGS)`)
$(info ) # Print blank newline.

$(info ###############################################################################)
$(info # Compiler versions)
$(info ###############################################################################)
$(info # ISO C++ compiler version  : `$(shell $(ISO_CXX) --version)`)
$(info # CUDA C++ compiler Version : `$(shell $(CUDA_CXX) --version)`)
$(info ) # Print blank newline.

###############################################################################

# Tell the host/device CUDA C++ compiler which device architectures to generate
# code for when compiling device CUDA C++.
ifneq ($(DEVICE_ARCHS),all)
	GENCODE = -gencode arch=compute_$(arch),code=sm_$(arch)
  DEVICE_CUDA_CXX_FLAGS += $(foreach arch,$(DEVICE_ARCHS),$(GENCODE))
endif

ifeq      ($(BUILD_TYPE),release)
  HOST_ISO_CXX_XCOMPILER_FLAGS  += -O3
else ifeq ($(BUILD_TYPE),debug)
  HOST_ISO_CXX_XCOMPILER_FLAGS  += -O0
  DEVICE_CUDA_CXX_FLAGS         += -G
endif

HOST_ISO_CXX_FLAGS    += -std=$(CXX_VERSION)
DEVICE_CUDA_CXX_FLAGS += -std=$(CXX_VERSION)

# Tell the host/device CUDA C++ compiler which host ISO C++ compiler to use
# when compiling host ISO C++ code.
HOST_ISO_CXX_FLAGS += -ccbin $(ISO_CXX)

# Add `HOST_ISO_CXX_XCOMPILER_FLAGS` to `HOST_ISO_CXX_FLAGS`.
HOST_ISO_CXX_FLAGS += $(foreach flag,$(HOST_ISO_CXX_XCOMPILER_FLAGS),-Xcompiler $(flag))

###############################################################################

# Get the directory where the Makefile and code live.
ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Strip terminating `/`s from directory paths.
# This has to be done late because these inplace updates must be immediate.
ROOT            := $(ROOT:/=)
BUILD_DIRECTORY := $(BUILD_DIRECTORY:/=)

# Strip leading and trailing whitespace from flags.
HOST_ISO_CXX_FLAGS    := $(strip $(HOST_ISO_CXX_FLAGS))
DEVICE_CUDA_CXX_FLAGS := $(strip $(DEVICE_CUDA_CXX_FLAGS))

HOST_ISO_CXX_SOURCES    = $(wildcard $(ROOT)/*.cpp)
HOST_ISO_CXX_TARGETS    = $(HOST_ISO_CXX_SOURCES:.cpp=)

DEVICE_CUDA_CXX_SOURCES = $(wildcard $(ROOT)/*.cu)
DEVICE_CUDA_CXX_TARGETS = $(DEVICE_CUDA_CXX_SOURCES:.cu=.cubin)

###############################################################################

$(info ###############################################################################)
$(info # Computed variables)
$(info ###############################################################################)
$(info # ROOT                    : `$(ROOT)`)
$(info # BUILD_DIRECTORY         : `$(BUILD_DIRECTORY)`)
$(info # HOST_ISO_CXX_FLAGS      : `$(HOST_ISO_CXX_FLAGS)`)
$(info # HOST_ISO_CXX_SOURCES    : `$(HOST_ISO_CXX_SOURCES)`)
$(info # HOST_ISO_CXX_TARGETS    : `$(HOST_ISO_CXX_TARGETS)`)
$(info # DEVICE_CUDA_CXX_FLAGS   : `$(DEVICE_CUDA_CXX_FLAGS)`)
$(info # DEVICE_CUDA_CXX_SOURCES : `$(DEVICE_CUDA_CXX_SOURCES)`)
$(info # DEVICE_CUDA_CXX_TARGETS : `$(DEVICE_CUDA_CXX_TARGETS)`)
$(info ) # Print blank newline.

###############################################################################

all: $(HOST_ISO_CXX_TARGETS) $(DEVICE_CUDA_CXX_TARGETS)

clean:
	@echo "###############################################################################"
	@echo "# Cleaning build directory $(BUILD_DIRECTORY)"
	@echo "###############################################################################"
	@rm -f $(BUILD_DIRECTORY)/*
	@if [ -d "$(BUILD_DIRECTORY)" ]; then rmdir $(BUILD_DIRECTORY); fi
	@echo

.PHONY: all clean

$(BUILD_DIRECTORY):
	@mkdir -p $@

% : %.cpp $(BUILD_DIRECTORY)
	@echo "###############################################################################"
	@echo "# Building host executable $(*F) in directory $(BUILD_DIRECTORY)"
	@echo "###############################################################################"
	$(CUDA_CXX) $(HOST_ISO_CXX_FLAGS) $< -o $(BUILD_DIRECTORY)/$(*F) 2>&1 | tee $(BUILD_DIRECTORY)/$(*F).build_log
	@echo

%.cubin : %.cu $(BUILD_DIRECTORY)
	@echo "###############################################################################"
	@echo "# Building device binary $(*F) in directory $(BUILD_DIRECTORY)"
	@echo "###############################################################################"
	$(CUDA_CXX) $(DEVICE_CUDA_CXX_FLAGS) $< -cubin -o $(BUILD_DIRECTORY)/$(*F).cubin 2>&1 | tee $(BUILD_DIRECTORY)/$(*F).cubin.build_log
	@echo

