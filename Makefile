###############################################################################
# Copyright (c) 2018   NVIDIA Corporation
# Copyright (c) 2015-6 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
#
# Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)
###############################################################################

# Target device architecture (`[0-9][0-9]` or `all`).
DEVICE_ARCH     ?= all
# Class of build (`debug` or `release`).
BUILD_TYPE      ?= release
# The build directory to create and use.
BUILD_DIRECTORY ?= $(ROOT)/build__$(BUILD_TYPE)/

# C++ version (`c++14`, `c++17`, or `c++20`)
CXX_VERSION     ?= c++20

# Host C++ (`.cpp`) -> Host Object File (`.o`) -> Host Executable

# Host ISO C++ compiler (path to executable).
ISO_CXX                 ?= c++
# Flags passed to the host ISO C++ compiler when compiling host ISO C++ code.
ISO_CXX_XCOMPILER_FLAGS ?= -Wall -Wno-sign-compare

# Device CUDA C++ (`.cu`) -> Device Binary (`.cubin`)

# Host/device CUDA C++ compiler (path to executable).
CUDA_CXX       ?= nvcc
# Flags for compiling CUDA C++ code.
CUDA_CXX_FLAGS ?= -rdc=true

###############################################################################

# Clear previous build log, if there is one.
IGNORE := $(shell rm -f build.log)

# Diagnostics macro for use outside of rules. Prints its single argument to
# both stdout and the build log. Note that it has to escape `$`.
define PRINT_CONFIG =
  $(info $(shell echo $(subst $$,\$$,$(1)) | tee -a build.log))
endef

# Diagnostics macro for use within rules. Prints its single argument to both
# stdout and the build log.
define PRINT_RULE =
  @echo $(1) | tee -a build.log
  @$(1) 2>&1 | tee -a build.log
endef

IGNORE := $(call PRINT_CONFIG,"///////////////////////////////////////////////////////////////////////////////")
IGNORE := $(call PRINT_CONFIG,"// Settings")
IGNORE := $(call PRINT_CONFIG,"///////////////////////////////////////////////////////////////////////////////")
IGNORE := $(call PRINT_CONFIG,"// DEVICE_ARCH             : $(value DEVICE_ARCH)")
IGNORE := $(call PRINT_CONFIG,"// BUILD_TYPE              : $(value BUILD_TYPE)")
IGNORE := $(call PRINT_CONFIG,"// BUILD_DIRECTORY         : $(value BUILD_DIRECTORY)")
IGNORE := $(call PRINT_CONFIG,"// CXX_VERSION             : $(value CXX_VERSION)")
IGNORE := $(call PRINT_CONFIG,"// ISO_CXX                 : $(value CXX)")
IGNORE := $(call PRINT_CONFIG,"// ISO_CXX_XCOMPILER_FLAGS : $(value ISO_CXX_XCOMPILER_FLAGS)")
IGNORE := $(call PRINT_CONFIG,"// CUDA_CXX                : $(value CUDA_CXX)")
IGNORE := $(call PRINT_CONFIG,"// CUDA_CXX_FLAGS          : $(value DEVICE_CUDA_CXX_FLAGS)")
IGNORE := $(call PRINT_CONFIG) # Print blank newline.

###############################################################################

# Tell the host/device CUDA C++ compiler which device architectures to generate
# code for when compiling device CUDA C++.
ifeq ($(DEVICE_ARCH),all)
  CUDA_CXX_FLAGS += -gencode arch=compute_70,code=compute_70
else
  CUDA_CXX_FLAGS += -gencode arch=compute_$(DEVICE_ARCH),code=sm_$(DEVICE_ARCH)
endif

ifeq      ($(BUILD_TYPE),release)
  ISO_CXX_XCOMPILER_FLAGS += -O3
else ifeq ($(BUILD_TYPE),debug)
  ISO_CXX_XCOMPILER_FLAGS += -O0
  CUDA_CXX_FLAGS          += -G -O0
endif

CUDA_CXX_FLAGS += -std=$(CXX_VERSION)

# Tell the host/device CUDA C++ compiler which host ISO C++ compiler to use.
CUDA_CXX_FLAGS += -ccbin $(ISO_CXX)

# Add `HOST_ISO_CXX_XCOMPILER_FLAGS` to `CUDA_CXX_FLAGS`.
CUDA_CXX_FLAGS += $(foreach flag,$(ISO_CXX_XCOMPILER_FLAGS),-Xcompiler $(flag))

###############################################################################

# Get the directory where the Makefile and code live.
ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Strip terminating `/`s from directory paths.
# This has to be done late because these inplace updates must be immediate.
ROOT            := $(ROOT:/=)
BUILD_DIRECTORY := $(BUILD_DIRECTORY:/=)

CUDA_CXX_FLAGS += -I$(ROOT)

# Strip leading and trailing whitespace from flags.
CUDA_CXX_FLAGS := $(strip $(CUDA_CXX_FLAGS))

CUDA_CXX_SOURCES = $(wildcard $(ROOT)/*.cu)
CUDA_CXX_TARGETS = $(CUDA_CXX_SOURCES:.cu=)

###############################################################################

IGNORE := $(call PRINT_CONFIG,"///////////////////////////////////////////////////////////////////////////////")
IGNORE := $(call PRINT_CONFIG,"// Computed variables")
IGNORE := $(call PRINT_CONFIG,"///////////////////////////////////////////////////////////////////////////////")
IGNORE := $(call PRINT_CONFIG,"// ROOT             : $(ROOT)")
IGNORE := $(call PRINT_CONFIG,"// BUILD_DIRECTORY  : $(BUILD_DIRECTORY)")
IGNORE := $(call PRINT_CONFIG,"// CUDA_CXX_FLAGS   : $(CUDA_CXX_FLAGS)")
IGNORE := $(call PRINT_CONFIG,"// CUDA_CXX_SOURCES : $(CUDA_CXX_SOURCES)")
IGNORE := $(call PRINT_CONFIG,"// CUDA_CXX_TARGETS : $(CUDA_CXX_TARGETS)")
IGNORE := $(call PRINT_CONFIG) # Print blank newline.

###############################################################################

all: $(CUDA_CXX_TARGETS)

print_environment:
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	@echo '// Host ISO C++ compiler version' | tee -a build.log
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	@$(ISO_CXX) --version 2>&1 | tee -a build.log
	@echo | tee -a build.log
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	@echo '// Host/device CUDA C++ compiler version' | tee -a build.log
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	@$(CUDA_CXX) --version 2>&1 | tee -a build.log
	@echo | tee -a build.log
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	@echo '// Environment' | tee -a build.log
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	@env 2>&1 | tee -a build.log
	@echo | tee -a build.log

clean:
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	@echo '// Cleaning build directory $(BUILD_DIRECTORY)' | tee -a build.log
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	$(call PRINT_RULE,rm -f $(BUILD_DIRECTORY)/*)
	$(call PRINT_RULE,[ -d "$(BUILD_DIRECTORY)" ] && rmdir $(BUILD_DIRECTORY))
	@echo | tee -a build.log

.PHONY: all print_environment clean

$(BUILD_DIRECTORY): print_environment
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	@echo '// Making build directory $(BUILD_DIRECTORY)' | tee -a build.log
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	$(call PRINT_RULE,mkdir -p $@)
	@echo | tee -a build.log

% : %.cu $(BUILD_DIRECTORY)
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	@echo '// Building executable $(*F) in directory $(BUILD_DIRECTORY)' | tee -a build.log
	@echo '///////////////////////////////////////////////////////////////////////////////' | tee -a build.log
	$(call PRINT_RULE,$(CUDA_CXX) $(CUDA_CXX_FLAGS) $< -o $(BUILD_DIRECTORY)/$(*F))
	@echo | tee -a build.log

