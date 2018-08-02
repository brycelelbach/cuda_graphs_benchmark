# CUDA Graphs Benchmarks

**Reply-To:** Bryce Adelstein Lelbach <brycelelbach@gmail.com>

This repository contains a series of benchmarks for CUDA Graphs. The
benchmarks are currently focused on measuring **launch latency**, e.g. the
amount of execution time spent in host code asynchronously launch kernels.

## Building

### Native build on *NIX

The following process can be used to compile for *NIX operating systems like Linux:

```
# Check out the codebase locally using Git:
git clone ssh://git@gitlab-master.nvidia.com:12051/cuda_umd/cuda_graphs_benchmarks.git
cd cuda_graphs_benchmarks

# Build the codebase using the system's default host ISO C++ compiler and CUDA
# toolkit installation.
make 

# Run the benchmark once:
./build__release/cuda_graphs_benchmarks
```

The `ISO_CXX` and `CUDA_CXX` variables can be used to build the codebase with a
particular host ISO C++ compiler and/or CUDA toolkit installation:

```
# Check out the codebase locally using Git:
git clone ssh://git@gitlab-master.nvidia.com:12051/cuda_umd/cuda_graphs_benchmarks.git

# Build the codebase using `/path/to/c++` and `/path/to/nvcc`.
cd cuda_graphs_benchmarks
make ISO_CXX=/path/to/c++ CUDA_CXX=/path/to/nvcc

# Run the benchmark once:
./build__release/cuda_graphs_benchmarks
```

### Cross build targeting QNX

The following process can be used to cross compile for QNX:

```
# Check out the codebase locally using Git:
git clone ssh://git@gitlab-master.nvidia.com:12051/cuda_umd/cuda_graphs_benchmarks.git
cd cuda_graphs_benchmarks

# Build the codebase using a QNX GCC cross compiler (`/path/to/qnx/g++`) and
# `/path/to/nvcc`. 
export QNX_ROOT=/path/to/qnx
export QNX_HOST=${QNX_HOST}/host/your-os/your-arch
export QNX_TARGET=${QNX_TARGET}/target/qnx7 
make ISO_CXX=${QNX_ROOT}/path/to/g++ CUDA_CXX=/path/to/nvcc
```

## Running

After a success build, the executables can be found in a folder inside your
local copy of the codebase. By default, this folder is called `build__release`.

**NOTE: It is critical to run the benchmark in a correctly configured environment
to obtain accurate and consistent results.**

Before running the benchmark, ensure that:

- `nvidia-persistenced` is running and persistence mode is enabled.
- Your GPUs clocks are fixed at the default clock speed.
- Your CPU clock frequency is fixed.
  - Intel Turbo Boost or any similar feature is disabled (via BIOS/EFI).
  - Operating system CPU throttling is disabled.
  - Hardware CPU throttling is disabled (via BIOS/EFI)
- Hyper-threading is disabled on your CPU.
- Your operating system kernel is configured to be non-premptive or voluntarily preemptive.
  - Either `CONFIG_PREEMPT_VOLUNTARY` or `CONFIG_PREEMPT_NONE` should be set in your Linux kernel build config.

Running the benchmark without any arguments will conduct the default
experiments and report a summary of the results in a human-readable format.

For more advanced usage, the benchmark accepts a number of command line
options which control which experiments are conducted and what parameters
they are conducted with. To learn more about these options, invoke the benchmark
with the `--help` flag:

``
./build__release/cuda_graphs_benchmarks --help
``

## Prerequisites

The following are required to build this codebase.

- Any operating system that supports CUDA.
- A host ISO C++14 compiler (see below for details).
- CUDA toolkit r10.0 or higher.
- CUDA driver r410.19 or higher.
- GNU Make 4.1 or higher.
- Git 2.11 or higher.
-
### Supported Operating Systems

This codebase should be portable to any operating system that supports CUDA,
although it may not work out of the box.

It has been tested with the following operating systems:

- Debian 9.1 + Linux 4.9.0.

If you find you cannot build or run this software on your platform, please file
an issue.

### Supported Host ISO C++ Compilers

This codebase is written in ISO C++14 for host code (`.cpp` files) and CUDA C++
for device code (`.cu` files).

It has been tested with the following host ISO C++ compilers:

- GCC 8.1 Linux x86-64 native compiler.
- GCC 6.4 Linux x86-64 native compiler.
- GCC 5.5 Linux x86-64 native compiler.
- GCC 5.4 QNX aarch64le cross compiler.

This codebase should be portable to any of the following compilers, although it
may not work out of the box:

- GCC 5.0 and higher.
- Clang 3.4 and higher.
- PGI 8.6 and higher.
- MSVC 2015 and higher.

If you find you cannot build or run this software on your platform, please file
an issue.

## Troubleshooting

If you are encountering problems or have questions while building and running
this software, please file an issue in this GitLab repository.

**[Please follow these guidelines when filing issues. Issues that do not follow these guidelines may not be actionable and thus will not receive attention.](https://github.com/brycelelbach/cpp_bug_reporting_guidelines)**

The Makefile for this software will produce a log file, `build.log` when it is
invoked. When filing issues, you should always include the `build.log` from your
local build.

