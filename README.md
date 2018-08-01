# CUDA Graphs Benchmarks

**Reply-To:** Bryce Adelstein Lelbach <brycelelbach@gmail.com>

This repository contains a series of benchmarks for CUDA Graphs. The
benchmarks are currently focused on measuring **launch latency**, e.g. the
amount of execution time spent in host code asynchronously launch kernels.

## Getting Started

### Building

```
# Check out the codebase locally using Git:
git clone ssh://git@gitlab-master.nvidia.com:12051/cuda_umd/cuda_graphs_benchmarks.git

# Build the codebase using the system's default host ISO C++ compiler and CUDA
# toolkit installation.
cd cuda_graphs_benchmarks
make 

# Run the benchmark once:
./build__release/cuda_graphs_benchmarks
```

The `ISO_CXX` and `CUDA_CXX` variables can be used to build the codebase with a
particular host ISO C++ compiler and/or CUDA toolkit installation:

```
# Check out the codebase locally using Git:
git clone ssh://git@gitlab-master.nvidia.com:12051/cuda_umd/cuda_graphs_benchmarks.git

# Build the codebase using the system's default host ISO C++ compiler and CUDA
# toolkit installation.
cd cuda_graphs_benchmarks:
make ISO_CXX=/path/to/c++ CUDA_CXX=/path/to/nvcc

# Run the benchmark once:
./build__release/cuda_graphs_benchmarks
```

The executables will be built in a folder inside your local copy of the codebase.
By default, this folder is called `build__release`.

### Running

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

This codebase should work on any operating system that supports CUDA.

It has been tested with the following operating systems:

- Debian 9.1 + Linux 4.9.0.

### Supported Host ISO C++ Compilers

This codebase is written in ISO C++14 for host code (`.cpp` files) and CUDA C++
for device code (`.cu` files).

It has been tested with the following host ISO C++ compilers:

- GCC 8.1.
- GCC 6.4.
- GCC 5.5.

This codebase should work with the following compilers:

- GCC 5.0 and higher.
- Clang 3.4 and higher.
- PGI 8.6 and higher.
- MSVC 2015 and higher.
  - **NOTE:** This may not work out of the box.

