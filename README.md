# dg-on-cuda
A CUDA/C++ implementation of the Discontinuous Galerkin method.

Code under construction...

System requirements: g++ version supporting C++17; CUDA version 10.2 or later.

Code desing: one-thread-per-cell vs one-thread-per-DOF (degree-of-freedom)...

Usage: No installation needed; to build and run examples, go to the subfolders of /examples and run the make file.

Can be configured to run on GPU or CPU-only by the macro USE_CPU_ONLY defined in /src/config.h. Code tested on GPU with CUDA capability 7.2.
