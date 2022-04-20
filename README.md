# dg-on-cuda

A CUDA/C++ implementation of the Discontinuous Galerkin method as presented in the book:

**Nodal Discontinuous Galerkin Methods - Algorithms, Analysis, and Applications, Jan S. Hesthaven and Tim Warburton, Springer, 2008**

The code, built by **g++ 7.5.0** and **CUDA 10.2**, was tested on **NVIDIA Volta GPU** (CUDA Capability 7.2). It can be configured to run on GPU or CPU-only by the flag **`USE_CPU_ONLY`** defined in `/src/config.h`.

The code adopts the "one-thread-per-element" strategy for parallelization - each thread performs computations for a single element. Layouts of degree-of-freedoms (**DOF**s) in the solution vectors promote coalesced memory access. CUDA constant memory is used to store small-sized, but frequently accessed, data to improve performance.

### Usage

The code contains only header files in the `/src` folder and there is no third-party dependencies except CUDA. A few example problems are provided in the `/examples` folder. To build and run these examples, go to the individual subfolder and run **`make`**. If your CUDA installation directory is different from `/usr/local/cuda-10.2`, change the path of **`CUDA_PATH`** accordingly in the **`makefile`**.

In the subfolder of each example, there is a PDF file describing the problem, the boundary and initial conditions, the analytical solutions (if any), the meshes, and the performance results.

### TO DO

Example problems:

 - Stokes equation in 2D
 - Incompressible Navier-Stokes equation in 2D

3D problems will be covered by a future repository (`flux-reconstruction-schemes`).
