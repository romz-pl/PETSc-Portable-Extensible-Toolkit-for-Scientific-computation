# PETSc: The Portable, Extensible Toolkit for Scientific Computation

BOOK: [PETSc for Partial Differential Equations: Numerical Solutions in C and Python](https://epubs.siam.org/doi/book/10.1137/1.9781611976311)

## 1. What PETSc is

[**PETSc**](https://petsc.org) (pronounced "pet-see") is an open-source software library for the parallel solution of large-scale scientific problems. It provides distributed data structures, linear and nonlinear solvers, time integrators, optimization methods, and tools for mesh management and performance analysis. Its primary target is the solution of partial differential equations (PDEs) discretized by finite element, finite volume, finite difference, or spectral methods, although it is equally used for ODEs, DAEs, eigenvalue problems (through the companion package SLEPc), and PDE-constrained optimization (through TAO).

PETSc is developed mainly by the Mathematics and Computer Science Division at Argonne National Laboratory, with contributions from a large international community of universities, national laboratories, and companies. It is distributed under a permissive BSD-style license, so it can be used in both academic and commercial software.

The name reflects its design goals. "Portable" means it runs from laptops to the largest supercomputers, across CPU and GPU architectures. "Extensible" means users can add new data structures, solvers, and preconditioners. "Toolkit" means it is a collection of composable building blocks rather than a single monolithic solver.

## 2. Core design philosophy

Several principles run through the entire library.

**Separation of concerns.** The discretization of a problem (how the mesh is built and how the equations are assembled) is kept distinct from the algorithms that solve the resulting algebraic systems. A finite element code can assemble a sparse matrix and hand it to any Krylov solver and any preconditioner without knowing which one was chosen.

**Runtime selection of algorithms.** Most choices of solver, preconditioner, tolerance, monitor, and output are made through an options database, which is typically populated from the command line. A single compiled executable can be tried with many solver configurations:

```bash
./app -ksp_type gmres -pc_type gamg -ksp_rtol 1e-8 -ksp_monitor
./app -ksp_type cg    -pc_type hypre -pc_hypre_type boomeramg
./app -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps
```

This makes performance tuning and algorithm comparison largely a matter of changing run-time arguments, not recompiling.

**Hierarchical composition.** Solvers contain other solvers. A nonlinear solver (SNES) uses a linear solver (KSP), which uses a preconditioner (PC), which may itself contain sub-solvers, such as the local solves inside an additive Schwarz method or the blocks of a field-split preconditioner. This nesting is exposed uniformly through the same API at every level.

**Parallelism built in from the start.** PETSc is built on MPI. Its data objects are distributed by default, with ownership ranges, ghost points, and scatter contexts managed by the library. Users can write code that looks almost identical in serial and in parallel.

**Opaque objects with a C API.** PETSc's core is written in C using an object-oriented style: handles such as `Vec`, `Mat`, and `KSP` are pointers to opaque structures, and behavior is dispatched through function tables. This gives polymorphism (a `Mat` may be AIJ, dense, matrix-free, or GPU-resident) without requiring C++.

## 3. Language support and interfaces

The primary interface is C. Fortran (77 and 90 with modules), Python (through `petsc4py`), and Julia (through `PETSc.jl`) are maintained bindings. MATLAB and GNU Octave interfaces exist for prototyping. Because the library exposes a C API, C++ codes call it directly; many C++ projects wrap the handles in RAII classes to manage object lifetimes. Using PETSc from C++ is therefore straightforward but requires some discipline with destructors and the `PetscCall` error-checking macros.

Modern PETSc uses `PetscCall(...)` around every library call, which propagates error codes and reports stack traces. Typical code structure is:

```c
PetscCall(PetscInitialize(&argc, &argv, NULL, help));
/* create and assemble objects */
PetscCall(PetscFinalize());
```

## 4. Major components

### 4.1 Vectors (Vec)

`Vec` represents distributed dense vectors. Each MPI process owns a contiguous range of entries. Operations include `VecSet`, `VecAXPY`, `VecDot`, `VecNorm`, and `VecPointwiseMult`. Assembly is done with `VecSetValues` followed by `VecAssemblyBegin` and `VecAssemblyEnd`, which allow communication to overlap with computation. `VecScatter` moves data between differently partitioned vectors, and ghost vectors (`VecCreateGhost`) provide local copies of off-process entries needed for stencil operations.

Vector types include standard serial and MPI types, CUDA, HIP, SYCL, and Kokkos variants, and a `VECNEST` type for block-structured composition.

### 4.2 Matrices (Mat)

`Mat` is the most heavily developed type. Its main variants include:

- **AIJ (compressed sparse row):** the default sparse format, with MPI variants that split each process's rows into diagonal and off-diagonal blocks.
- **BAIJ (block AIJ):** stores dense blocks, efficient for systems with multiple unknowns per node.
- **SBAIJ (symmetric block AIJ):** stores only the upper triangle for symmetric problems.
- **Dense:** serial and distributed dense matrices.
- **MATSHELL:** a user-defined matrix that only provides a matrix-vector product. Essential for matrix-free Krylov methods.
- **MATNEST:** a block matrix made of submatrices, used for multiphysics problems.
- **MATIS:** an "unassembled" format for domain decomposition methods.
- **MATH2OPUS, MATHTOOL, and related types:** hierarchical and low-rank structured formats.
- **GPU types:** `MATAIJCUSPARSE`, `MATAIJHIPSPARSE`, `MATAIJKOKKOS`, and others, which keep data and kernels on the accelerator.

Matrices are assembled with `MatSetValues` (or the blocked variant `MatSetValuesBlocked`), and then finalized with `MatAssemblyBegin` and `MatAssemblyEnd`. Preallocation (`MatXAIJSetPreallocation`) is strongly recommended for performance, because it avoids repeated reallocation during insertion. PETSc also supports matrix-free differentiation through `MATMFFD`, which approximates Jacobian-vector products by finite differences of the residual.

### 4.3 Index sets and scatters (IS, VecScatter, PetscSF)

Index sets (`IS`) describe subsets of global indices, used for extracting submatrices, defining field splits, and specifying scatter patterns. `PetscSF` (star forest) is a lower-level abstraction for graph-based communication used by much of the mesh and data management infrastructure.

### 4.4 Distributed meshes and discretizations (DM)

The `DM` abstraction describes the layout of data over a grid or mesh. Its specializations include:

- **DMDA:** structured Cartesian grids for finite difference and finite volume codes, with stencil width and boundary handling built in.
- **DMPlex:** unstructured meshes represented as directed acyclic graphs of points (cells, faces, edges, vertices). DMPlex can read and write common mesh formats, distribute meshes across processes with partitioners (ParMETIS, PT-Scotch, and others), refine and coarsen, and assemble finite element systems through `PetscFE` and `PetscFV` objects. It is the basis for many FEM frameworks.
- **DMStag:** staggered grids, used in fluid and geophysical solvers.
- **DMSwarm:** particle data, for particle-in-cell and Lagrangian methods.
- **DMNetwork:** graphs and networks, such as power grids, pipe networks, and circuits.
- **DMForest:** an interface to p4est for adaptive, octree-based meshes.

Together, these let a solver developer rely on PETSc for parallel mesh handling, rather than writing partitioning, ghost exchange, and output code from scratch.

### 4.5 Linear solvers (KSP)

The `KSP` object solves $Ax = b$. It supports a wide range of Krylov subspace methods. A Krylov method builds an approximate solution from the subspace

$$
\mathcal{K}_k(A, r_0) = \operatorname{span}\{r_0, A r_0, A^2 r_0, \dots, A^{k-1} r_0\},
$$

where $r_0 = b - A x_0$ is the initial residual. Methods in PETSc include:

- **CG** (conjugate gradients) for symmetric positive definite systems,
- **GMRES** and its restarted and flexible variants (`fgmres`) for general systems,
- **BiCGStab**, **TFQMR**, **MINRES**, **SYMMLQ**, **LSQR**, and **CGLS**,
- **Chebyshev** and **Richardson** iterations, often used as smoothers in multigrid,
- **preonly**, which applies the preconditioner once and is used with direct solvers.

Left, right, and symmetric preconditioning are all supported. The solver is configured with:

$$
M^{-1} A x = M^{-1} b \quad \text{(left preconditioning)},
$$

where $M$ is a preconditioner that approximates $A$ cheaply. Convergence is controlled by relative, absolute, and divergence tolerances, and by maximum iteration counts. Monitors (`-ksp_monitor`, `-ksp_monitor_true_residual`) and convergence reasons (`-ksp_converged_reason`) provide diagnostic output.

### 4.6 Preconditioners (PC)

The `PC` object is where most numerical effort in practical codes goes. PETSc offers:

- **Simple methods:** Jacobi, SOR, SSOR, and point or block variants.
- **Incomplete factorizations:** ILU(k), ICC, and ILUTP-style methods.
- **Domain decomposition:** block Jacobi, additive Schwarz (ASM, with restricted and basic variants), and balancing domain decomposition by constraints (BDDC). PCHPDDM provides a high-performance interface for Schwarz methods with coarse space constructions.
- **Multigrid:** geometric multigrid (`PCMG`), where the user supplies grid hierarchies, and algebraic multigrid through `PCGAMG` (smoothed aggregation) and an interface to BoomerAMG from hypre.
- **Direct solvers:** LU and Cholesky through external packages including MUMPS, SuperLU_DIST, PARDISO (MKL), and PaStiX, used either as complete solvers (with `preonly`) or as coarse-grid and subdomain solvers.
- **Field-split (block) preconditioners:** `PCFIELDSPLIT` decomposes a block system, such as a Stokes or Navier-Stokes system with velocity and pressure unknowns, into sub-solves using Schur complement approximations. This is central to multiphysics applications.
- **Composite preconditioners:** `PCCOMPOSITE` combines several preconditioners, and `PCSHELL` lets users provide their own.

Choosing the right preconditioner is usually the largest determinant of performance and scalability. PETSc's modular design makes it practical to experiment with many options and to combine them hierarchically.

### 4.7 Nonlinear solvers (SNES)

The `SNES` object solves $F(x) = 0$ for a nonlinear residual function $F$. The default method is Newton's method with line search or trust region globalization. Each Newton step solves a linear system with the Jacobian $J(x) = \partial F / \partial x$, using a KSP and PC. PETSc also provides:

- quasi-Newton and nonlinear Krylov methods such as NGMRES and Anderson acceleration,
- nonlinear additive Schwarz (NASM) and nonlinear preconditioning,
- the full approximation scheme (FAS) for nonlinear multigrid,
- matrix-free Jacobians through `MATMFFD`, and automatic checking of user-provided Jacobians with `-snes_test_jacobian`.

### 4.8 Time integration (TS)

The `TS` object integrates $\dot{u} = F(t, u)$ or implicit forms $G(t, u, \dot{u}) = 0$. It supports explicit Runge-Kutta methods, implicit methods including backward Euler, theta methods, Crank-Nicolson, and BDF, additive and implicit-explicit (IMEX) splittings, and adaptive time-stepping. Discrete and continuous adjoints are available for sensitivity analysis and optimization, which makes TS useful for PDE-constrained inverse problems. Pseudo-transient continuation is available to drive steady-state problems toward equilibrium.

### 4.9 Optimization (TAO)

The Toolkit for Advanced Optimization provides algorithms for unconstrained minimization, bound-constrained problems, least squares, and PDE-constrained optimization. TAO integrates with SNES and TS for gradient and Hessian information, and with the rest of PETSc for linear algebra.

### 4.10 Eigenvalue problems (SLEPc)

SLEPc, the Scalable Library for Eigenvalue Problem Computations, is a separate project built on PETSc. It provides Krylov-Schur, Arnoldi, Lanczos, and Jacobi-Davidson methods for standard and generalized eigenproblems, singular value decomposition, and nonlinear eigenproblems. Users who need eigenvalues almost always use SLEPc alongside PETSc.

### 4.11 Viewers, I/O, and visualization

`PetscViewer` abstracts output. It can write binary vectors and matrices, ASCII files for debugging, HDF5 files for checkpointing and postprocessing, and VTK or XDMF output for visualization in ParaView and VisIt. Because most objects can be viewed with `-vec_view`, `-mat_view`, or `-ksp_view`, inspecting the state of a computation requires no additional code.

### 4.12 Logging and performance analysis

PETSc includes built-in profiling. Running with `-log_view` produces a detailed table of time, flop rates, message counts, and memory use per event (for example, `MatMult`, `KSPSolve`, `PCSetUp`, and `VecDot`), broken down by stage and by MPI rank. User code can define its own stages and events with `PetscLogStageRegister` and `PetscLogEventRegister`. This makes PETSc a practical tool for scaling studies and for identifying communication bottlenecks.

## 5. GPU and accelerator support

PETSc supports NVIDIA GPUs through CUDA, AMD GPUs through HIP, and cross-platform execution through Kokkos and SYCL. Vector and matrix types with device-resident data (for example, `VECCUDA`, `MATAIJCUSPARSE`, `VECKOKKOS`) keep operations on the accelerator, and many solvers and preconditioners have GPU-capable implementations. Coverage varies by algorithm and backend, so it is important to check the documentation and the release notes for a given configuration. Selecting a device type is done through the options database and the default vector and matrix types, so application code can remain largely unchanged between CPU and GPU runs.

## 6. Installation and configuration

PETSc is built with a Python-based configure system and `make`. A typical installation looks like this:

```bash
./configure --with-debugging=0 --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 \
    --download-hypre --download-superlu_dist --download-mumps \
    --download-metis --download-parmetis --download-scalapack \
    --download-fblaslapack
make PETSC_DIR=$PWD PETSC_ARCH=arch-optimized all
make PETSC_DIR=$PWD PETSC_ARCH=arch-optimized check
```

The `--download-*` options let configure fetch and build external packages automatically, which is useful for portable installations. On clusters, most sites provide preinstalled PETSc modules, and user builds often link against the site's MPI and BLAS/LAPACK. Separate `PETSC_ARCH` directories allow debug and optimized builds to coexist. Debug builds include extensive runtime checks and are recommended during development.

## 7. Typical usage pattern

A minimal linear solve follows a standard sequence: initialize, create and assemble the matrix and vectors, configure the solver, solve, inspect the result, and clean up.

```c
#include <petscksp.h>

int main(int argc, char **argv)
{
  Mat A;
  Vec x, b;
  KSP ksp;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

  /* Create and assemble A and b (omitted) */

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));   /* reads -ksp_type, -pc_type, ... */
  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return 0;
}
```

The key point is `KSPSetFromOptions`: the same binary can run with an iterative solver, a direct solver, or a multigrid preconditioner, chosen at launch time.

For a nonlinear or time-dependent application, the same pattern applies one level up: the user supplies residual and Jacobian callbacks to `SNES`, or a right-hand side to `TS`, and the library drives the iteration, calling back into user code only where needed.

## 8. Ecosystem and users

PETSc is a foundational dependency for many scientific codes. Examples include the MOOSE framework for multiphysics simulation, the Firedrake and FEniCSx/DOLFINx finite element systems, deal.II, libMesh, and a wide range of geophysics, fusion, climate, and CFD codes built on top of PETSc's data structures and solvers. Its companion projects, SLEPc and TAO, and its bindings, petsc4py and PETSc.jl, extend its reach into Python and Julia workflows.

The community communicates through the `petsc-users` mailing list, which is active and responsive, and through the project's issue tracker and documentation at petsc.org. Documentation includes a user manual, an extensive set of manual pages for every public function, and a large collection of tutorial examples in the source tree. The source is hosted on GitLab.

## 9. Strengths

- **Scalability.** PETSc has been demonstrated on very large machines, and its distributed data structures and communication patterns are designed for that regime.
- **Breadth.** Few libraries provide such a complete stack from vectors to time integrators to optimization, with interfaces to many third-party solvers.
- **Flexibility without recompilation.** Runtime control lets users explore solver space rapidly.
- **Maturity.** The library has been under continuous development for decades, with careful attention to portability and numerical robustness.
- **Diagnostics.** Monitoring, viewing, and logging tools are built in and consistent across the library.
- **Interoperability.** Its common data model makes it a convenient backbone for coupling different solvers and discretization packages.

## 10. Limitations and practical considerations

- **Learning curve.** The number of options and concepts is large. New users often struggle with the interplay between preconditioners, options prefixes, and the assembly model. Good starting points are the tutorials and the "PETSc solver tuning" guidance in the documentation.
- **Verbosity of the C API.** Writing code directly against the C interface involves many calls and explicit error checking. Wrapping it in higher-level abstractions is common in larger projects.
- **Build complexity.** Configuring a full installation with many external packages can be time-consuming, and version compatibility between PETSc and its dependencies requires attention.
- **Performance tuning is problem-dependent.** No single solver is best for all problems. Achieving good performance requires understanding the spectral and structural properties of the matrix, and PETSc exposes the tools to do so rather than making the choice automatically.
- **GPU coverage is uneven.** Support is extensive but not uniform across all algorithms and backends, so verification for a specific configuration is advisable.
- **Version changes.** Options and defaults evolve across releases. Code and command lines should be checked against the documentation for the version in use, and the current release series should be confirmed on the official website, since the library is updated regularly.

## 11. Summary

PETSc is a comprehensive, MPI-based library for the scalable solution of scientific problems, centered on distributed vectors, matrices, and mesh objects, and on a composable hierarchy of linear, nonlinear, time-stepping, and optimization solvers. Its key contribution is the combination of portability, extensibility, and runtime configurability: the same application code can be run with many algorithms, on many machines, and with many backends, while the library handles the parallel bookkeeping. For scientists and engineers who work on large PDE-based simulations, it is one of the most important pieces of numerical infrastructure in the high-performance computing ecosystem.
