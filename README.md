# HPC Poisson laboratory

[![Verify project](https://github.com/Alexkode/hpc-poisson/actions/workflows/verify.yml/badge.svg)](https://github.com/Alexkode/hpc-poisson/actions/workflows/verify.yml)

A reproducible study of a 3D finite-difference Poisson solver, covering numerical
correctness, CPU threading, distributed-memory communication and GPU execution.

## Mathematical reference

Solve `-Δu = f` on the unit cube with homogeneous Dirichlet boundaries, where
`u(x,y,z) = sin(πx) sin(πy) sin(πz)` and `f = 3π²u`.
The continuous reference makes the discretization error observable. The same sine
mode is also a known eigenvector of the discrete operator, giving an independent
prediction of the residual after a prescribed number of weighted Jacobi sweeps.

The CPU implementation supports rectangular grids, uneven decomposition along
the z axis, MPI halo exchange, OpenMP, and overlapping communication with interior
updates. The CUDA implementation uses a resident FP64 field on one GPU. It is a
baseline GPU kernel; it does not implement multi-GPU communication. The mixed CPU
variant stores fields in FP32 and performs updates and reductions in FP64.

## Run locally

Install the Python dependencies with `python -m pip install -r requirements.txt`.

```bash
make
./build/poisson --n 24 --tolerance 1e-7
python -m pytest tests -q
python scripts/benchmark.py --n 96 --iterations 500
```

`--tolerance 0` runs a fixed number of sweeps for performance comparisons.
Convergence checks are included in the measured time when a tolerance is enabled.
Initialization, warm-up and final validation are excluded. CPU times are measured
with a monotonic wall clock and reduced using the slowest MPI rank. CUDA times use
events on the default stream. Do not compare fixed-sweep throughput with
time-to-solution without stating the distinction.

## Cluster execution

```bash
mkdir -p outputs results
sbatch scripts/cluster_cpu.sbatch
sbatch scripts/cluster_gpu.sbatch
```

The scripts target the CESGA 2025 stack and the `short` partition. CESGA requires
32 CPU cores per allocated A100 GPU. Adjust partition, modules and allocation
parameters for another system. No heavy work should run on a login node.

For another NVIDIA GPU, set its architecture explicitly, for example
`make cuda CUDA_ARCH=80`. An architecture supporting FP64 atomic operations is
required. `TEST_CUDA=1 python -m pytest tests -q` enables numerical CUDA validation.

## Experiments and interpretation

Strong scaling holds the global 128³ grid and 500 sweeps fixed. Weak scaling uses
64×64×(64×ranks), so each rank owns the same number of points; it measures sweep
throughput, not equivalent convergence behavior. Three repetitions are retained.
Report median times and the spread, speedup relative to one rank, and efficiency
`speedup/ranks`. GPU comparisons use the same 128³ problem and sweep count.

`scripts/cluster_precision.sbatch` measures OpenMP threads 1/2/4/8 and compares FP64 with FP32 field storage at residual tolerance 1e-5. It compiles with GCC independently of MPI. The current CESGA MPI wrapper may require an NVIDIA management library on some CPU nodes; adapt cluster modules if it is unavailable. The recorded MPI measurements and GitHub MPI tests succeeded.

Results must retain hardware and software provenance. Raw JSON files, the numerical
verification and plots belong in `results/`. Missing measurements are not replaced
by estimated timings. Nsight traces are optional and must be collected separately
from uninstrumented performance runs.

## What this project demonstrates

- A numerical reference and second-order grid convergence.
- Correct communication across uneven MPI domains.
- The distinction between precision, residual and discretization error.
- Measured scaling, including cases where additional resources do not help.

Created with AI assistance. The experimental records document what was actually run.

## Installation and verification

Clone this repository and run commands from its root. Each repository has its own dependencies and tests.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/verify.sh
```

GitHub Actions checks the code contracts and the consistency of published experimental records. It does not rerun training or cluster benchmarks. See [experiment provenance](ENVIRONMENT.md) and [measured report](results/REPORT.md).

A C++17 compiler is required. `make openmp` and `make mpi` enable the optional CPU backends; GitHub Actions builds both. CUDA tests require an allocated GPU and are optional.
