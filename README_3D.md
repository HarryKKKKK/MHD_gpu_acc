# Compressible Euler CPU/GPU/MPI solver

This repository now solves only the ideal compressible Euler equations. A cell
stores exactly five conserved variables:

```text
(rho, rho*u, rho*v, rho*w, E)
```

There are no magnetic fields, induction equations, divergence-cleaning scalar,
cleaning waves, or damping passes. The CUDA 2D path uses a five-array SoA
layout; the 3D CPU/CUDA paths use a 40-byte five-double state.

Available Riemann solvers are HLL, HLLC, and FORCE. HLLC is the default because
it resolves the Euler contact wave without carrying the much larger
magnetohydrodynamic wave fan.

## Build

```bash
make cpu cpu_serial cpu_3d
make gpu gpu_3d
make mpi mpi_3d
```

## Run

Two-dimensional Euler cases:

```bash
./bin/main_cpu blast_wave --solver hllc --no-out
./bin/main_cpu kelvin_helmholtz --solver hllc --no-out
./bin/main_cpu shock_bubble --solver hllc --no-out
./bin/main_gpu blast_wave --solver hllc --no-out
```

Three-dimensional spherical Euler blast:

```bash
./bin/main_cpu_3d --resolution 64 --solver hllc --no-out
./bin/main_gpu_3d --resolution 128 --solver hllc --no-out
mpirun -np 8 ./bin/main_mpi_3d --resolution 128 --solver hllc --no-out
```

The CPU and GPU timing lines report `Mcell_updates_s`. For a meaningful
MHD-to-Euler acceleration comparison, use the same grid, precision, compiler
flags, device, CFL, and measurement window. Euler can take a different number
of timesteps because its signal speed no longer contains magnetic waves, so
compare both elapsed time and cell-updates per second.

## GPU profiling on CSD3

The profiling job builds an instrumented 2D CUDA binary, runs five unprofiled
timing repetitions, captures an Nsight Systems timeline, and profiles one
steady-state `advance_x` and `advance_y` launch with Nsight Compute:

```bash
mkdir -p logs
sbatch --export=ALL,N=4,CASE=blast_wave,SOLVER=hllc \
  scripts/csd3_slurm/profile_gpu_systems.sh
```

Results are written below `profiling/euler_JOBID_CASE_SOLVER_nN`. Reduce
`NCU_SET` to `basic` and the step counts for a quick smoke test; the script
header documents all supported overrides.

## 3D snapshot format

The CPU 3D driver writes `.euler3d` files with magic `EUL3D01`. The header
contains dimensions, bounds, time, and gamma, followed by five float32
primitive fields per x-fastest active cell:

```text
rho, p, u, v, w
```
