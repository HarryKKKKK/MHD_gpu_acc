# Unoptimised 3D GPU baseline

This target is the performance baseline for the 3D optimisation study. It
supports the same two benchmark cases used by the DGX optimisation branches:
`orszag_tang` and `rotor`.

Both are extended uniformly in z. The initial state is identical on every
z-plane, the z boundary is periodic, and the solver still stores and advances
the complete `N x N x N` state through x, y, and z MUSCL-Hancock sweeps. This
keeps the physical benchmark comparable with the 2D branches while exercising
a real 3D implementation.

The baseline intentionally uses:

- one AoS global-memory state array per grid;
- one thread per updated cell;
- direct global-memory stencil loads;
- runtime axis and Riemann-solver selection;
- separate x, y, and z kernels;
- no shared-memory tile, face cache, launch bounds, or solver specialization.

Build:

```bash
make gpu_3d_baseline
```

Short correctness/smoke run:

```bash
./bin/main_gpu_3d_baseline --case orszag_tang --solver hlld \
  --resolution 32 --max-steps 2 --output
```

Full benchmark at the case's standard resolution:

```bash
./bin/main_gpu_3d_baseline 1 --case orszag_tang --solver hlld --no-out
./bin/main_gpu_3d_baseline 1 --case rotor --solver hlld --no-out
```

`rotor` at its standard `400^3` resolution requires roughly 20 GiB. The
executable checks free device memory before allocating. Unlike the 2D scripts,
do not use a default `n=1,2,4` sweep for 3D; choose resolutions that fit and
report the exact `nx`, `ny`, `nz`, and active-cell count.

For validation, `--output` writes the final center-z slice as CSV. Because the
initial condition is z-invariant, this slice can be compared directly with the
existing 2D result at the same resolution, solver, CFL, and stopping time.

On the DGX cluster, the supplied job first builds the target and runs a
two-step smoke test for both cases before starting measurement:

```bash
sbatch scripts/dgx_slurm/slurm_gpu_3d_baseline.sh

# Example report sweep:
RESOLUTIONS_STR="64 96 128" SOLVERS_STR="hlld" \
  sbatch scripts/dgx_slurm/slurm_gpu_3d_baseline.sh
```

Set `MAX_STEPS` to a common positive value when comparing kernel throughput
independently of case-dependent timestep counts. Leave it at `0` for
end-to-end time-to-solution.

For CSD3 Ampere:

```bash
mkdir -p logs
sbatch scripts/csd3_slurm/slurm_gpu_3d_baseline.sh

export RESOLUTIONS_STR="64 96 128" SOLVERS_STR="hlld"
sbatch --export=ALL,RESOLUTIONS_STR,SOLVERS_STR \
  scripts/csd3_slurm/slurm_gpu_3d_baseline.sh
```

The CSD3 script uses `rhel8/default-amp`, detects the allocated GPU's compute
capability, builds into job-specific directories, runs both smoke cases, and
writes its CSV to `timing/gpu_3d_baseline/`.
