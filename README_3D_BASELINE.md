# Unoptimised 3D GPU baseline

This target is the performance baseline for the 3D optimisation study. It
supports the same true-3D cases and configurations as the `3D_task` branch:
`blast` and `imtg`.

- `blast`: Athena-style spherical magnetized blast on `[-0.5,0.5]^3`,
  `gamma=5/3`, `CFL=0.20`, and `t_end=0.10`.
- `imtg`: Ms0.2_Ma1 insulating magnetic Taylor-Green case on
  `[-0.5,0.5]^3`, `gamma=5/3`, `CFL=0.20`, and
  `t_end=5.809475019311126`.

The initial-condition constants and formulae are copied from `3D_task`.

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
./bin/main_gpu_3d_baseline --case blast --solver hlld \
  --resolution 32 --max-steps 2 --output
```

Full benchmark at the case's standard resolution:

```bash
./bin/main_gpu_3d_baseline --case blast --resolution 128 \
  --solver hlld --cfl 0.20 --t-end 0.10 --snapshots 5 --no-out
./bin/main_gpu_3d_baseline --case imtg --resolution 128 \
  --solver hlld --cfl 0.20 --t-end 5.809475019311126 \
  --snapshots 5 --no-out
```

The literature IMTG reference resolution is `1024^3`, but four
double-precision state grids do not fit on one GPU at that size. The executable
checks free device memory before allocating. Use `128^3` for direct comparison
with the existing `3D_task` timing table.

For validation, `--output` writes the final center-z slice as CSV.

On the DGX cluster, the supplied job first builds the target and runs a
two-step smoke test for both cases before starting measurement:

```bash
sbatch scripts/dgx_slurm/slurm_gpu_3d_baseline.sh

# Example report sweep:
RESOLUTIONS_STR="64 128" SOLVERS_STR="hlld" \
  sbatch scripts/dgx_slurm/slurm_gpu_3d_baseline.sh
```

Set `MAX_STEPS` to a common positive value when comparing kernel throughput
independently of case-dependent timestep counts. Leave it at `0` for
end-to-end time-to-solution.

For CSD3 Ampere:

```bash
mkdir -p logs
sbatch scripts/csd3_slurm/slurm_gpu_3d_baseline.sh

export RESOLUTIONS_STR="64 128" SOLVERS_STR="hlld"
sbatch --export=ALL,RESOLUTIONS_STR,SOLVERS_STR \
  scripts/csd3_slurm/slurm_gpu_3d_baseline.sh
```

The CSD3 script uses `rhel8/default-amp`, detects the allocated GPU's compute
capability, builds into job-specific directories, runs both smoke cases, and
writes its CSV to `timing/gpu_3d_baseline/`. It also writes a compact
`case,backend,resolution,workers,elapsed_s` CSV that can be appended directly
to the existing `3D_task` timing table.
