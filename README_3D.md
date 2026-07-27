# 3D CPU solver and visualization

The original 2D CPU/GPU/MPI paths are unchanged. The 3D baseline is a separate
CPU executable using the same nine-component GLM-MHD state and Riemann solvers.
It adds a ghosted `Grid3D`, all six boundary faces, three-direction CFL control,
and MUSCL-Hancock x/y/z sweeps.

Build and run the display case:

```bash
make cpu_3d
./bin/main_cpu_3d --resolution 48 --snapshots 5 --out output/blast3d
python visualization/plot_blast3d.py --input output/blast3d
python visualization/plot_blast3d_volume.py --input output/blast3d
```

On Windows PowerShell:

```powershell
make cpu_3d
.\bin\main_cpu_3d.exe --resolution 48 --snapshots 5 --out output/blast3d
python .\visualization\plot_blast3d.py --input output/blast3d
python .\visualization\plot_blast3d_volume.py --input output/blast3d
```

The default `--case blast` is a moderate, three-dimensional extension of the
Athena magnetized blast-wave test. On `[-0.5,0.5]^3`, it uses `rho=1`, zero
velocity, `gamma=5/3`, and
`B=(1/sqrt(2),1/sqrt(2),0)`. Pressure is 10 for `r<=0.09`, 0.1 for
`r>=0.10`, and linearly interpolated in between. All boundaries are periodic
and the default output time is `t=0.10`. These parameters live in
`head/blast3d_case.hpp` and are shared by the CPU and CUDA drivers.

The former Derigs et al. Sec. 5.6 setup is retained as
`--case blast_extreme`. It uses pressure 1000/0.1,
`B=(100/sqrt(4*pi),0,0)`, `gamma=1.4`, and `t_end=0.01`. It is an
intentionally stringent positivity test and may not complete until the solver
has an updated-cell positivity fallback. Use `--field pressure` to render
pressure instead of density.

`plot_blast3d_volume.py` produces a genuine three-dimensional voxel surface
rather than planar slices. By default it writes several fixed-camera PNGs at
different physical times and one combined evolution overview. The visible surface is selected by
`abs(field - far_field) >= fraction * global_max_deviation`; adjust it with
`--fraction 0.1`, or specify an absolute `--level`. GIF output is disabled by
default; it is available only when explicitly requested with `--rotation-gif`
or `--evolution-gif`.

For publication layout, first iterate on an empty six-panel preview. This does
not read snapshots or draw voxels:

```bash
python3 visualization/plot_blast3d_volume.py \
  --layout-preview --preview-case imtg --field Bmag \
  --paper --overview-only --paper-png-only --paper-dpi 150
```

The preview is written under `figs/layout_preview`. Once the layout is
approved, render the same layout with real data:

```bash
python3 visualization/plot_blast3d_volume.py \
  --input outputs/imtg3d_gpu_n128_hlld_JOBID \
  --field Bmag --fraction 0.15 --stride 1 --png-frames 6 \
  --paper --overview-only --paper-dpi 300
```

Paper mode uses a 7.2-inch two-column canvas, white background, `(a)`--`(f)`
panel labels, normalized IMTG time `t/T`, mathematical field labels, one global
color scale, and writes both a 300-dpi PNG and a compact rasterized PDF.

For the moderate blast, the recommended paper figure is a single `2x3`
summary: density at three times in the top row and signed
magnetic-magnitude perturbations at the same times in the bottom row.
The `dBmag` field is defined as
`|B| - median_boundary(|B|)` for each snapshot. Its color scale is symmetric
about zero and independent of the density color scale, so magnetic
amplification and depletion remain visually distinct. The visibility mask
uses the perturbation from the non-zero far-field `|B0|`, so it no longer
fills the complete cube:

```bash
python3 visualization/plot_blast3d_paper.py \
  --input outputs/blast3d_gpu_n128_hlld_JOBID \
  --min-time 0.02 --panels 3 --stride 1 --dpi 300
```

The dedicated rendering Slurm script selects this combined figure
automatically for a blast when `PAPER=1`.

## Slurm

Submit the generic one-node OpenMP job from the repository root. Supply the
account and partition on the `sbatch` command because these are cluster
specific:

```bash
mkdir -p logs
sbatch -A YOUR_ACCOUNT -p YOUR_PARTITION scripts/cpu/slurm_cpu_3d.sh
```

For a 96-cubed presentation run on 64 CPU cores:

```bash
sbatch -A YOUR_ACCOUNT -p YOUR_PARTITION \
  --cpus-per-task=64 --mem=32G --time=08:00:00 \
  --export=ALL,RESOLUTION=96,SNAPSHOTS=10,VISUALIZE=1 \
  scripts/cpu/slurm_cpu_3d.sh
```

Useful exported variables are `CASE`, `RESOLUTION`, `T_END`, `SNAPSHOTS`,
`SOLVER`, `CFL`, `OUT_DIR`, `OMP_THREADS`, `FIELD`, `FRACTION`,
`PNG_FRAMES`, `PLOT_STRIDE`,
`VISUALIZE`, `PYTHON_BIN`, and `MODULES_STR`. If Python rendering libraries are
not installed on compute nodes, use `VISUALIZE=0` and run the visualization
script later against the generated snapshot directory.

### True CUDA 3D on CSD3

The CUDA backend has its own `main_gpu_3d` executable; it does not reserve a
GPU merely to run the OpenMP solver. The CSD3 script builds this executable,
runs an 8-cubed CPU/GPU one-step HLLD parity check, and only then starts the
production simulation:

```bash
mkdir -p logs
sbatch scripts/csd3_slurm/slurm_gpu_3d.sh
```

The default is a 128-cubed HLLD run of the moderate Athena case to `t=0.10`,
with five output intervals (six files including `t=0`) on one Ampere GPU; the
default CFL is 0.20. Select the preserved stress test with
`--export=ALL,CASE=blast_extreme`.
For a larger run without rendering on the compute node:

```bash
sbatch --export=ALL,RESOLUTION=192,SNAPSHOTS=8,VISUALIZE=0 \
  scripts/csd3_slurm/slurm_gpu_3d.sh
```

Build and test interactively on a CUDA node with:

```bash
module purge
module load rhel8/default-amp
make gpu_3d
make test_gpu_3d
```

### OpenMP/MPI/GPU 3D timing comparison

The pure-MPI 3D executable uses a periodic z-slab decomposition with two
exchanged ghost planes per rank:

```bash
make mpi_3d
mpirun -np 8 ./bin/main_mpi_3d \
  --case blast --resolution 64 --solver hlld --no-out
```

On CSD3, one script builds the OpenMP, pure-MPI and CUDA 3D executables and
runs both `blast` and `imtg` exactly once per backend on the same Ampere node:

```bash
mkdir -p logs
sbatch scripts/csd3_slurm/slurm_compare_3d_backends.sh
```

The default comparison uses `64^3`, eight OpenMP threads, eight MPI ranks and
one GPU. Set `RESOLUTION=128` for the larger comparison. Results are written
under `timing/compare3d_JOBID`, including raw logs, `backend_times.csv`, and
`gpu_speedup_summary.csv`. The summary reports `T_CPU/T_GPU`, GPU time as a
percentage of CPU time, and the corresponding percentage of time saved.

For a quick smoke test:

```bash
./bin/main_cpu_3d --resolution 12 --t-end 0.005 --snapshots 2 --solver hll
```

Snapshot files use the compact `MHD3D01` binary format documented directly in
`scripts/cpu/main_cpu_3d.cpp`: dimensions, bounds/time/gamma, then eight
float32 primitive fields per x-fastest cell.

## Weakly compressible Taylor-Green MHD reference case

The CPU and CUDA executables provide `--case imtg`. Its defaults reproduce the
`Ms0.2_Ma1` initial conditions of Glines, Grete & O'Shea, *Phys. Rev. E* 103,
043203 (2021), arXiv:2009.01331:

```text
u_x =  u0 sin(x/L) cos(y/L) cos(z/L)
u_y = -u0 cos(x/L) sin(y/L) cos(z/L),  u_z = 0
P   = P0 + rho0*u0^2/16
      * [cos(2x/L)+cos(2y/L)] * [cos(2z/L)+2]
rho = P*rho0/P0
```

The magnetic field is the paper's insulating TG field. The periodic domain is
`[-0.5,0.5]^3`, `L=1/(2*pi)`, `P0=rho0=1`, `gamma=5/3`,
`u0=0.5163977795`, and `B0=0.2981423970`. These values give
`Ms_rms=0.2` and equal initial kinetic and magnetic energies (`Ma=1`).
The reference dynamical time is `T=0.9682458366`; the default end time is
`6T=5.8094750193`. The solver deliberately retains GLM divergence cleaning
where the paper uses constrained transport.

CPU smoke test:

```bash
make cpu_3d
./bin/main_cpu_3d --case imtg --resolution 32 --t-end 0.2 \
  --snapshots 4 --out output/imtg3d
```

CSD3 CUDA run:

```bash
mkdir -p logs
sbatch --export=ALL,CASE=imtg,RESOLUTION=128 \
  scripts/csd3_slurm/slurm_gpu_3d.sh
```

The literature default is `1024^3`, HLLD, CFL 0.20, and `t_end=6T`. The Slurm
script uses five equal output intervals, so the initial condition plus the five
evolved states give exactly six files at `t/T = 0, 1.2, 2.4, 3.6, 4.8, 6`.
It automatically renders six-panel `rho`, `current`, and `Bmag` evolution
figures after a successful run (`VISUALIZE=1`, the default). Rendering uses
`stride=2`; the current plot uses the absolute level `|J|=4.5`, which keeps
the analytic initial state visible. The Slurm log reports each field, snapshot,
individual frame, and contact-sheet rendering step. Set, for example,
`PLOT_FIELDS=current` to render only one field or `PLOT_STRIDE=1` for the full
display grid.
A `1024^3` grid does **not** fit the current single-GPU double-precision
implementation: four full 9-variable state grids alone need
about 292 GiB before overhead. The Slurm script detects this and exits instead
of failing inside CUDA. Until multi-GPU domain decomposition is implemented,
use the same physical initial condition at the largest grid accepted by the
GPU-memory preflight check. For example, on an 80-GiB A100:

```bash
sbatch --export=ALL,CASE=imtg,RESOLUTION=512,VISUALIZE=0 \
  scripts/csd3_slurm/slurm_gpu_3d.sh
```

Render the current-sheet evolution as multiple PNGs:

```bash
python3 visualization/plot_blast3d_volume.py \
  --input outputs/imtg3d_gpu_n512_hlld_JOB_ID \
  --field current --fraction 0.45 --stride 2 --png-frames 6
```

The visualization additionally supports `--field vorticity`, `speed`, and
`Bmag`. It checks every selected snapshot for NaN/Inf and accepts
`--max-time` when a known failed tail must be excluded without altering the
original data. `--stride 2` reduces only the rendered voxel grid after the
curl has been evaluated at full resolution, making Matplotlib practical for a
large dataset without changing the simulation.
