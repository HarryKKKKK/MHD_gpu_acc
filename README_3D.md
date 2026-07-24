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

The test now reproduces the three-dimensional magnetized blast in Sec. 5.6 of
Derigs et al., JCP 317 (2016), 223-256 (DOI:
10.1016/j.jcp.2016.04.048). On `[-0.5,0.5]^3`, it uses `rho=1`, zero velocity,
`gamma=1.4`, and `B=(100/sqrt(4*pi),0,0)`. Pressure is 1000 for `r<=0.09`,
0.1 for `r>=0.10`, and linearly interpolated in between. All boundaries are
periodic and the reference output time is `t=0.01`. These parameters live in
`head/blast3d_case.hpp` and are shared by the CPU and CUDA drivers.

The strong magnetic field makes the shock markedly anisotropic. Use
`--field pressure` to render pressure instead of density.

`plot_blast3d_volume.py` produces a genuine three-dimensional voxel surface
rather than planar slices. By default it writes several fixed-camera PNGs at
different physical times and one combined evolution overview. The visible surface is selected by
`abs(field - far_field) >= fraction * global_max_deviation`; adjust it with
`--fraction 0.1`, or specify an absolute `--level`. GIF output is disabled by
default; it is available only when explicitly requested with `--rotation-gif`
or `--evolution-gif`.

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

The default is a 128-cubed HLLD run to the reference time `t=0.01`, with five
output intervals on one Ampere GPU. This matches the maximum 3D resolution
reported for the paper's Fig. 19; the default CFL is 0.20.
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
