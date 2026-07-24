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

## Taylor-Green MHD compatibility case

The same CPU and CUDA executables also provide `--case imtg`. This uses the
Insulating Magnetic Taylor-Green initial condition from Pouquet et al.,
arXiv:0906.1384:

```text
u = (sin(x) cos(y) cos(z), -cos(x) sin(y) cos(z), 0)
B = (cos(x) sin(y) sin(z),
     sin(x) cos(y) sin(z),
    -2 sin(x) sin(y) cos(z)) / sqrt(3)
```

The domain is `[0,2*pi]^3` with periodic boundaries. The amplitudes give
`EV=EM=0.125` and total initial kinetic-plus-magnetic energy 0.25, as in the
paper. Since this repository solves compressible ideal GLM-MHD rather than
incompressible viscous-resistive MHD, this is explicitly a compatibility
counterpart: it uses `rho=1`, `gamma=5/3`, and uniform thermal pressure `p=10`
to keep the initial flow at low Mach number. It does not reproduce the
paper's `nu=eta`, pressure projection, pseudo-spectral discretization, or
2/3 dealiasing.

CPU smoke test:

```bash
make cpu_3d
./bin/main_cpu_3d --case imtg --resolution 32 --t-end 0.2 \
  --snapshots 4 --out output/imtg3d
```

CSD3 CUDA run:

```bash
mkdir -p logs
sbatch --export=ALL,CASE=imtg,VISUALIZE=0 \
  scripts/csd3_slurm/slurm_gpu_3d.sh
```

The IMTG Slurm defaults are `128^3`, HLLD, CFL 0.20, `t_end=2`, and eight
output intervals. Render the current-sheet evolution as multiple PNGs:

```bash
python3 visualization/plot_blast3d_volume.py \
  --input outputs/imtg3d_gpu_n128_hlld_JOB_ID \
  --field current --fraction 0.45 --stride 2 --png-frames 6
```

The visualization additionally supports `--field vorticity`, `speed`, and
`Bmag`. It checks every selected snapshot for NaN/Inf and accepts
`--max-time` when a known failed tail must be excluded without altering the
original data. `--stride 2` reduces only the rendered voxel grid after the
curl has been evaluated at full resolution, making Matplotlib practical for a
`128^3` dataset without changing the simulation.
