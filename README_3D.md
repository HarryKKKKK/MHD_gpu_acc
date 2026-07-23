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

The test is a spherical pressure pulse in a strong, uniform x-directed magnetic field.
At early time it is nearly spherical; magnetic pressure and tension then make
the shock anisotropic. The plotting script writes a final three-plane figure
and an animated GIF showing the density evolution. Use `--field pressure` to
plot pressure instead.

`plot_blast3d_volume.py` produces a genuine three-dimensional voxel surface
rather than planar slices. It writes a static 3D view, a 360-degree rotating
GIF, and a fixed-camera time-evolution GIF. The visible surface is selected by
`abs(field - far_field) >= fraction * global_max_deviation`; adjust it with
`--fraction 0.1`, or specify an absolute `--level`.

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

Useful exported variables are `RESOLUTION`, `T_END`, `SNAPSHOTS`, `SOLVER`,
`CFL`, `OUT_DIR`, `OMP_THREADS`, `FIELD`, `FRACTION`, `ROTATION_FRAMES`,
`VISUALIZE`, `PYTHON_BIN`, and `MODULES_STR`. If Python rendering libraries are
not installed on compute nodes, use `VISUALIZE=0` and run the visualization
script later against the generated snapshot directory.

For a quick smoke test:

```bash
./bin/main_cpu_3d --resolution 12 --t-end 0.005 --snapshots 2 --solver hll
```

Snapshot files use the compact `MHD3D01` binary format documented directly in
`scripts/cpu/main_cpu_3d.cpp`: dimensions, bounds/time/gamma, then eight
float32 primitive fields per x-fastest cell.
