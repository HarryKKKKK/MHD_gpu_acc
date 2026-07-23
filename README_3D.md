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
```

On Windows PowerShell:

```powershell
make cpu_3d
.\bin\main_cpu_3d.exe --resolution 48 --snapshots 5 --out output/blast3d
python .\visualization\plot_blast3d.py --input output/blast3d
```

The test is a spherical pressure pulse in a strong, uniform x-directed magnetic field.
At early time it is nearly spherical; magnetic pressure and tension then make
the shock anisotropic. The plotting script writes a final three-plane figure
and an animated GIF showing the density evolution. Use `--field pressure` to
plot pressure instead.

For a quick smoke test:

```bash
./bin/main_cpu_3d --resolution 12 --t-end 0.005 --snapshots 2 --solver hll
```

Snapshot files use the compact `MHD3D01` binary format documented directly in
`scripts/cpu/main_cpu_3d.cpp`: dimensions, bounds/time/gamma, then eight
float32 primitive fields per x-fastest cell.
