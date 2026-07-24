#!/usr/bin/env python3
"""True 3D voxel-surface rendering for MHD3D01 volume snapshots.

This intentionally depends only on NumPy, Matplotlib and Pillow.  Cells whose
value differs sufficiently from the far-field background are retained;
Matplotlib then draws only the exposed faces of that 3D cell set.
"""

import argparse
import struct
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
# Matplotlib bundled with Python 3.6 on CSD3 does not auto-register the "3d"
# projection. Importing Axes3D performs that registration as a side effect.
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

FIELDS = ("rho", "pressure", "u", "v", "w", "Bx", "By", "Bz")

# Keep the compatibility import visibly used for linters; registration already
# happened when mpl_toolkits.mplot3d was imported above.
_AXES3D_COMPAT = Axes3D


def read_snapshot(path: Path):
    with path.open("rb") as f:
        if f.read(8) != b"MHD3D01\x00":
            raise ValueError(f"{path} is not an MHD3D01 file")
        nx, ny, nz = struct.unpack("<III", f.read(12))
        meta = struct.unpack("<8d", f.read(64))
        values = np.fromfile(f, dtype="<f4")
    expected = nx * ny * nz * len(FIELDS)
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} values, found {values.size}")
    return values.reshape(nz, ny, nx, len(FIELDS)), meta


def boundary_background(volume):
    """Robust far-field value from the six outermost active-cell faces."""
    boundary = np.concatenate((
        volume[0].ravel(), volume[-1].ravel(),
        volume[:, 0].ravel(), volume[:, -1].ravel(),
        volume[:, :, 0].ravel(), volume[:, :, -1].ravel(),
    ))
    return float(np.median(boundary))


def physical_edges(bounds, shape):
    x0, x1, y0, y1, z0, z1 = bounds
    nz, ny, nx = shape
    xe = np.linspace(x0, x1, nx + 1)
    ye = np.linspace(y0, y1, ny + 1)
    ze = np.linspace(z0, z1, nz + 1)
    # ax.voxels expects arrays ordered as x, y, z, while file data is z, y, x.
    return np.meshgrid(xe, ye, ze, indexing="ij")


def prepare_frames(snapshots, field_index, fraction, absolute_level):
    frames = []
    global_deviation = 0.0
    for data, meta in snapshots:
        volume = data[..., field_index]
        background = boundary_background(volume)
        deviation = np.abs(volume - background)
        global_deviation = max(global_deviation, float(deviation.max()))
        frames.append((volume, background, meta))
    if global_deviation <= 0:
        raise ValueError(
            f"{FIELDS[field_index]} is spatially uniform in every snapshot")
    threshold = absolute_level if absolute_level is not None else fraction * global_deviation

    selected_values = []
    prepared = []
    for volume, background, meta in frames:
        mask = np.abs(volume - background) >= threshold
        prepared.append((volume, mask, background, meta))
        if np.any(mask):
            selected_values.append(volume[mask])
    if not selected_values:
        raise ValueError("the selected level leaves no visible cells")
    visible = np.concatenate(selected_values)
    vmin, vmax = float(visible.min()), float(visible.max())
    if vmin == vmax:
        vmax = vmin + 1.0
    return prepared, threshold, Normalize(vmin=vmin, vmax=vmax)


def style_axis(ax, bounds):
    x0, x1, y0, y1, z0, z1 = bounds
    ax.set(xlim=(x0, x1), ylim=(y0, y1), zlim=(z0, z1),
           xlabel="x", ylabel="y", zlabel="z")
    # set_box_aspect was added in Matplotlib 3.3. The domain is cubic for this
    # benchmark, so older releases still produce a valid view without it.
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.grid(False)
    ax.set_facecolor("#080b14")
    ax.tick_params(colors="#cbd5e1")
    ax.xaxis.label.set_color("#e2e8f0")
    ax.yaxis.label.set_color("#e2e8f0")
    ax.zaxis.label.set_color("#e2e8f0")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if hasattr(axis, "pane"):
            axis.pane.set_facecolor((0.04, 0.05, 0.09, 1.0))
            axis.pane.set_edgecolor((0.35, 0.38, 0.48, 0.45))
        if hasattr(axis, "line"):
            axis.line.set_color("#64748b")


def draw_frame(ax, frame, field_name, threshold, norm, cmap,
               elevation=24.0, azimuth=38.0):
    volume, mask, background, meta = frame
    x0, x1, y0, y1, z0, z1, time, _gamma = meta
    bounds = (x0, x1, y0, y1, z0, z1)
    ax.clear()
    style_axis(ax, bounds)
    ax.view_init(elev=elevation, azim=azimuth)

    if np.any(mask):
        # Convert z,y,x file layout to x,y,z plotting layout.
        values_xyz = volume.transpose(2, 1, 0)
        mask_xyz = mask.transpose(2, 1, 0)
        rgba = cmap(norm(values_xyz))
        rgba[..., 3] = 0.82
        X, Y, Z = physical_edges(bounds, volume.shape)
        ax.voxels(X, Y, Z, mask_xyz, facecolors=rgba,
                  edgecolor=(0.04, 0.04, 0.07, 0.16), linewidth=0.12)
    else:
        ax.text2D(0.5, 0.5, "No cells above threshold",
                  transform=ax.transAxes, ha="center", color="white")

    # Show the imposed magnetic-field direction for this benchmark.
    arrow_length = 0.26 * (x1 - x0)
    ax.quiver(x0 + 0.08*(x1-x0), y1 - 0.10*(y1-y0), z1 - 0.10*(z1-z0),
              arrow_length, 0, 0, color="#55d9ff", linewidth=2.0,
              arrow_length_ratio=0.18)
    ax.text(x0 + 0.08*(x1-x0), y1 - 0.10*(y1-y0), z1 - 0.06*(z1-z0),
            r"$\mathbf{B}_0\parallel x$", color="#55d9ff")
    ax.set_title(
        f"3D magnetized blast: {field_name}\n"
        f"t = {time:.4f}, background = {background:.3g}, "
        f"|Δ| ≥ {threshold:.3g}",
        color="#111827", pad=14)


def main():
    parser = argparse.ArgumentParser(
        description="Render MHD3D01 snapshots as genuine 3D voxel surfaces.")
    parser.add_argument("--input", default="output/blast3d")
    parser.add_argument("--field", choices=FIELDS, default="rho")
    parser.add_argument(
        "--fraction", type=float, default=0.18,
        help="visible deviation threshold as a fraction of the global maximum")
    parser.add_argument(
        "--level", type=float,
        help="absolute |field-background| threshold; overrides --fraction")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--rotation-frames", type=int, default=36)
    parser.add_argument(
        "--png-frames", type=int, default=6,
        help="number of physical times to save as PNGs (default: 6)")
    parser.add_argument(
        "--rotation-gif", action="store_true",
        help="also create a rotating GIF (off by default)")
    parser.add_argument(
        "--evolution-gif", action="store_true",
        help="also create a time-evolution GIF (off by default)")
    # Retain compatibility with commands written for the older GIF-by-default
    # version. GIFs are now opt-in, so these flags are harmless.
    parser.add_argument("--no-rotation", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--no-evolution", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.fraction <= 0 or args.rotation_frames < 2 or args.fps < 1 or
            args.png_frames < 1):
        parser.error(
            "fraction and png-frames must be >0, rotation-frames >=2, fps >=1")

    folder = Path(args.input)
    paths = sorted(folder.glob("blast3d_*.mhd3d"))
    if not paths:
        raise SystemExit(f"No blast3d_*.mhd3d snapshots in {folder}")
    snapshots = [read_snapshot(path) for path in paths]
    fi = FIELDS.index(args.field)
    frames, threshold, norm = prepare_frames(
        snapshots, fi, args.fraction, args.level)
    # `matplotlib.colormaps` is unavailable on older CSD3 installations.
    # Turbo itself appeared in Matplotlib 3.3, so fall back to the widely
    # available viridis map when needed.
    try:
        cmap = plt.get_cmap("turbo")
    except ValueError:
        cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(8.4, 7.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.03, top=0.91)
    draw_frame(ax, frames[-1], args.field, threshold, norm, cmap)
    cax = fig.add_axes((0.90, 0.20, 0.025, 0.60))
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    # Matplotlib 2.x requires an attached array even when norm/cmap are given.
    mappable.set_array(np.asarray([]))
    colorbar = fig.colorbar(mappable, cax=cax)
    colorbar.set_label(args.field)

    png = folder / f"blast3d_{args.field}_3d.png"
    # Passing Path directly is unreliable with the Python 3.6 Matplotlib
    # bundled on CSD3.
    fig.savefig(str(png), dpi=190)
    print(f"Wrote {png}")

    # Select evenly spaced physical times that actually contain a visible
    # structure. For density, the t=0 state is spatially uniform and is
    # intentionally skipped rather than producing an empty panel.
    visible_indices = [i for i, frame in enumerate(frames) if np.any(frame[1])]
    count = min(args.png_frames, len(visible_indices))
    positions = np.linspace(0, len(visible_indices) - 1, count).round().astype(int)
    selected_indices = []
    for position in positions:
        index = visible_indices[int(position)]
        if index not in selected_indices:
            selected_indices.append(index)

    # Write one full-resolution image per selected physical time.
    for index in selected_indices:
        draw_frame(ax, frames[index], args.field, threshold, norm, cmap)
        frame_png = folder / (
            f"blast3d_{args.field}_3d_frame_{index:03d}.png")
        fig.savefig(str(frame_png), dpi=190)
        print(f"Wrote {frame_png}")

    # Also produce a single contact sheet for papers and presentations.
    if len(selected_indices) >= 5:
        columns = 3
    elif len(selected_indices) >= 2:
        columns = 2
    else:
        columns = 1
    rows = int(np.ceil(len(selected_indices) / float(columns)))
    overview = plt.figure(
        figsize=(5.2 * columns + 0.8, 4.6 * rows), facecolor="white")
    overview_axes = []
    for panel, index in enumerate(selected_indices, start=1):
        panel_ax = overview.add_subplot(rows, columns, panel, projection="3d")
        draw_frame(panel_ax, frames[index], args.field, threshold, norm, cmap)
        overview_axes.append(panel_ax)
    overview.subplots_adjust(
        left=0.01, right=0.89, bottom=0.04, top=0.94, wspace=0.03, hspace=0.28)
    overview_cax = overview.add_axes((0.91, 0.20, 0.018, 0.60))
    overview_mappable = ScalarMappable(norm=norm, cmap=cmap)
    overview_mappable.set_array(np.asarray([]))
    overview_colorbar = overview.colorbar(overview_mappable, cax=overview_cax)
    overview_colorbar.set_label(args.field)
    overview_png = folder / f"blast3d_{args.field}_3d_evolution.png"
    overview.savefig(str(overview_png), dpi=180)
    plt.close(overview)
    print(f"Wrote {overview_png}")

    if args.rotation_gif and not args.no_rotation:
        gif = folder / f"blast3d_{args.field}_3d_rotation.gif"
        writer = PillowWriter(fps=args.fps)
        with writer.saving(fig, str(gif), dpi=125):
            for azimuth in np.linspace(0, 360, args.rotation_frames, endpoint=False):
                draw_frame(ax, frames[-1], args.field, threshold, norm, cmap,
                           elevation=24, azimuth=float(azimuth))
                writer.grab_frame()
        print(f"Wrote {gif}")

    if args.evolution_gif and not args.no_evolution:
        gif = folder / f"blast3d_{args.field}_3d_evolution.gif"
        writer = PillowWriter(fps=args.fps)
        with writer.saving(fig, str(gif), dpi=125):
            for frame in frames:
                draw_frame(ax, frame, args.field, threshold, norm, cmap)
                writer.grab_frame()
        print(f"Wrote {gif}")


if __name__ == "__main__":
    main()
