#!/usr/bin/env python3
"""Render orthogonal slices and an animation from main_cpu_3d snapshots."""

import argparse
import struct
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np

FIELDS = ("rho", "pressure", "u", "v", "w")


def read_snapshot(path: Path):
    with path.open("rb") as f:
        if f.read(8) != b"EUL3D01\x00":
            raise ValueError(f"{path} is not an EUL3D01 file")
        nx, ny, nz = struct.unpack("<III", f.read(12))
        x0, x1, y0, y1, z0, z1, time, gamma = struct.unpack("<8d", f.read(64))
        values = np.fromfile(f, dtype="<f4")
    expected = nx * ny * nz * len(FIELDS)
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} values, found {values.size}")
    data = values.reshape(nz, ny, nx, len(FIELDS))
    return data, (x0, x1, y0, y1, z0, z1), time, gamma


def slices(data):
    nz, ny, nx, _ = data.shape
    return data[nz // 2], data[:, ny // 2], data[:, :, nx // 2]


def draw(axs, data, bounds, time, field_index, vmin, vmax):
    x0, x1, y0, y1, z0, z1 = bounds
    xy, xz, yz = slices(data)
    panels = (
        (xy[..., field_index], [x0, x1, y0, y1], "z = 0 (xy)"),
        (xz[..., field_index], [x0, x1, z0, z1], "y = 0 (xz)"),
        (yz[..., field_index], [y0, y1, z0, z1], "x = 0 (yz)"),
    )
    images = []
    for ax, (image, extent, title) in zip(axs, panels):
        ax.clear()
        images.append(ax.imshow(image, origin="lower", extent=extent,
                                cmap="magma", vmin=vmin, vmax=vmax,
                                interpolation="bilinear"))
        ax.set_title(title)
        ax.set_aspect("equal")
    axs[0].set_ylabel("y")
    axs[1].set_ylabel("z")
    axs[2].set_ylabel("z")
    axs[0].set_xlabel("x")
    axs[1].set_xlabel("x")
    axs[2].set_xlabel("y")
    axs[1].figure.suptitle(
        f"3D Euler blast — {FIELDS[field_index]}, t = {time:.4f}")
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/euler_blast3d")
    parser.add_argument("--field", choices=FIELDS, default="rho")
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    folder = Path(args.input)
    paths = sorted(folder.glob("euler_blast3d_*.euler3d"))
    if not paths:
        raise SystemExit(f"No euler_blast3d_*.euler3d snapshots in {folder}")
    snapshots = [read_snapshot(path) for path in paths]
    fi = FIELDS.index(args.field)
    all_midplanes = [
        plane[..., fi] for data, *_ in snapshots for plane in slices(data)
    ]
    vmin = min(float(a.min()) for a in all_midplanes)
    vmax = max(float(a.max()) for a in all_midplanes)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    images = draw(axs, *snapshots[-1][:3], fi, vmin, vmax)
    cbar = fig.colorbar(images[0], ax=axs, shrink=0.82, pad=0.02)
    cbar.set_label(args.field)
    png = folder / f"blast3d_{args.field}_final.png"
    fig.savefig(png, dpi=180)
    print(f"Wrote {png}")

    if not args.no_gif:
        gif = folder / f"blast3d_{args.field}_evolution.gif"
        writer = PillowWriter(fps=args.fps)
        with writer.saving(fig, gif, dpi=130):
            for data, bounds, time, _gamma in snapshots:
                draw(axs, data, bounds, time, fi, vmin, vmax)
                writer.grab_frame()
        print(f"Wrote {gif}")


if __name__ == "__main__":
    main()
