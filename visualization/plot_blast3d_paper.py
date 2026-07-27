#!/usr/bin/env python3
"""Compact 2x3 publication figure for the 3D magnetized blast.

Top row: density surfaces at early, middle and final times.
Bottom row: signed magnetic-magnitude perturbations at the same times.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np

from plot_blast3d_volume import (
    FIELD_LABELS,
    draw_frame,
    field_colormap,
    prepare_frames,
    read_snapshot,
)


def main():
    parser = argparse.ArgumentParser(
        description="Render the paper-ready 2x3 blast summary.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--min-time", type=float, default=0.02)
    parser.add_argument("--max-time", type=float)
    parser.add_argument("--panels", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--rho-fraction", type=float, default=0.10)
    parser.add_argument("--bmag-fraction", type=float, default=0.15)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--png-only", action="store_true")
    args = parser.parse_args()
    if (args.panels < 2 or args.stride < 1 or args.dpi < 72 or
            args.rho_fraction <= 0 or args.bmag_fraction <= 0):
        parser.error(
            "panels>=2, stride>=1, dpi>=72 and fractions>0 are required")

    folder = Path(args.input)
    paths = sorted(folder.glob("*.mhd3d"))
    if not paths:
        raise SystemExit("No *.mhd3d snapshots in {}".format(folder))

    loaded = []
    print("[1/4] Loading {} snapshots".format(len(paths)), flush=True)
    for number, path in enumerate(paths, start=1):
        print("      read {}/{}: {}".format(
            number, len(paths), path.name), flush=True)
        snapshot = read_snapshot(path)
        if np.any(~np.isfinite(snapshot[0])):
            raise SystemExit("{} contains non-finite values".format(path))
        time = snapshot[1][6]
        if time < args.min_time:
            continue
        if args.max_time is not None and time > args.max_time:
            continue
        loaded.append((path, snapshot))
    if len(loaded) < 2:
        raise SystemExit(
            "Fewer than two snapshots remain after time filtering")

    count = min(args.panels, len(loaded))
    positions = np.linspace(0, len(loaded)-1, count).round().astype(int)
    selected = [loaded[int(position)] for position in positions]
    snapshots = [item[1] for item in selected]
    times = [snapshot[1][6] for snapshot in snapshots]
    print("[2/4] Selected times: {}".format(
        ", ".join("{:.6g}".format(time) for time in times)), flush=True)

    rho_frames, rho_threshold, rho_norm = prepare_frames(
        snapshots, "rho", args.rho_fraction, None, args.stride)
    dbmag_frames, dbmag_threshold, dbmag_norm = prepare_frames(
        snapshots, "dBmag", args.bmag_fraction, None, args.stride)
    print("[3/4] thresholds: rho={:.6g}, |delta Bmag|={:.6g}".format(
        rho_threshold, dbmag_threshold), flush=True)

    rho_cmap = field_colormap("rho")
    dbmag_cmap = field_colormap("dBmag")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": True,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })

    dataset_stem = selected[0][0].stem.rsplit("_", 1)[0]
    extreme = dataset_stem.startswith("blast3d_extreme")
    uniform_field = ((1.0, 0.0, 0.0) if extreme
                     else (1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0), 0.0))

    columns = count
    figure = plt.figure(
        figsize=(7.2, 4.85 if columns == 3 else 4.5), facecolor="white")
    rows = (
        ("rho", rho_frames, rho_threshold, rho_norm, rho_cmap),
        ("dBmag", dbmag_frames, dbmag_threshold, dbmag_norm, dbmag_cmap),
    )
    panel = 0
    for row, (field, frames, threshold, norm, cmap) in enumerate(rows):
        for column, frame in enumerate(frames):
            panel += 1
            axis = figure.add_subplot(2, columns, panel, projection="3d")
            draw_frame(
                axis, frame, field, threshold, norm, cmap,
                "3D magnetized blast",
                uniform_field_direction=uniform_field if panel == 1 else None,
                paper=True,
                panel_label="({})".format(chr(ord("a")+panel-1)),
                show_time=(row == 0))

    figure.subplots_adjust(
        left=0.025, right=0.895, bottom=0.015, top=0.985,
        wspace=0.02, hspace=0.08)

    # Each row has its own normalization and its own colormap. The density
    # scale is sequential; the magnetic perturbation scale is signed and
    # symmetric around zero.
    for field, norm, cmap, y0 in (
            ("rho", rho_norm, rho_cmap, 0.585),
            ("dBmag", dbmag_norm, dbmag_cmap, 0.105)):
        color_axis = figure.add_axes((0.925, y0, 0.015, 0.31))
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(np.asarray([]))
        colorbar = figure.colorbar(mappable, cax=color_axis)
        colorbar.set_label(FIELD_LABELS[field])
        colorbar.ax.tick_params(labelsize=7)

    output_base = folder / "{}_rho_dBmag_3d_paper".format(dataset_stem)
    png = str(output_base) + ".png"
    print("[4/4] Writing {}".format(png), flush=True)
    figure.savefig(png, dpi=args.dpi)
    if not args.png_only:
        pdf = str(output_base) + ".pdf"
        figure.savefig(pdf, dpi=args.dpi)
        print("      wrote {}".format(pdf), flush=True)
    plt.close(figure)
    print("      wrote {}".format(png), flush=True)


if __name__ == "__main__":
    main()
