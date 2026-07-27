#!/usr/bin/env python3
"""True 3D voxel-surface rendering for MHD3D01 volume snapshots.

This intentionally depends only on NumPy, Matplotlib and Pillow. Cells whose
value differs sufficiently from the far-field background are retained. The
script also derives current density and vorticity for the IMTG case.
"""

import argparse
import math
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

BASE_FIELDS = ("rho", "pressure", "u", "v", "w", "Bx", "By", "Bz")
FIELDS = BASE_FIELDS + ("speed", "Bmag", "dBmag", "current", "vorticity")

# Keep the compatibility import visibly used for linters; registration already
# happened when mpl_toolkits.mplot3d was imported above.
_AXES3D_COMPAT = Axes3D

# Glines, Grete & O'Shea weakly compressible IMTG normalization used by the
# solver: T = pi*L/u0, L=1/(2*pi), u0=2*Ms*sqrt(gamma), Ms=0.2.
IMTG_DYNAMICAL_TIME = (
    0.5 / (2.0 * 0.2 * math.sqrt(5.0 / 3.0)))

FIELD_LABELS = {
    "rho": r"$\rho$",
    "pressure": r"$p$",
    "u": r"$u_x$",
    "v": r"$u_y$",
    "w": r"$u_z$",
    "Bx": r"$B_x$",
    "By": r"$B_y$",
    "Bz": r"$B_z$",
    "speed": r"$|\mathbf{u}|$",
    "Bmag": r"$|\mathbf{B}|$",
    "dBmag": r"$\Delta|\mathbf{B}|$",
    "current": r"$|\mathbf{J}|$",
    "vorticity": r"$|\boldsymbol{\omega}|$",
}


def read_snapshot(path: Path):
    with path.open("rb") as f:
        if f.read(8) != b"MHD3D01\x00":
            raise ValueError(f"{path} is not an MHD3D01 file")
        nx, ny, nz = struct.unpack("<III", f.read(12))
        meta = struct.unpack("<8d", f.read(64))
        values = np.fromfile(f, dtype="<f4")
    expected = nx * ny * nz * len(BASE_FIELDS)
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} values, found {values.size}")
    return values.reshape(nz, ny, nx, len(BASE_FIELDS)), meta


def derived_volume(data, field_name, meta):
    """Return a field and whether its physically meaningful background is 0."""
    if field_name in BASE_FIELDS:
        return data[..., BASE_FIELDS.index(field_name)], False
    if field_name == "speed":
        return np.sqrt(
            data[..., 2]**2 + data[..., 3]**2 + data[..., 4]**2), True
    if field_name in ("Bmag", "dBmag"):
        # A magnetized blast has a non-zero uniform |B| in the far field.
        # Treating this magnitude as a zero-background field fills the entire
        # plotting cube. dBmag is converted to a signed perturbation later,
        # after estimating the boundary value independently in every frame.
        return np.sqrt(
            data[..., 5]**2 + data[..., 6]**2 + data[..., 7]**2), False

    x0, x1, y0, y1, z0, z1 = meta[:6]
    nz, ny, nx = data.shape[:3]
    dx, dy, dz = (x1-x0)/nx, (y1-y0)/ny, (z1-z0)/nz

    def dd(a, axis, spacing):
        return (np.roll(a, -1, axis=axis) -
                np.roll(a, 1, axis=axis))/(2.0*spacing)

    if field_name == "current":
        ax, ay, az = data[..., 5], data[..., 6], data[..., 7]
    else:
        ax, ay, az = data[..., 2], data[..., 3], data[..., 4]
    # File array order is z,y,x.
    curl_x = dd(az, 1, dy) - dd(ay, 0, dz)
    curl_y = dd(ax, 0, dz) - dd(az, 2, dx)
    curl_z = dd(ay, 2, dx) - dd(ax, 1, dy)
    return np.sqrt(curl_x**2 + curl_y**2 + curl_z**2), True


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


def prepare_frames(snapshots, field_name, fraction, absolute_level, stride,
                   bmag_background="boundary"):
    frames = []
    global_deviation = 0.0
    for data, meta in snapshots:
        volume, zero_background = derived_volume(data, field_name, meta)
        if field_name == "Bmag" and bmag_background == "zero":
            zero_background = True
        volume = volume[::stride, ::stride, ::stride]
        background = 0.0 if zero_background else boundary_background(volume)
        if field_name == "dBmag":
            # Preserve the sign: positive values are magnetic amplification
            # and negative values are magnetic depletion relative to |B0|.
            volume = volume - background
            background = 0.0
            deviation = np.abs(volume)
        else:
            deviation = (volume if zero_background
                         else np.abs(volume - background))
        global_deviation = max(global_deviation, float(deviation.max()))
        frames.append((volume, background, meta))
    if global_deviation <= 0:
        raise ValueError(
            f"{field_name} is spatially uniform in every snapshot")
    threshold = absolute_level if absolute_level is not None else fraction * global_deviation

    selected_values = []
    prepared = []
    for volume, background, meta in frames:
        mask = ((volume >= threshold)
                if field_name in ("speed", "current", "vorticity")
                else (np.abs(volume-background) >= threshold))
        prepared.append((volume, mask, background, meta))
        if np.any(mask):
            selected_values.append(volume[mask])
    if not selected_values:
        raise ValueError("the selected level leaves no visible cells")
    visible = np.concatenate(selected_values)
    vmin, vmax = float(visible.min()), float(visible.max())
    if field_name == "dBmag":
        limit = max(abs(vmin), abs(vmax))
        vmin, vmax = -limit, limit
    if vmin == vmax:
        vmax = vmin + 1.0
    return prepared, threshold, Normalize(vmin=vmin, vmax=vmax)


def field_colormap(field_name):
    """Use a signed map for perturbations and sequential maps otherwise."""
    if field_name == "dBmag":
        return plt.get_cmap("RdBu_r")
    try:
        return plt.get_cmap("turbo")
    except ValueError:
        return plt.get_cmap("viridis")


def style_axis(ax, bounds, paper=False):
    x0, x1, y0, y1, z0, z1 = bounds
    ax.set(xlim=(x0, x1), ylim=(y0, y1), zlim=(z0, z1),
           xlabel="x", ylabel="y", zlabel="z")
    # set_box_aspect was added in Matplotlib 3.3. The domain is cubic for this
    # benchmark, so older releases still produce a valid view without it.
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.grid(False)
    if paper:
        foreground = "#111827"
        pane_face = (1.0, 1.0, 1.0, 0.0)
        pane_edge = (0.25, 0.29, 0.36, 0.72)
        ax.set_facecolor("white")
        ax.tick_params(colors=foreground, labelsize=6.5, pad=0)
        ax.xaxis.label.set_size(7.5)
        ax.yaxis.label.set_size(7.5)
        ax.zaxis.label.set_size(7.5)
        try:
            ax.xaxis.labelpad = 0
            ax.yaxis.labelpad = 0
            ax.zaxis.labelpad = 0
        except AttributeError:
            pass
    else:
        foreground = "#e2e8f0"
        pane_face = (0.04, 0.05, 0.09, 1.0)
        pane_edge = (0.35, 0.38, 0.48, 0.45)
        ax.set_facecolor("#080b14")
        ax.tick_params(colors="#cbd5e1")
    ax.xaxis.label.set_color(foreground)
    ax.yaxis.label.set_color(foreground)
    ax.zaxis.label.set_color(foreground)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if hasattr(axis, "pane"):
            axis.pane.set_facecolor(pane_face)
            axis.pane.set_edgecolor(pane_edge)
        if hasattr(axis, "line"):
            axis.line.set_color("#4b5563" if paper else "#64748b")


def draw_frame(ax, frame, field_name, threshold, norm, cmap, case_title,
               uniform_field_direction=None, elevation=24.0, azimuth=38.0,
               paper=False, panel_label=None, time_scale=None,
               layout_preview=False, show_time=True):
    volume, mask, background, meta = frame
    x0, x1, y0, y1, z0, z1, time, _gamma = meta
    bounds = (x0, x1, y0, y1, z0, z1)
    ax.clear()
    style_axis(ax, bounds, paper=paper)
    ax.view_init(elev=elevation, azim=azimuth)

    if np.any(mask):
        # Convert z,y,x file layout to x,y,z plotting layout.
        values_xyz = volume.transpose(2, 1, 0)
        mask_xyz = mask.transpose(2, 1, 0)
        rgba = cmap(norm(values_xyz))
        rgba[..., 3] = 0.90 if paper else 0.82
        X, Y, Z = physical_edges(bounds, volume.shape)
        artists = ax.voxels(
            X, Y, Z, mask_xyz, facecolors=rgba,
            edgecolor=(0.04, 0.04, 0.07, 0.055 if paper else 0.16),
            linewidth=0.035 if paper else 0.12)
        if paper:
            # Keep PDF files compact: axes/text remain vector while the very
            # large voxel collections are embedded as high-resolution rasters.
            for artist in artists.values():
                if hasattr(artist, "set_rasterized"):
                    artist.set_rasterized(True)
    elif not layout_preview:
        ax.text2D(0.5, 0.5, "No cells above threshold",
                  transform=ax.transAxes, ha="center",
                  color="#111827" if paper else "white")

    # The blast has an imposed uniform field; IMTG deliberately does not.
    if uniform_field_direction is not None:
        bx, by, bz = uniform_field_direction
        magnitude = math.sqrt(bx*bx + by*by + bz*bz)
        bx, by, bz = bx/magnitude, by/magnitude, bz/magnitude
        arrow_length = 0.25*(x1-x0)
        arrow_color = "#087f8c" if paper else "#55d9ff"
        arrow_x = x0 + 0.08*(x1-x0)
        arrow_y = y0 + 0.10*(y1-y0)
        arrow_z = z1 - 0.10*(z1-z0)
        ax.quiver(
            arrow_x, arrow_y, arrow_z,
            arrow_length*bx, arrow_length*by, arrow_length*bz,
            color=arrow_color, linewidth=1.4 if paper else 2.0,
            arrow_length_ratio=0.18)
        field_text = (r"$\mathbf{B}_0\parallel x$"
                      if abs(by)+abs(bz) < 1.0e-12
                      else r"$\mathbf{B}_0\parallel(1,1,0)$")
        ax.text(
            arrow_x, arrow_y, z1 - 0.055*(z1-z0), field_text,
            color=arrow_color, fontsize=7 if paper else 10)
    if paper:
        if panel_label:
            ax.text2D(
                0.04, 0.96, panel_label, transform=ax.transAxes,
                color="#111827", fontsize=8.5, fontweight="bold",
                va="top", ha="left")
        if show_time:
            if time_scale is not None:
                time_text = r"$t/T={:.1f}$".format(time/time_scale)
            else:
                time_text = r"$t={:.4g}$".format(time)
            ax.text2D(
                0.94, 0.96, time_text, transform=ax.transAxes,
                color="#111827", fontsize=7.5, va="top", ha="right")
    else:
        ax.set_title(
            f"{case_title}: {field_name}", color="#cbd5e1", pad=10)
        ax.text2D(
            0.03, 0.96,
            f"t={time:.4f}, background={background:.3g}, "
            f"threshold={threshold:.3g}",
            transform=ax.transAxes, color="#cbd5e1", fontsize=9, va="top")


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
    parser.add_argument(
        "--max-time", type=float,
        help="ignore snapshots later than this time")
    parser.add_argument(
        "--min-time", type=float,
        help="ignore snapshots earlier than this time")
    parser.add_argument(
        "--stride", type=int, default=1,
        help="plot every Nth cell in each direction after deriving the field")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--rotation-frames", type=int, default=36)
    parser.add_argument(
        "--png-frames", type=int, default=6,
        help="number of physical times to save as PNGs (default: 6)")
    parser.add_argument(
        "--paper", action="store_true",
        help="use compact publication layout, normalized IMTG time, and labels")
    parser.add_argument(
        "--paper-dpi", type=int, default=300,
        help="raster resolution for --paper output (default: 300)")
    parser.add_argument(
        "--paper-png-only", action="store_true",
        help="with --paper, skip the rasterized PDF copy")
    parser.add_argument(
        "--overview-only", action="store_true",
        help="write only the multi-time contact sheet")
    parser.add_argument(
        "--layout-preview", action="store_true",
        help="render six empty 3D panels without reading simulation data")
    parser.add_argument(
        "--preview-case", choices=("imtg", "blast"), default="imtg",
        help="labels/time convention for --layout-preview (default: imtg)")
    parser.add_argument(
        "--preview-output", default="figs/layout_preview",
        help="output directory for --layout-preview")
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
            args.png_frames < 1 or args.stride < 1 or args.paper_dpi < 72):
        parser.error(
            "fraction, png-frames and stride must be >0; "
            "rotation-frames >=2, fps >=1 and paper-dpi >=72")
    if args.overview_only and (args.rotation_gif or args.evolution_gif):
        parser.error("--overview-only cannot be combined with GIF output")
    if args.layout_preview and (args.rotation_gif or args.evolution_gif):
        parser.error("--layout-preview does not support GIF output")
    if args.paper:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": True,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        })

    if args.layout_preview:
        folder = Path(args.preview_output)
        folder.mkdir(parents=True, exist_ok=True)
        is_imtg = args.preview_case == "imtg"
        is_extreme_blast = False
        dataset_stem = args.preview_case + "3d_layout_preview"
        print("[1/5] Layout preview: no simulation data will be read",
              flush=True)
        bounds = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
        if is_imtg:
            times = np.linspace(0.0, 6.0*IMTG_DYNAMICAL_TIME, 6)
            gamma = 5.0/3.0
        else:
            times = np.linspace(0.0, 0.10, 6)
            gamma = 5.0/3.0
        frames = []
        for time in times:
            volume = np.zeros((4, 4, 4), dtype=float)
            mask = np.zeros_like(volume, dtype=bool)
            meta = tuple(bounds) + (float(time), gamma)
            frames.append((volume, mask, 0.0, meta))
        threshold = 0.0
        norm = Normalize(vmin=0.0, vmax=1.0)
    else:
        folder = Path(args.input)
        paths = sorted(folder.glob("*.mhd3d"))
        if not paths:
            raise SystemExit(f"No *.mhd3d snapshots in {folder}")
        print(f"[1/5] Loading {len(paths)} snapshots from {folder}", flush=True)
        loaded = []
        for number, path in enumerate(paths, start=1):
            print(f"      read {number}/{len(paths)}: {path.name}", flush=True)
            loaded.append((path, read_snapshot(path)))
        if args.max_time is not None:
            loaded = [item for item in loaded if item[1][1][6] <= args.max_time]
        if args.min_time is not None:
            loaded = [item for item in loaded if item[1][1][6] >= args.min_time]
        if not loaded:
            raise SystemExit(
                "No snapshots remain after applying --min-time/--max-time")
        for path, (data, _meta) in loaded:
            bad = int(np.count_nonzero(~np.isfinite(data)))
            if bad:
                raise SystemExit(
                    f"{path}: contains {bad} non-finite primitive values; "
                    "use --max-time to exclude a known failed tail")
        paths = [item[0] for item in loaded]
        snapshots = [item[1] for item in loaded]
        dataset_stem = paths[0].stem.rsplit("_", 1)[0]
        is_imtg = dataset_stem.startswith("imtg")
        is_extreme_blast = dataset_stem.startswith("blast3d_extreme")
    case_title = ("Weakly compressible IMTG: Ms0.2_Ma1"
                  if is_imtg else "3D magnetized blast")
    time_scale = IMTG_DYNAMICAL_TIME if is_imtg and args.paper else None
    if is_imtg:
        uniform_field_direction = None
    elif is_extreme_blast:
        uniform_field_direction = (1.0, 0.0, 0.0)
    else:
        uniform_field_direction = (
            1.0/math.sqrt(2.0), 1.0/math.sqrt(2.0), 0.0)
    output_suffix = "_paper" if args.paper else ""
    if args.layout_preview:
        print("[2/5] Using empty 4x4x4 placeholder volumes", flush=True)
    else:
        print(f"[2/5] Computing {args.field} and visibility masks "
              f"(stride={args.stride})", flush=True)
        frames, threshold, norm = prepare_frames(
            snapshots, args.field, args.fraction, args.level, args.stride,
            bmag_background="zero" if is_imtg else "boundary")
        print(f"      threshold={threshold:.6g}", flush=True)
    # `matplotlib.colormaps` is unavailable on older CSD3 installations.
    cmap = field_colormap(args.field)

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    # Matplotlib 2.x requires an attached array even when norm/cmap are given.
    mappable.set_array(np.asarray([]))
    fig = None
    ax = None
    if not args.overview_only:
        print("[3/5] Rendering final-time image", flush=True)
        single_size = (4.0, 3.5) if args.paper else (8.4, 7.2)
        fig = plt.figure(figsize=single_size, facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(
            left=0.01, right=0.86, bottom=0.01,
            top=0.99 if args.paper else 0.91)
        draw_frame(
            ax, frames[-1], args.field, threshold, norm, cmap,
            case_title, uniform_field_direction=uniform_field_direction,
            paper=args.paper, time_scale=time_scale,
            layout_preview=args.layout_preview)
        cax = fig.add_axes((0.89, 0.20, 0.025, 0.60))
        colorbar = fig.colorbar(mappable, cax=cax)
        colorbar.set_label(FIELD_LABELS[args.field] if args.paper
                           else args.field)
        if args.paper:
            colorbar.ax.tick_params(labelsize=7)

        png = folder / (
            f"{dataset_stem}_{args.field}_3d{output_suffix}.png")
        # Passing Path directly is unreliable with the Python 3.6 Matplotlib
        # bundled on CSD3.
        fig.savefig(str(png), dpi=args.paper_dpi if args.paper else 190)
        print(f"      wrote {png}", flush=True)
    else:
        print("[3/5] Skipping final-time image (--overview-only)", flush=True)

    # Select evenly spaced physical times that actually contain a visible
    # structure. For density, the t=0 state is spatially uniform and is
    # intentionally skipped rather than producing an empty panel.
    visible_indices = (
        list(range(len(frames))) if args.layout_preview
        else [i for i, frame in enumerate(frames) if np.any(frame[1])])
    count = min(args.png_frames, len(visible_indices))
    positions = np.linspace(0, len(visible_indices) - 1, count).round().astype(int)
    selected_indices = []
    for position in positions:
        index = visible_indices[int(position)]
        if index not in selected_indices:
            selected_indices.append(index)

    if not args.overview_only:
        print(f"[4/5] Rendering {len(selected_indices)} selected time images",
              flush=True)
        for frame_number, index in enumerate(selected_indices, start=1):
            frame_time = frames[index][3][6]
            print(f"      frame {frame_number}/{len(selected_indices)}: "
                  f"snapshot={index:03d}, t={frame_time:.6f}", flush=True)
            draw_frame(
                ax, frames[index], args.field, threshold, norm, cmap,
                case_title, uniform_field_direction=uniform_field_direction,
                paper=args.paper, time_scale=time_scale,
                layout_preview=args.layout_preview)
            frame_png = folder / (
                f"{dataset_stem}_{args.field}_3d_frame_{index:03d}"
                f"{output_suffix}.png")
            fig.savefig(
                str(frame_png), dpi=args.paper_dpi if args.paper else 190)
            print(f"      wrote {frame_png}", flush=True)
    else:
        print("[4/5] Skipping individual time images (--overview-only)",
              flush=True)

    # Also produce a single contact sheet for papers and presentations.
    print("[5/5] Rendering evolution contact sheet", flush=True)
    if len(selected_indices) >= 5 or len(selected_indices) == 3:
        columns = 3
    elif len(selected_indices) >= 2:
        columns = 2
    else:
        columns = 1
    rows = int(np.ceil(len(selected_indices) / float(columns)))
    if args.paper:
        # 7.2 inches fits a conventional two-column journal figure.
        overview_size = (7.2, 2.58*rows)
    else:
        overview_size = (5.2 * columns + 0.8, 4.6 * rows)
    overview = plt.figure(figsize=overview_size, facecolor="white")
    overview_axes = []
    for panel, index in enumerate(selected_indices, start=1):
        print(f"      panel {panel}/{len(selected_indices)}", flush=True)
        panel_ax = overview.add_subplot(rows, columns, panel, projection="3d")
        draw_frame(
            panel_ax, frames[index], args.field, threshold, norm, cmap,
            case_title, uniform_field_direction=uniform_field_direction,
            paper=args.paper,
            panel_label="({})".format(chr(ord("a")+panel-1))
            if args.paper else None,
            time_scale=time_scale, layout_preview=args.layout_preview)
        overview_axes.append(panel_ax)
    if args.paper:
        overview.subplots_adjust(
            left=0.025, right=0.895, bottom=0.015, top=0.985,
            wspace=0.02, hspace=0.08)
        overview_cax = overview.add_axes((0.925, 0.22, 0.016, 0.56))
    else:
        overview.subplots_adjust(
            left=0.01, right=0.89, bottom=0.04, top=0.94,
            wspace=0.03, hspace=0.28)
        overview_cax = overview.add_axes((0.91, 0.20, 0.018, 0.60))
    overview_mappable = ScalarMappable(norm=norm, cmap=cmap)
    overview_mappable.set_array(np.asarray([]))
    overview_colorbar = overview.colorbar(overview_mappable, cax=overview_cax)
    overview_colorbar.set_label(
        FIELD_LABELS[args.field] if args.paper else args.field)
    if args.paper:
        overview_colorbar.ax.tick_params(labelsize=7)
    overview_png = folder / (
        f"{dataset_stem}_{args.field}_3d_evolution{output_suffix}.png")
    overview.savefig(
        str(overview_png), dpi=args.paper_dpi if args.paper else 180)
    if args.paper and not args.paper_png_only:
        overview_pdf = folder / (
            f"{dataset_stem}_{args.field}_3d_evolution_paper.pdf")
        overview.savefig(str(overview_pdf), dpi=args.paper_dpi)
        print(f"      wrote {overview_pdf}", flush=True)
    plt.close(overview)
    print(f"      wrote {overview_png}", flush=True)
    print(f"Completed {args.field} rendering.", flush=True)

    if args.rotation_gif and not args.no_rotation:
        gif = folder / f"{dataset_stem}_{args.field}_3d_rotation.gif"
        writer = PillowWriter(fps=args.fps)
        with writer.saving(fig, str(gif), dpi=125):
            for azimuth in np.linspace(0, 360, args.rotation_frames, endpoint=False):
                draw_frame(
                    ax, frames[-1], args.field, threshold, norm, cmap,
                    case_title,
                    uniform_field_direction=uniform_field_direction,
                    elevation=24, azimuth=float(azimuth))
                writer.grab_frame()
        print(f"Wrote {gif}")

    if args.evolution_gif and not args.no_evolution:
        gif = folder / f"{dataset_stem}_{args.field}_3d_evolution.gif"
        writer = PillowWriter(fps=args.fps)
        with writer.saving(fig, str(gif), dpi=125):
            for frame in frames:
                draw_frame(ax, frame, args.field, threshold, norm, cmap,
                           case_title,
                           uniform_field_direction=uniform_field_direction)
                writer.grab_frame()
        print(f"Wrote {gif}")


if __name__ == "__main__":
    main()
