#!/bin/bash -l
#SBATCH -J mhd3d_plot
#SBATCH -p icelake
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Render existing MHD3D01 snapshots on a CSD3 CPU node.
# This script only plots data; it does not rebuild or rerun the MHD solver.
# It creates independent PNG frames, a final-time PNG, and an evolution
# contact sheet. It never requests GIF output.
#
# Submit from the repository root. The logs directory must exist before
# sbatch is called:
#
#   mkdir -p logs
#
# 1. Athena/moderate blast (rho, pressure and |B| by default):
#
#   sbatch -A YOUR_CSD3_CPU_ACCOUNT \
#     --export=ALL,INPUT_DIR=outputs/blast3d_gpu_n128_hlld_JOBID,CASE=blast \
#     scripts/csd3_slurm/slurm_render_3d.sh
#
# 2. Extreme/Derigs blast:
#
#   sbatch -A YOUR_CSD3_CPU_ACCOUNT \
#     --export=ALL,INPUT_DIR=outputs/blast3d_extreme_gpu_n128_hlld_JOBID,CASE=blast_extreme \
#     scripts/csd3_slurm/slurm_render_3d.sh
#
# 3. Weakly compressible IMTG (rho, current and |B| by default):
#
#   sbatch -A YOUR_CSD3_CPU_ACCOUNT \
#     --export=ALL,INPUT_DIR=outputs/imtg3d_gpu_n128_hlld_JOBID,CASE=imtg \
#     scripts/csd3_slurm/slurm_render_3d.sh
#
# CASE may be omitted: it is inferred from the first *.mhd3d filename.
# INPUT_DIR may instead be supplied as the first positional argument:
#
#   sbatch -A YOUR_CSD3_CPU_ACCOUNT \
#     scripts/csd3_slurm/slurm_render_3d.sh \
#     outputs/blast3d_gpu_n128_hlld_JOBID
#
# After submission, follow the per-field and per-frame progress with:
#
#   tail -f logs/mhd3d_plot_SLURM_JOB_ID.out
#
# Common overrides:
#
#   PLOT_FIELDS=rho                 render only density
#   PLOT_FIELDS=rho+pressure+Bmag   multiple fields in sbatch --export
#   PNG_FRAMES=6                    number of selected physical times
#   PLOT_STRIDE=2                   plot every second cell in x/y/z
#   RHO_FRACTION=0.08               show more of the density disturbance
#   PRESSURE_FRACTION=0.10          pressure visibility threshold
#   CURRENT_FRACTION=0.12           current-density visibility threshold
#   BMAG_FRACTION=0.15              magnetic-magnitude threshold
#   FRACTION=0.18                   fallback threshold for other fields
#   LEVEL=VALUE                     absolute level for every requested field
#   MAX_TIME=VALUE                  ignore snapshots after this physical time
#   PYTHON_BIN=/path/to/venv/bin/python
#   PAPER=1                         publication layout + 300 dpi PNG/PDF
#   PAPER_DPI=300                   publication raster resolution
#   OVERVIEW_ONLY=1                 skip final/individual PNGs (PAPER default)
#   PAPER_PNG_ONLY=1                do not write the rasterized PDF
#
# Example with custom rendering parameters:
#
#   sbatch -A YOUR_CSD3_CPU_ACCOUNT \
#     --export=ALL,INPUT_DIR=outputs/blast3d_gpu_n128_hlld_JOBID,CASE=blast,PLOT_FIELDS=rho,PNG_FRAMES=6,RHO_FRACTION=0.06,PLOT_STRIDE=1 \
#     scripts/csd3_slurm/slurm_render_3d.sh
#
# Final IMTG |B| figure in publication layout:
#
#   sbatch -A YOUR_CSD3_CPU_ACCOUNT \
#     --export=ALL,INPUT_DIR=outputs/imtg3d_gpu_n128_hlld_JOBID,CASE=imtg,PLOT_FIELDS=Bmag,PAPER=1 \
#     scripts/csd3_slurm/slurm_render_3d.sh
#
# A comma-separated PLOT_FIELDS value also works when exported by the shell
# instead of embedded in sbatch --export:
#
#   export PLOT_FIELDS='rho,pressure,Bmag'
#   sbatch -A YOUR_CSD3_CPU_ACCOUNT --export=ALL,INPUT_DIR=outputs/blast3d_gpu_n128_hlld_JOBID \
#     scripts/csd3_slurm/slurm_render_3d.sh
#
# If your account does not use the icelake partition, override it at submit
# time with, for example, `sbatch -p sapphire ...`.

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SUBMIT_DIR}}"
cd "${WORKDIR}"

if [[ ! -f visualization/plot_blast3d_volume.py ]]; then
    echo "[ERROR] Submit from the MHD repository root or set WORKDIR."
    exit 1
fi

INPUT_DIR="${INPUT_DIR:-${1:-}}"
if [[ -z "${INPUT_DIR}" ]]; then
    echo "[ERROR] INPUT_DIR is required."
    echo "Example: sbatch --export=ALL,INPUT_DIR=outputs/blast3d_gpu_n128_hlld_JOBID \\"
    echo "         scripts/csd3_slurm/slurm_render_3d.sh"
    exit 2
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "[ERROR] Snapshot directory does not exist: ${INPUT_DIR}"
    exit 2
fi

shopt -s nullglob
SNAPSHOT_FILES=("${INPUT_DIR}"/*.mhd3d)
shopt -u nullglob
if (( ${#SNAPSHOT_FILES[@]} == 0 )); then
    echo "[ERROR] No *.mhd3d snapshots found in ${INPUT_DIR}"
    exit 2
fi

FIRST_SNAPSHOT="$(basename "${SNAPSHOT_FILES[0]}")"
CASE="${CASE:-auto}"
if [[ "${CASE}" == "auto" ]]; then
    case "${FIRST_SNAPSHOT}" in
        imtg3d_*) CASE=imtg ;;
        blast3d_extreme_*) CASE=blast_extreme ;;
        blast3d_*) CASE=blast ;;
        *)
            echo "[ERROR] Cannot infer CASE from ${FIRST_SNAPSHOT}."
            echo "Set CASE=blast, CASE=blast_extreme, or CASE=imtg."
            exit 2
            ;;
    esac
fi

case "${CASE}" in
    blast|blast_athena)
        PLOT_FIELDS="${PLOT_FIELDS:-rho,pressure,Bmag}"
        DEFAULT_PLOT_STRIDE=1
        ;;
    blast_extreme)
        PLOT_FIELDS="${PLOT_FIELDS:-rho,pressure,Bmag}"
        DEFAULT_PLOT_STRIDE=1
        ;;
    imtg)
        PLOT_FIELDS="${PLOT_FIELDS:-rho,current,Bmag}"
        DEFAULT_PLOT_STRIDE=2
        ;;
    *)
        echo "[ERROR] CASE must be blast, blast_athena, blast_extreme, or imtg."
        exit 2
        ;;
esac

# '+' is safe inside Slurm's comma-separated --export syntax. Internally the
# rendering loop uses commas, so accept both notations.
PLOT_FIELDS="${PLOT_FIELDS//+/,}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PNG_FRAMES="${PNG_FRAMES:-6}"
PLOT_STRIDE="${PLOT_STRIDE:-${DEFAULT_PLOT_STRIDE}}"
RHO_FRACTION="${RHO_FRACTION:-0.10}"
PRESSURE_FRACTION="${PRESSURE_FRACTION:-0.10}"
CURRENT_FRACTION="${CURRENT_FRACTION:-0.12}"
BMAG_FRACTION="${BMAG_FRACTION:-0.15}"
FRACTION="${FRACTION:-0.18}"
LEVEL="${LEVEL:-}"
MAX_TIME="${MAX_TIME:-}"
PAPER="${PAPER:-0}"
PAPER_DPI="${PAPER_DPI:-300}"
OVERVIEW_ONLY="${OVERVIEW_ONLY:-${PAPER}}"
PAPER_PNG_ONLY="${PAPER_PNG_ONLY:-0}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found: ${PYTHON_BIN}"
    echo "Set PYTHON_BIN=/path/to/your/venv/bin/python."
    exit 1
fi
if ! "${PYTHON_BIN}" -c "import numpy, matplotlib, PIL" >/dev/null 2>&1; then
    echo "[ERROR] ${PYTHON_BIN} cannot import numpy, matplotlib and Pillow."
    echo "Activate your environment before sbatch, or export PYTHON_BIN."
    exit 1
fi

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

echo "===== CSD3 3D RENDER JOB ====="
echo "Job ID        : ${SLURM_JOB_ID:-manual}"
echo "Host          : $(hostname)"
echo "Start         : $(date --iso-8601=seconds)"
echo "Workdir       : ${WORKDIR}"
echo "Input         : ${INPUT_DIR}"
echo "Snapshots     : ${#SNAPSHOT_FILES[@]}"
echo "Detected case : ${CASE}"
echo "Fields        : ${PLOT_FIELDS}"
echo "PNG frames    : ${PNG_FRAMES}"
echo "Plot stride   : ${PLOT_STRIDE}"
echo "Paper layout  : ${PAPER}"
echo "Overview only : ${OVERVIEW_ONLY}"
echo "Python        : $(command -v "${PYTHON_BIN}")"

IFS=',' read -r -a RENDER_FIELDS <<< "${PLOT_FIELDS}"
RENDER_TOTAL="${#RENDER_FIELDS[@]}"
RENDER_INDEX=0

for RENDER_FIELD in "${RENDER_FIELDS[@]}"; do
    RENDER_INDEX=$((RENDER_INDEX + 1))
    RENDER_FIELD="${RENDER_FIELD//[[:space:]]/}"
    if [[ -z "${RENDER_FIELD}" ]]; then
        echo "[ERROR] PLOT_FIELDS contains an empty field."
        exit 2
    fi

    PLOT_ARGS=(
        --input "${INPUT_DIR}"
        --field "${RENDER_FIELD}"
        --png-frames "${PNG_FRAMES}"
        --stride "${PLOT_STRIDE}"
    )

    if [[ -n "${LEVEL}" ]]; then
        PLOT_ARGS+=(--level "${LEVEL}")
    else
        case "${RENDER_FIELD}" in
            rho) PLOT_ARGS+=(--fraction "${RHO_FRACTION}") ;;
            pressure) PLOT_ARGS+=(--fraction "${PRESSURE_FRACTION}") ;;
            current) PLOT_ARGS+=(--fraction "${CURRENT_FRACTION}") ;;
            Bmag) PLOT_ARGS+=(--fraction "${BMAG_FRACTION}") ;;
            *) PLOT_ARGS+=(--fraction "${FRACTION}") ;;
        esac
    fi
    if [[ -n "${MAX_TIME}" ]]; then
        PLOT_ARGS+=(--max-time "${MAX_TIME}")
    fi
    if [[ "${PAPER}" == "1" ]]; then
        PLOT_ARGS+=(--paper --paper-dpi "${PAPER_DPI}")
        if [[ "${PAPER_PNG_ONLY}" == "1" ]]; then
            PLOT_ARGS+=(--paper-png-only)
        fi
    fi
    if [[ "${OVERVIEW_ONLY}" == "1" ]]; then
        PLOT_ARGS+=(--overview-only)
    fi

    echo
    echo "----- RENDER ${RENDER_INDEX}/${RENDER_TOTAL}: ${RENDER_FIELD} -----"
    echo "Start: $(date --iso-8601=seconds)"
    printf "Command: %q -u visualization/plot_blast3d_volume.py" "${PYTHON_BIN}"
    printf " %q" "${PLOT_ARGS[@]}"
    printf "\n"
    RENDER_START="${SECONDS}"

    if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
        srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
            "${PYTHON_BIN}" -u visualization/plot_blast3d_volume.py \
            "${PLOT_ARGS[@]}"
    else
        "${PYTHON_BIN}" -u visualization/plot_blast3d_volume.py \
            "${PLOT_ARGS[@]}"
    fi

    echo "Done : ${RENDER_FIELD} in $((SECONDS - RENDER_START)) s"
done

echo
echo "===== RENDER COMPLETE ====="
echo "End    : $(date --iso-8601=seconds)"
echo "Results: ${WORKDIR}/${INPUT_DIR}"
echo "PNG files:"
find "${INPUT_DIR}" -maxdepth 1 -type f -name '*_3d*.png' -print | sort
