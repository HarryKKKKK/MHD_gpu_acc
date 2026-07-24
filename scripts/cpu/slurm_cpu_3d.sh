#!/bin/bash -l
#SBATCH --job-name=mhd3d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Generic one-node OpenMP job for the 3D magnetized-blast demonstration.
#
# Submit from the repository root.  Account and partition are intentionally
# not hard-coded because they differ between clusters:
#
#   mkdir -p logs
#   sbatch -A YOUR_ACCOUNT -p YOUR_PARTITION scripts/cpu/slurm_cpu_3d.sh
#
# Every run parameter can be overridden through sbatch --export, for example:
#
#   sbatch -A PROJECT -p icelake \
#     --cpus-per-task=64 --mem=32G --time=08:00:00 \
#     --export=ALL,RESOLUTION=96,SNAPSHOTS=10,VISUALIZE=1 \
#     scripts/cpu/slurm_cpu_3d.sh

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SUBMIT_DIR}}"
cd "${WORKDIR}"

if [[ ! -f Makefile ]]; then
    echo "[ERROR] Makefile not found in ${WORKDIR}."
    echo "Submit from the repository root or export WORKDIR=/path/to/MHD."
    exit 1
fi

# Cluster-specific modules can be supplied without editing this script:
#   --export=ALL,MODULES_STR="gcc/12.2 python/3.11"
if [[ -n "${MODULES_STR:-}" ]]; then
    if ! command -v module >/dev/null 2>&1; then
        echo "[ERROR] MODULES_STR was set, but the module command is unavailable."
        exit 1
    fi
    # Intentional word splitting: MODULES_STR is a space-separated module list.
    # shellcheck disable=SC2086
    module load ${MODULES_STR}
fi

CASE="${CASE:-blast}"
if [[ "${CASE}" == "imtg" ]]; then
    DEFAULT_T_END=2.0
    DEFAULT_SNAPSHOTS=8
    DEFAULT_FIELD=current
    DEFAULT_FRACTION=0.45
    DEFAULT_PLOT_STRIDE=2
    CASE_STEM=imtg3d
elif [[ "${CASE}" == "blast" ]]; then
    DEFAULT_T_END=0.01
    DEFAULT_SNAPSHOTS=5
    DEFAULT_FIELD=rho
    DEFAULT_FRACTION=0.12
    DEFAULT_PLOT_STRIDE=1
    CASE_STEM=blast3d
else
    echo "[ERROR] CASE must be blast or imtg."
    exit 2
fi

RESOLUTION="${RESOLUTION:-64}"
T_END="${T_END:-${DEFAULT_T_END}}"
SNAPSHOTS="${SNAPSHOTS:-${DEFAULT_SNAPSHOTS}}"
SOLVER="${SOLVER:-hlld}"
CFL="${CFL:-0.20}"
FIELD="${FIELD:-${DEFAULT_FIELD}}"
FRACTION="${FRACTION:-${DEFAULT_FRACTION}}"
PNG_FRAMES="${PNG_FRAMES:-6}"
PLOT_STRIDE="${PLOT_STRIDE:-${DEFAULT_PLOT_STRIDE}}"
VISUALIZE="${VISUALIZE:-1}"
RUN_TEST="${RUN_TEST:-1}"
MAKE_JOBS="${MAKE_JOBS:-8}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUT_DIR="${OUT_DIR:-outputs/${CASE_STEM}_n${RESOLUTION}_${SOLVER}_${JOB_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OMP_THREADS="${OMP_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export OMP_NUM_THREADS="${OMP_THREADS}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export MPLBACKEND=Agg

mkdir -p logs outputs "${OUT_DIR}"

echo "===== 3D MHD SLURM JOB ====="
echo "Job ID          : ${SLURM_JOB_ID:-manual}"
echo "Host            : $(hostname)"
echo "Start           : $(date)"
echo "Workdir         : ${WORKDIR}"
echo "Git commit      : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "OMP threads     : ${OMP_NUM_THREADS}"
echo "Case            : ${CASE}"
echo "Resolution      : ${RESOLUTION}^3"
echo "t_end           : ${T_END}"
echo "Snapshots       : ${SNAPSHOTS}"
echo "Solver          : ${SOLVER}"
echo "CFL             : ${CFL}"
echo "Output          : ${OUT_DIR}"
echo "Visualize       : ${VISUALIZE}"
echo

if ! command -v g++ >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1; then
    echo "[ERROR] g++ and make must be available. Load compiler modules with"
    echo "        --export=ALL,MODULES_STR=\"gcc/... python/...\""
    exit 1
fi

echo "===== BUILD ====="
make -j "${MAKE_JOBS}" cpu_3d
if [[ "${RUN_TEST}" == "1" ]]; then
    make test_3d
fi

echo "===== RUN ====="
RUN_CMD=(
    ./bin/main_cpu_3d
    --case "${CASE}"
    --resolution "${RESOLUTION}"
    --t-end "${T_END}"
    --snapshots "${SNAPSHOTS}"
    --solver "${SOLVER}"
    --cfl "${CFL}"
    --out "${OUT_DIR}"
)

if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -v srun --ntasks=1 --cpus-per-task="${OMP_THREADS}" \
        --cpu-bind=cores "${RUN_CMD[@]}"
else
    srun --ntasks=1 --cpus-per-task="${OMP_THREADS}" \
        --cpu-bind=cores "${RUN_CMD[@]}"
fi

echo "===== VISUALIZATION ====="
if [[ "${VISUALIZE}" == "1" ]]; then
    if command -v "${PYTHON_BIN}" >/dev/null 2>&1 &&
       "${PYTHON_BIN}" -c "import numpy, matplotlib, PIL" >/dev/null 2>&1; then
        "${PYTHON_BIN}" visualization/plot_blast3d_volume.py \
            --input "${OUT_DIR}" \
            --field "${FIELD}" \
            --fraction "${FRACTION}" \
            --png-frames "${PNG_FRAMES}" \
            --stride "${PLOT_STRIDE}"
    else
        echo "[WARN] Simulation completed, but visualization was skipped."
        echo "[WARN] ${PYTHON_BIN} needs numpy, matplotlib and Pillow."
        echo "[WARN] Run plot_blast3d_volume.py later on a login/visualization node."
    fi
else
    echo "VISUALIZE=0: snapshots were written without rendering."
fi

echo
echo "===== COMPLETE ====="
echo "Finished : $(date)"
echo "Results  : ${WORKDIR}/${OUT_DIR}"
