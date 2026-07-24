#!/bin/bash -l
#SBATCH -J mhd3d_gpu
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# True CUDA 3D GLM-MHD run on one CSD3 Ampere GPU.
# Submit from the repository root after `mkdir -p logs`.
#
# Override example:
#   sbatch --export=ALL,CASE=imtg,RESOLUTION=128 \
#     scripts/csd3_slurm/slurm_gpu_3d.sh

set -euo pipefail

JOB_ID="${SLURM_JOB_ID:-manual}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SUBMIT_DIR}}"
cd "${WORKDIR}"

if [[ ! -f Makefile ]]; then
    echo "[ERROR] Submit from the MHD repository root or set WORKDIR."
    exit 1
fi
mkdir -p logs outputs validation

echo "===== MODULE SETUP ====="
if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash \
                /usr/local/Modules/init/bash; do
        if [[ -f "${init}" ]]; then source "${init}"; break; fi
    done
fi
if ! command -v module >/dev/null 2>&1; then
    echo "[ERROR] Environment Modules is unavailable."
    exit 1
fi
module purge
module load rhel8/default-amp
module list 2>&1 || true

for tool in nvcc nvidia-smi make; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "[ERROR] ${tool} is unavailable after loading rhel8/default-amp."
        exit 1
    fi
done

CASE="${CASE:-blast}"
if [[ "${CASE}" == "imtg" ]]; then
    # Glines, Grete & O'Shea (PRE 103, 043203), Ms0.2_Ma1.
    DEFAULT_RESOLUTION=1024
    DEFAULT_T_END=5.809475019311126
    # SNAPSHOTS is the number of intervals. Five intervals plus the t=0
    # initial condition produce exactly six uniformly spaced output files.
    DEFAULT_SNAPSHOTS=5
    DEFAULT_PLOT_FIELDS=rho,current,Bmag
    DEFAULT_RHO_FRACTION=0.10
    DEFAULT_PLOT_STRIDE=2
    CASE_STEM=imtg3d
elif [[ "${CASE}" == "blast" ]]; then
    DEFAULT_RESOLUTION=128
    DEFAULT_T_END=0.01
    DEFAULT_SNAPSHOTS=5
    DEFAULT_PLOT_FIELDS=rho
    DEFAULT_RHO_FRACTION=0.12
    DEFAULT_PLOT_STRIDE=1
    CASE_STEM=blast3d
else
    echo "[ERROR] CASE must be blast or imtg."
    exit 2
fi

RESOLUTION="${RESOLUTION:-${DEFAULT_RESOLUTION}}"
T_END="${T_END:-${DEFAULT_T_END}}"
SNAPSHOTS="${SNAPSHOTS:-${DEFAULT_SNAPSHOTS}}"
SOLVER="${SOLVER:-hlld}"
CFL="${CFL:-0.20}"
RUN_TEST="${RUN_TEST:-1}"
VISUALIZE="${VISUALIZE:-1}"
PLOT_FIELDS="${PLOT_FIELDS:-${FIELD:-${DEFAULT_PLOT_FIELDS}}}"
RHO_FRACTION="${RHO_FRACTION:-${DEFAULT_RHO_FRACTION}}"
CURRENT_LEVEL="${CURRENT_LEVEL:-4.5}"
BMAG_FRACTION="${BMAG_FRACTION:-0.15}"
FRACTION="${FRACTION:-0.18}"
LEVEL="${LEVEL:-}"
PNG_FRAMES="${PNG_FRAMES:-6}"
PLOT_STRIDE="${PLOT_STRIDE:-${DEFAULT_PLOT_STRIDE}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAKE_JOBS="${MAKE_JOBS:-8}"
OUT_DIR="${OUT_DIR:-outputs/${CASE_STEM}_gpu_n${RESOLUTION}_${SOLVER}_${JOB_ID}}"
BUILD_ROOT="${BUILD_ROOT:-build/csd3_gpu3d_${JOB_ID}}"
BIN_ROOT="${BIN_ROOT:-bin/csd3_gpu3d_${JOB_ID}}"

if ! [[ "${RESOLUTION}" =~ ^[1-9][0-9]*$ ]] || (( RESOLUTION < 8 )); then
    echo "[ERROR] RESOLUTION must be an integer >= 8."
    exit 2
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)"
COMPUTE_CAP="$(
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
    head -1 | tr -d '[:space:]' || true
)"
if [[ "${COMPUTE_CAP}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    CUDA_SM="${COMPUTE_CAP/./}"
elif [[ "${GPU_NAME}" == *A100* || "${GPU_NAME}" == *A30* ]]; then
    CUDA_SM=80
elif [[ "${GPU_NAME}" == *A10* || "${GPU_NAME}" == *A40* ]]; then
    CUDA_SM=86
else
    CUDA_SM="${CUDA_SM:-80}"
fi
CUDA_ARCH_FLAG="-arch=sm_${CUDA_SM}"

TOTAL_GPU_MIB="$(
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits |
    head -1 | tr -d '[:space:]'
)"
# Four full conserved-state grids dominate allocation. Add 25% headroom for
# reduction storage, runtime overhead, and temporary allocations.
GRID_EXTENT=$((RESOLUTION + 4))
EST_GPU_MIB=$((4 * GRID_EXTENT * GRID_EXTENT * GRID_EXTENT * 72 * 5 / 4 / 1048576))
if [[ "${TOTAL_GPU_MIB}" =~ ^[0-9]+$ ]] &&
   (( EST_GPU_MIB > TOTAL_GPU_MIB * 85 / 100 )); then
    echo "[ERROR] Estimated GPU allocation ${EST_GPU_MIB} MiB is too close to"
    echo "        the available ${TOTAL_GPU_MIB} MiB. Reduce RESOLUTION."
    if [[ "${CASE}" == "imtg" && "${RESOLUTION}" == "1024" ]]; then
        echo "        1024^3 is the paper grid, but this implementation stores"
        echo "        four double-precision 9-variable grids on one GPU."
        echo "        An exact-grid run requires multi-GPU domain decomposition,"
        echo "        which this single-GPU executable does not yet provide."
        echo "        Use RESOLUTION=512 on an 80-GiB A100, or a lower value"
        echo "        accepted by this preflight check, for the same physical case."
    fi
    exit 2
fi

mkdir -p "${OUT_DIR}"

echo "===== CSD3 CUDA 3D JOB ====="
echo "Job ID              : ${JOB_ID}"
echo "Host                : $(hostname)"
echo "Start               : $(date --iso-8601=seconds)"
echo "Workdir             : ${WORKDIR}"
echo "Git commit          : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "GPU                 : ${GPU_NAME}"
echo "GPU memory          : ${TOTAL_GPU_MIB:-unknown} MiB"
echo "Compute capability  : ${COMPUTE_CAP:-unknown}"
echo "CUDA architecture   : ${CUDA_ARCH_FLAG}"
echo "Estimated allocation: ${EST_GPU_MIB} MiB"
echo "Case                : ${CASE}"
echo "Resolution          : ${RESOLUTION}^3"
echo "Solver              : ${SOLVER}"
echo "t_end / CFL         : ${T_END} / ${CFL}"
echo "Output intervals    : ${SNAPSHOTS}"
echo "Output files        : $((SNAPSHOTS + 1)) (including t=0)"
echo "Plot fields         : ${PLOT_FIELDS}"
echo "Plot stride         : ${PLOT_STRIDE}"
echo "Output              : ${OUT_DIR}"
echo "Build root          : ${BUILD_ROOT}"
echo "Binary root         : ${BIN_ROOT}"
echo
nvidia-smi
nvcc --version

echo "===== BUILD CUDA 3D ====="
make -j "${MAKE_JOBS}" gpu_3d CUDA_ARCH="${CUDA_ARCH_FLAG}" \
    BUILD_DIR="${BUILD_ROOT}" BIN_DIR="${BIN_ROOT}"

if [[ "${RUN_TEST}" == "1" ]]; then
    echo "===== CPU/GPU ONE-STEP PARITY TEST ====="
    make test_gpu_3d CUDA_ARCH="${CUDA_ARCH_FLAG}" \
        BUILD_DIR="${BUILD_ROOT}" BIN_DIR="${BIN_ROOT}"
    echo "===== IMTG INITIAL-CONDITION TEST ====="
    make test_imtg_3d BUILD_DIR="${BUILD_ROOT}" BIN_DIR="${BIN_ROOT}"
fi

echo "===== PRODUCTION RUN ====="
APP=(
    "${BIN_ROOT}/main_gpu_3d"
    --case "${CASE}"
    --resolution "${RESOLUTION}"
    --t-end "${T_END}"
    --snapshots "${SNAPSHOTS}"
    --solver "${SOLVER}"
    --cfl "${CFL}"
    --out "${OUT_DIR}"
)
printf "Command:"; printf " %q" "${APP[@]}"; printf "\n"

if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -v srun --ntasks=1 "${APP[@]}"
else
    srun --ntasks=1 "${APP[@]}"
fi

echo "===== AUTOMATIC RENDERING ====="
if [[ "${VISUALIZE}" == "1" ]]; then
    export MPLBACKEND=Agg
    if command -v "${PYTHON_BIN}" >/dev/null 2>&1 &&
       "${PYTHON_BIN}" -c "import numpy, matplotlib, PIL" >/dev/null 2>&1; then
        IFS=',' read -r -a RENDER_FIELDS <<< "${PLOT_FIELDS}"
        RENDER_TOTAL="${#RENDER_FIELDS[@]}"
        RENDER_INDEX=0
        for RENDER_FIELD in "${RENDER_FIELDS[@]}"; do
            RENDER_INDEX=$((RENDER_INDEX + 1))
            RENDER_FIELD="${RENDER_FIELD//[[:space:]]/}"
            PLOT_ARGS=(
                --input "${OUT_DIR}"
                --field "${RENDER_FIELD}"
                --png-frames "${PNG_FRAMES}"
                --stride "${PLOT_STRIDE}"
            )
            case "${RENDER_FIELD}" in
                rho)
                    PLOT_ARGS+=(--fraction "${RHO_FRACTION}")
                    ;;
                current)
                    # |J|=4.5 keeps the analytic t=0 state visible.
                    PLOT_ARGS+=(--level "${CURRENT_LEVEL}")
                    ;;
                Bmag)
                    PLOT_ARGS+=(--fraction "${BMAG_FRACTION}")
                    ;;
                *)
                    PLOT_ARGS+=(--fraction "${FRACTION}")
                    if [[ -n "${LEVEL}" ]]; then
                        PLOT_ARGS+=(--level "${LEVEL}")
                    fi
                    ;;
            esac
            echo
            echo "----- RENDER ${RENDER_INDEX}/${RENDER_TOTAL}: ${RENDER_FIELD} -----"
            echo "Start: $(date --iso-8601=seconds)"
            printf "Command: %q -u visualization/plot_blast3d_volume.py" "${PYTHON_BIN}"
            printf " %q" "${PLOT_ARGS[@]}"
            printf "\n"
            RENDER_START="${SECONDS}"
            PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u \
                visualization/plot_blast3d_volume.py "${PLOT_ARGS[@]}"
            echo "Done : ${RENDER_FIELD} in $((SECONDS - RENDER_START)) s"
            echo "Plot : ${OUT_DIR}/${CASE_STEM}_${RENDER_FIELD}_3d_evolution.png"
        done
    else
        echo "[WARN] CUDA run succeeded, but rendering dependencies are absent."
        echo "[WARN] Render later with visualization/plot_blast3d_volume.py."
    fi
fi

echo "===== COMPLETE ====="
echo "End     : $(date --iso-8601=seconds)"
echo "Results : ${WORKDIR}/${OUT_DIR}"
