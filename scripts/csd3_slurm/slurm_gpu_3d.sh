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
#   sbatch --export=ALL,RESOLUTION=128,SNAPSHOTS=12,VISUALIZE=0 \
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

RESOLUTION="${RESOLUTION:-128}"
T_END="${T_END:-${DEFAULT_T_END}}"
SNAPSHOTS="${SNAPSHOTS:-${DEFAULT_SNAPSHOTS}}"
SOLVER="${SOLVER:-hlld}"
CFL="${CFL:-0.20}"
RUN_TEST="${RUN_TEST:-1}"
VISUALIZE="${VISUALIZE:-1}"
FIELD="${FIELD:-${DEFAULT_FIELD}}"
FRACTION="${FRACTION:-${DEFAULT_FRACTION}}"
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
echo "Snapshots           : ${SNAPSHOTS}"
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

echo "===== OPTIONAL RENDERING ====="
if [[ "${VISUALIZE}" == "1" ]]; then
    export MPLBACKEND=Agg
    if command -v "${PYTHON_BIN}" >/dev/null 2>&1 &&
       "${PYTHON_BIN}" -c "import numpy, matplotlib, PIL" >/dev/null 2>&1; then
        "${PYTHON_BIN}" visualization/plot_blast3d_volume.py \
            --input "${OUT_DIR}" --field "${FIELD}" \
            --fraction "${FRACTION}" --png-frames "${PNG_FRAMES}" \
            --stride "${PLOT_STRIDE}"
    else
        echo "[WARN] CUDA run succeeded, but rendering dependencies are absent."
        echo "[WARN] Render later with visualization/plot_blast3d_volume.py."
    fi
fi

echo "===== COMPLETE ====="
echo "End     : $(date --iso-8601=seconds)"
echo "Results : ${WORKDIR}/${OUT_DIR}"
