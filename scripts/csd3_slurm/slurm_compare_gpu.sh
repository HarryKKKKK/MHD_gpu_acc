#!/bin/bash -l
#SBATCH -J mhd_cmp_gpu
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

# Final GPU time-to-solution matrix on CSD3:
#   4 array tasks, one solver per task/GPU.
# Each task builds its solver once, then runs both cases and n={1,2,4,8}
# sequentially.  n=1/2 run three times; n=4/8 run once.  No field files
# are written.
# HLLD uses the measured canonical-Y + X/Y launch-bound-3 build.  The other
# solvers use the repository defaults, so their kernels are not changed by the
# HLLD-specific experiment.
#
# Submit from the repository root (create logs first because Slurm opens its
# output file before this script starts):
#   mkdir -p logs
#   sbatch scripts/csd3_slurm/slurm_compare_gpu.sh

SUBMIT_ROOT="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
COMMON_SCRIPT="${SUBMIT_ROOT}/scripts/csd3_slurm/comparison_common.sh"
if [ ! -f "${COMMON_SCRIPT}" ]; then
    echo "[ERROR] Cannot find comparison helper: ${COMMON_SCRIPT}"
    echo "[ERROR] Submit this job from the MHD repository root, or set WORKDIR."
    exit 1
fi
# shellcheck source=comparison_common.sh
source "${COMMON_SCRIPT}"

BACKEND="gpu"
FAILED_RUNS=0
THREADS_REPORTED=0
RANKS_REPORTED=0

comparison_init_config
comparison_load_module rhel8/default-amp
comparison_prepare_paths

if ! command -v nvcc >/dev/null 2>&1; then
    echo "[ERROR] nvcc is unavailable after loading rhel8/default-amp."
    exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
if [[ "${COMPUTE_CAP}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    CUDA_SM="${COMPUTE_CAP/./}"
else
    case "${GPU_NAME}" in
        *A100*|*A30*) CUDA_SM=80 ;;
        *A10*|*A40*) CUDA_SM=86 ;;
        *H100*|*H800*) CUDA_SM=90 ;;
        *) echo "[ERROR] Unknown compute capability for ${GPU_NAME}. Set CUDA_ARCH_FLAG."; exit 1 ;;
    esac
fi
CUDA_ARCH_FLAG="${CUDA_ARCH_FLAG:--arch=sm_${CUDA_SM}}"

if [ "${SOLVER_NAME}" = "hlld" ]; then
    BUILD_VARIANT="hlld_canonical_y_xlb3_ylb3"
    NVCC_COMPARISON_FLAGS="-DMHD_HLLD_CANONICALIZE_Y=1 -DMHD_ADVANCE_X_MIN_BLOCKS_PER_SM=3 -DMHD_ADVANCE_Y_MIN_BLOCKS_PER_SM=3"
else
    BUILD_VARIANT="repository_default"
    NVCC_COMPARISON_FLAGS=""
fi

COMPILER_INFO="$(nvcc --version 2>&1)"
comparison_write_metadata "${COMPILER_INFO}"
{
    echo "gpu_name=${GPU_NAME}"
    echo "compute_capability=${COMPUTE_CAP}"
    echo "cuda_arch_flag=${CUDA_ARCH_FLAG}"
    echo "nvcc_extra_flags=${NVCC_COMPARISON_FLAGS}"
    echo "nvidia_smi_begin"
    nvidia-smi || true
    echo "nvidia_smi_end"
} >> "${METADATA_FILE}"

echo "===== BUILD ${BUILD_VARIANT} ====="
BUILD_START_NS="$(date +%s%N)"
make -j "${MAKE_JOBS:-8}" gpu \
    BUILD_DIR="${BUILD_ROOT}" \
    BIN_DIR="${BIN_ROOT}" \
    CUDA_ARCH="${CUDA_ARCH_FLAG}" \
    NVCC_EXTRA_FLAGS="${NVCC_COMPARISON_FLAGS}"
BUILD_END_NS="$(date +%s%N)"
awk -v start="${BUILD_START_NS}" -v end="${BUILD_END_NS}" 'BEGIN {printf "build_wall_seconds=%.9f\n", (end-start)/1.0e9}' >> "${METADATA_FILE}"

BIN="${BIN_ROOT}/main_gpu"
if [ ! -x "${BIN}" ]; then
    echo "[ERROR] GPU binary was not produced: ${BIN}"
    exit 1
fi

unset CUDA_LAUNCH_BLOCKING
for N_SCALE_VALUE in "${SCALES[@]}"; do
    for CASE_VALUE in "${CASES[@]}"; do
        comparison_set_case_scale "${CASE_VALUE}" "${N_SCALE_VALUE}"
        for ((repeat = 1; repeat <= NUM_REPEATS; ++repeat)); do
            comparison_run_once "${repeat}" \
                "${BIN}" "${N_SCALE}" --case "${CASE_NAME}" --solver "${SOLVER_NAME}" --no-out
        done
    done
done

comparison_finish
