#!/bin/bash -l
#SBATCH -J hlld_y_ncu
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Targeted Nsight Compute profile for the optimized HLLD advance_y kernel.
# Defaults match the winning n=8 Orszag-Tang configuration:
#   canonical Y, X launch bound 3, Y launch bound 3, 16x8 blocks.
#
# The application executes 10 warm-up steps plus 3 measured/profiled steps.
# NCU skips the first 10 matching advance_y launches and collects the next 3.

set -euo pipefail

N="${N:-8}"
CASE="${CASE:-orszag_tang}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
PROFILE_STEPS="${PROFILE_STEPS:-3}"
NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-advance_y_kernel}"
NCU_SET="${NCU_SET:-targeted}"

HLLD_CANONICALIZE_Y="${HLLD_CANONICALIZE_Y:-1}"
ADVANCE_X_MIN_BLOCKS_PER_SM="${ADVANCE_X_MIN_BLOCKS_PER_SM:-3}"
ADVANCE_Y_MIN_BLOCKS_PER_SM="${ADVANCE_Y_MIN_BLOCKS_PER_SM:-3}"
PROFILE_NVCC_FLAGS="${PROFILE_NVCC_FLAGS:--lineinfo -Xptxas=-v}"
MAKE_CLEAN="${MAKE_CLEAN:-1}"

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
cd "${WORKDIR}"

is_positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_nonnegative_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }

if ! is_positive_integer "${N}" || \
   ! is_positive_integer "${PROFILE_STEPS}" || \
   ! is_nonnegative_integer "${WARMUP_STEPS}" || \
   ! is_nonnegative_integer "${ADVANCE_X_MIN_BLOCKS_PER_SM}" || \
   ! is_nonnegative_integer "${ADVANCE_Y_MIN_BLOCKS_PER_SM}"; then
    echo "[ERROR] Invalid numeric configuration."
    exit 2
fi

if [[ "${HLLD_CANONICALIZE_Y}" != "0" && "${HLLD_CANONICALIZE_Y}" != "1" ]]; then
    echo "[ERROR] HLLD_CANONICALIZE_Y must be 0 or 1."
    exit 2
fi

if [[ "${NCU_SET}" != "targeted" && "${NCU_SET}" != "full" ]]; then
    echo "[ERROR] NCU_SET must be targeted or full."
    exit 2
fi

mkdir -p logs profiling

if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -f /usr/share/Modules/init/bash ]; then
        source /usr/share/Modules/init/bash
    fi
fi

if command -v module >/dev/null 2>&1; then
    module purge
    module load rhel8/default-amp
fi

if ! command -v ncu >/dev/null 2>&1; then
    echo "[ERROR] ncu was not found after loading rhel8/default-amp."
    module spider nsight 2>&1 || true
    module spider cuda 2>&1 || true
    exit 127
fi

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
unset CUDA_LAUNCH_BLOCKING

PROFILE_DIR="${WORKDIR}/profiling/${SLURM_JOB_ID}_${CASE}_hlld_n${N}_ncu_y"
mkdir -p "${PROFILE_DIR}"

BASE="${PROFILE_DIR}/ncu_${CASE}_hlld_n${N}_advance_y"
REPORT="${BASE}.ncu-rep"
METADATA_FILE="${PROFILE_DIR}/metadata.txt"
BUILD_LOG="${PROFILE_DIR}/build.log"
RUN_LOG="${PROFILE_DIR}/validation_run.log"
CONSOLE_LOG="${BASE}_console.log"

{
    echo "date=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "workdir=${WORKDIR}"
    echo "N=${N}"
    echo "case=${CASE}"
    echo "solver=hlld"
    echo "warmup_steps=${WARMUP_STEPS}"
    echo "profile_steps=${PROFILE_STEPS}"
    echo "ncu_kernel_regex=${NCU_KERNEL_REGEX}"
    echo "ncu_set=${NCU_SET}"
    echo "hlld_canonicalize_y=${HLLD_CANONICALIZE_Y}"
    echo "advance_x_min_blocks_per_sm=${ADVANCE_X_MIN_BLOCKS_PER_SM}"
    echo "advance_y_min_blocks_per_sm=${ADVANCE_Y_MIN_BLOCKS_PER_SM}"
    echo "git_branch=$(git branch --show-current 2>/dev/null || echo unknown)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "ncu=$(ncu --version 2>&1 | tr '\n' ' ')"
    nvidia-smi --query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem \
        --format=csv,noheader 2>/dev/null || true
} | tee "${METADATA_FILE}"

echo "===== BUILD ====="
BUILD_NVCC_FLAGS="${PROFILE_NVCC_FLAGS}"
BUILD_NVCC_FLAGS+=" -DMHD_HLLD_CANONICALIZE_Y=${HLLD_CANONICALIZE_Y}"
BUILD_NVCC_FLAGS+=" -DMHD_ADVANCE_X_BLOCK_X=16 -DMHD_ADVANCE_X_BLOCK_Y=8"
BUILD_NVCC_FLAGS+=" -DMHD_ADVANCE_Y_BLOCK_X=16 -DMHD_ADVANCE_Y_BLOCK_Y=8"
BUILD_NVCC_FLAGS+=" -DMHD_ADVANCE_X_MIN_BLOCKS_PER_SM=${ADVANCE_X_MIN_BLOCKS_PER_SM}"
BUILD_NVCC_FLAGS+=" -DMHD_ADVANCE_Y_MIN_BLOCKS_PER_SM=${ADVANCE_Y_MIN_BLOCKS_PER_SM}"

if [ "${MAKE_CLEAN}" = "1" ]; then
    make clean
fi
make gpu NVCC_EXTRA_FLAGS="${BUILD_NVCC_FLAGS}" 2>&1 | tee "${BUILD_LOG}"

if [ ! -x ./bin/main_gpu ]; then
    echo "[ERROR] ./bin/main_gpu was not produced."
    exit 1
fi

APP=(
    ./bin/main_gpu
    "${N}"
    --case "${CASE}"
    --solver hlld
    --no-out
    --warmup-steps "${WARMUP_STEPS}"
    --benchmark-steps "${PROFILE_STEPS}"
)

printf "[INFO] Application command:"
printf " %q" "${APP[@]}"
printf "\n"

echo "===== UNPROFILED VALIDATION RUN ====="
"${APP[@]}" 2>&1 | tee "${RUN_LOG}"
if ! grep -q '^TUNING_CSV,' "${RUN_LOG}"; then
    echo "[ERROR] Validation run did not produce TUNING_CSV."
    exit 1
fi

echo "===== NSIGHT COMPUTE ====="
NCU_ARGS=(
    --force-overwrite
    --target-processes all
    --replay-mode kernel
    --kernel-name-base demangled
    --kernel-name "regex:${NCU_KERNEL_REGEX}"
    --launch-skip "${WARMUP_STEPS}"
    --launch-count "${PROFILE_STEPS}"
    --export "${BASE}"
)

if [ "${NCU_SET}" = "full" ]; then
    NCU_ARGS+=(--set full)
else
    AVAILABLE_SECTIONS="$(ncu --list-sections 2>&1 || true)"
    REQUESTED_SECTIONS=(
        LaunchStats
        Occupancy
        SpeedOfLight
        SpeedOfLight_RooflineChart
        ComputeWorkloadAnalysis
        MemoryWorkloadAnalysis
        MemoryWorkloadAnalysis_Chart
        SchedulerStats
        WarpStateStats
        InstructionStats
        SourceCounters
    )
    SELECTED_COUNT=0
    for section in "${REQUESTED_SECTIONS[@]}"; do
        if grep -q "${section}" <<< "${AVAILABLE_SECTIONS}"; then
            NCU_ARGS+=(--section "${section}")
            SELECTED_COUNT=$((SELECTED_COUNT + 1))
        fi
    done
    if [ "${SELECTED_COUNT}" -eq 0 ]; then
        NCU_ARGS+=(--set basic)
    fi
fi

printf "[INFO] Command: ncu"
printf " %q" "${NCU_ARGS[@]}"
printf " "
printf "%q " "${APP[@]}"
printf "\n"

set +e
ncu "${NCU_ARGS[@]}" "${APP[@]}" 2>&1 | tee "${CONSOLE_LOG}"
STATUS=${PIPESTATUS[0]}
set -e

if [ "${STATUS}" -ne 0 ]; then
    echo "[ERROR] Nsight Compute failed with status ${STATUS}."
    exit "${STATUS}"
fi

if [ ! -f "${REPORT}" ]; then
    echo "[ERROR] Expected report was not produced: ${REPORT}"
    exit 1
fi

ncu --import "${REPORT}" --page details --csv \
    > "${BASE}_details.csv" 2>&1 || true
ncu --import "${REPORT}" --page raw --csv \
    > "${BASE}_raw.csv" 2>&1 || true
ncu --import "${REPORT}" --page details --print-summary per-kernel \
    > "${BASE}_summary.txt" 2>&1 || true

echo "===== GENERATED FILES ====="
ls -lh "${PROFILE_DIR}"
echo "Profile directory: ${PROFILE_DIR}"
echo "Report           : ${REPORT}"
echo "Details CSV      : ${BASE}_details.csv"
echo "Raw CSV          : ${BASE}_raw.csv"
echo "Summary          : ${BASE}_summary.txt"
