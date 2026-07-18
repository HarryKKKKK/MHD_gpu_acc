#!/bin/bash -l
#SBATCH -J sys_eq_gpu_profile
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ============================================================
# Nsight Compute (ncu) profiling on CSD3 Ampere GPU nodes.
#
# This script only produces Nsight Compute output (no Nsight Systems
# pass). ncu replays each captured kernel launch to collect metrics, so
# only NCU_LAUNCH_COUNT launches (after skipping NCU_LAUNCH_SKIP) are
# profiled per submission — sweeping many (case, solver, n) combos the
# way slurm_gpu.sh does would take far too long under replay.
#
# Default configuration:
#   N       = 8
#   CASE    = shock_bubble
#   SOLVER  = hllc
#
# Recommended workflow:
#
# 1. Exploratory run — leave NCU_KERNEL_REGEX empty to see what the
#    first NCU_LAUNCH_COUNT kernel launches actually are (one solver
#    step launches ~10 kernels: dt-reduction, advance_x, boundary x2,
#    advance_y, boundary x2, psi damping):
#
#   CASE=shock_bubble SOLVER=hllc N=8 \
#   sbatch scripts/csd3_slurm/profile_gpu.sh
#
# 2. Inspect "${base}_summary.txt" / "${base}_details.csv" for the
#    kernel names, pick the hotspot, then re-run targeted at just that
#    kernel with a couple of replays:
#
#   CASE=shock_bubble SOLVER=hllc N=8 \
#   NCU_KERNEL_REGEX='advance_x_kernel' \
#   NCU_LAUNCH_COUNT=3 \
#   sbatch scripts/csd3_slurm/profile_gpu.sh
#
# Optional:
#
#   NCU_SET=full ...        # full section set instead of targeted (slow)
#   NCU_LAUNCH_SKIP=20 ...  # skip past startup/warm-up launches
#   PROFILE_NVCC_FLAGS=...  # override profiling compile flags
#   ADVANCE_X_MIN_BLOCKS_PER_SM=0  # disable x launch bounds
#   ADVANCE_Y_MIN_BLOCKS_PER_SM=3  # enable the tested y launch bounds
#   MAKE_CLEAN=0 ...        # reuse objects (not recommended for comparisons)
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

N="${N:-8}"
CASE="${CASE:-orszag_tang}"
SOLVER="${SOLVER:-hlld}"

# Run the program once without profiling before collection.
WARMUP="${WARMUP:-1}"

# Nsight Compute options.
#
# Leave NCU_KERNEL_REGEX empty only for an exploratory run — profiling
# the first matching launches may just catch initialization/boundary
# kernels rather than the hot loop.
NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-0}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-12}"

# targeted: selected performance sections
# full:     Nsight Compute full section set, much slower
NCU_SET="${NCU_SET:-targeted}"

# A profiling build needs line information for source correlation and ptxas
# resource statistics for detecting register spills.  Clean by default because
# make cannot otherwise tell that command-line NVCC flags changed between the
# baseline and an experimental build.
PROFILE_NVCC_FLAGS="${PROFILE_NVCC_FLAGS:--lineinfo -Xptxas=-v}"
ADVANCE_X_MIN_BLOCKS_PER_SM="${ADVANCE_X_MIN_BLOCKS_PER_SM:-3}"
ADVANCE_Y_MIN_BLOCKS_PER_SM="${ADVANCE_Y_MIN_BLOCKS_PER_SM:-0}"
MAKE_CLEAN="${MAKE_CLEAN:-1}"

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${SLURM_SUBMIT_DIR}"

cd "${WORKDIR}"

mkdir -p logs profiling

PROFILE_DIR="${WORKDIR}/profiling/${SLURM_JOB_ID}_${CASE}_${SOLVER}_n${N}"
mkdir -p "${PROFILE_DIR}"

# ------------------------------------------------------------
# Validate configuration
# ------------------------------------------------------------

if ! [[ "${N}" =~ ^[0-9]+$ ]] || [ "${N}" -le 0 ]; then
    echo "[ERROR] N must be a positive integer."
    exit 2
fi

if ! [[ "${NCU_LAUNCH_SKIP}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] NCU_LAUNCH_SKIP must be a non-negative integer."
    exit 2
fi

if ! [[ "${NCU_LAUNCH_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] NCU_LAUNCH_COUNT must be a positive integer."
    exit 2
fi

if ! [[ "${ADVANCE_X_MIN_BLOCKS_PER_SM}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] ADVANCE_X_MIN_BLOCKS_PER_SM must be a non-negative integer."
    exit 2
fi

if ! [[ "${ADVANCE_Y_MIN_BLOCKS_PER_SM}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] ADVANCE_Y_MIN_BLOCKS_PER_SM must be a non-negative integer."
    exit 2
fi

# ------------------------------------------------------------
# Module setup
# ------------------------------------------------------------

echo "===== MODULE SETUP ====="

if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -f /usr/share/Modules/init/bash ]; then
        source /usr/share/Modules/init/bash
    elif [ -f /usr/local/Modules/init/bash ]; then
        source /usr/local/Modules/init/bash
    fi
fi

if command -v module >/dev/null 2>&1; then
    module purge
    module load rhel8/default-amp
    echo "[INFO] Loaded module: rhel8/default-amp"
else
    echo "[WARN] module command is unavailable."
    echo "[WARN] Continuing with the current environment."
fi

# Keep the host side lightweight.
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Avoid accidental debug-style synchronous execution.
unset CUDA_LAUNCH_BLOCKING

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

require_command() {
    local command_name="$1"

    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "[ERROR] Required command not found: ${command_name}"
        echo ""
        echo "Available related modules:"
        if command -v module >/dev/null 2>&1; then
            module spider nsight 2>&1 || true
            module spider cuda 2>&1 || true
        fi
        exit 127
    fi
}

# ------------------------------------------------------------
# Environment and metadata
# ------------------------------------------------------------

echo ""
echo "===== JOB CONFIGURATION ====="
echo "Job ID             : ${SLURM_JOB_ID}"
echo "Host               : $(hostname)"
echo "Start              : $(date --iso-8601=seconds)"
echo "Workdir            : ${WORKDIR}"
echo "Profile directory  : ${PROFILE_DIR}"
echo "Partition          : ${SLURM_JOB_PARTITION:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "N                  : ${N}"
echo "CASE               : ${CASE}"
echo "SOLVER             : ${SOLVER}"
echo "WARMUP             : ${WARMUP}"
echo "NCU_KERNEL_REGEX   : ${NCU_KERNEL_REGEX:-<not set>}"
echo "NCU_LAUNCH_SKIP    : ${NCU_LAUNCH_SKIP}"
echo "NCU_LAUNCH_COUNT   : ${NCU_LAUNCH_COUNT}"
echo "NCU_SET            : ${NCU_SET}"
echo "PROFILE_NVCC_FLAGS : ${PROFILE_NVCC_FLAGS}"
echo "ADVANCE_X_MIN_BLOCKS: ${ADVANCE_X_MIN_BLOCKS_PER_SM}"
echo "ADVANCE_Y_MIN_BLOCKS: ${ADVANCE_Y_MIN_BLOCKS_PER_SM}"
echo "MAKE_CLEAN         : ${MAKE_CLEAN}"
echo ""

METADATA_FILE="${PROFILE_DIR}/metadata.txt"

{
    echo "===== DATE ====="
    date --iso-8601=seconds

    echo ""
    echo "===== HOST ====="
    hostname
    uname -a

    echo ""
    echo "===== SLURM ====="
    echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
    echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unknown}"
    echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unknown}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

    echo ""
    echo "===== PROFILE CONFIGURATION ====="
    echo "N=${N}"
    echo "CASE=${CASE}"
    echo "SOLVER=${SOLVER}"
    echo "NCU_KERNEL_REGEX=${NCU_KERNEL_REGEX}"
    echo "NCU_LAUNCH_SKIP=${NCU_LAUNCH_SKIP}"
    echo "NCU_LAUNCH_COUNT=${NCU_LAUNCH_COUNT}"
    echo "NCU_SET=${NCU_SET}"
    echo "PROFILE_NVCC_FLAGS=${PROFILE_NVCC_FLAGS}"
    echo "ADVANCE_X_MIN_BLOCKS_PER_SM=${ADVANCE_X_MIN_BLOCKS_PER_SM}"
    echo "ADVANCE_Y_MIN_BLOCKS_PER_SM=${ADVANCE_Y_MIN_BLOCKS_PER_SM}"
    echo "MAKE_CLEAN=${MAKE_CLEAN}"

    echo ""
    echo "===== CPU ====="
    lscpu || true

    echo ""
    echo "===== GPU ====="
    nvidia-smi || true

    echo ""
    echo "===== GPU QUERY ====="
    nvidia-smi \
        --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.total,memory.used \
        --format=csv || true

    echo ""
    echo "===== MODULES ====="
    module list 2>&1 || true

    echo ""
    echo "===== COMPILERS ====="
    which g++ || true
    g++ --version || true

    which nvcc || true
    nvcc --version || true

    echo ""
    echo "===== PROFILER ====="
    which ncu || true
    ncu --version || true

    echo ""
    echo "===== GIT ====="
    git rev-parse --show-toplevel 2>/dev/null || true
    git branch --show-current 2>/dev/null || true
    git rev-parse HEAD 2>/dev/null || true
    git status --short 2>/dev/null || true
} | tee "${METADATA_FILE}"

# ------------------------------------------------------------
# Build
# ------------------------------------------------------------

echo ""
echo "===== BUILD ====="

BUILD_LOG="${PROFILE_DIR}/build.log"
BUILD_NVCC_FLAGS="${PROFILE_NVCC_FLAGS}"
BUILD_NVCC_FLAGS+=" -DMHD_ADVANCE_X_MIN_BLOCKS_PER_SM=${ADVANCE_X_MIN_BLOCKS_PER_SM}"
BUILD_NVCC_FLAGS+=" -DMHD_ADVANCE_Y_MIN_BLOCKS_PER_SM=${ADVANCE_Y_MIN_BLOCKS_PER_SM}"

set +e
{
    if [ "${MAKE_CLEAN}" = "1" ]; then
        make clean
    fi

    make gpu NVCC_EXTRA_FLAGS="${BUILD_NVCC_FLAGS}"
} 2>&1 | tee "${BUILD_LOG}"

BUILD_STATUS=${PIPESTATUS[0]}
set -e

if [ "${BUILD_STATUS}" -ne 0 ]; then
    echo "[ERROR] GPU build failed with status ${BUILD_STATUS}."
    echo "[ERROR] See ${BUILD_LOG}."
    exit "${BUILD_STATUS}"
fi

BIN="./bin/main_gpu"

if [ ! -x "${BIN}" ]; then
    echo "[ERROR] ${BIN} was not produced or is not executable."
    exit 1
fi

echo "[INFO] Binary:"
ls -lh "${BIN}"

# Application command stored as an array to preserve arguments safely.
APP=(
    "${BIN}"
    "${N}"
    --case "${CASE}"
    --solver "${SOLVER}"
    --no-out
)

printf "[INFO] Application command:"
printf " %q" "${APP[@]}"
printf "\n"

# ------------------------------------------------------------
# Warm-up
# ------------------------------------------------------------

if [ "${WARMUP}" = "1" ]; then
    echo ""
    echo "===== WARM-UP RUN ====="

    WARMUP_LOG="${PROFILE_DIR}/warmup.log"

    set +e
    /usr/bin/time \
        -f "[TIME] real_seconds=%e
[TIME] user_seconds=%U
[TIME] sys_seconds=%S
[TIME] max_rss_kb=%M" \
        "${APP[@]}" >"${WARMUP_LOG}" 2>&1

    WARMUP_STATUS=$?
    set -e

    if [ "${WARMUP_STATUS}" -ne 0 ]; then
        echo "[ERROR] Warm-up failed with status ${WARMUP_STATUS}."
        echo "[ERROR] See ${WARMUP_LOG}."
        tail -n 50 "${WARMUP_LOG}" || true
        exit "${WARMUP_STATUS}"
    fi

    echo "[INFO] Warm-up completed."
    tail -n 20 "${WARMUP_LOG}" || true
fi

# ------------------------------------------------------------
# Nsight Compute
# ------------------------------------------------------------

require_command ncu

echo ""
echo "============================================================"
echo "===== NSIGHT COMPUTE PROFILE ====="
echo "============================================================"

BASE="${PROFILE_DIR}/ncu_${CASE}_${SOLVER}_n${N}"
CONSOLE_LOG="${BASE}_console.log"
REPORT="${BASE}.ncu-rep"

NCU_ARGS=(
    --force-overwrite
    --target-processes all
    --replay-mode kernel
    --launch-skip "${NCU_LAUNCH_SKIP}"
    --launch-count "${NCU_LAUNCH_COUNT}"
    --export "${BASE}"
)

# Nsight Compute versions differ in whether source embedding is available.
# Enable it when supported so a copied .ncu-rep remains source-correlated away
# from the cluster filesystem.
if ncu --help 2>&1 | grep -q -- "--import-source"; then
    NCU_ARGS+=(--import-source yes)
    echo "[INFO] Source files will be embedded in the Nsight Compute report."
else
    echo "[INFO] Installed Nsight Compute does not expose --import-source."
fi

if [ -n "${NCU_KERNEL_REGEX}" ]; then
    NCU_ARGS+=(
        --kernel-name-base demangled
        --kernel-name "regex:${NCU_KERNEL_REGEX}"
    )

    echo "[INFO] Kernel filter: regex:${NCU_KERNEL_REGEX}"
else
    echo "[WARN] NCU_KERNEL_REGEX is empty."
    echo "[WARN] Nsight Compute will profile the first ${NCU_LAUNCH_COUNT} kernel launches."
    echo "[WARN] Use this exploratory pass to find kernel names, then re-run with a regex."
fi

if [ "${NCU_SET}" = "full" ]; then
    echo "[WARN] Using the full Nsight Compute section set."
    echo "[WARN] This can require many kernel replays."

    NCU_ARGS+=(--set full)
else
    echo "[INFO] Selecting targeted Nsight Compute sections."

    # Discover sections supported by the installed version.
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

    for SECTION in "${REQUESTED_SECTIONS[@]}"; do
        if grep -q "${SECTION}" <<< "${AVAILABLE_SECTIONS}"; then
            NCU_ARGS+=(--section "${SECTION}")
            echo "[INFO] Enabled section: ${SECTION}"
            SELECTED_COUNT=$((SELECTED_COUNT + 1))
        else
            echo "[INFO] Section unavailable in installed version: ${SECTION}"
        fi
    done

    if [ "${SELECTED_COUNT}" -eq 0 ]; then
        echo "[WARN] Could not identify targeted sections."
        echo "[WARN] Falling back to Nsight Compute basic set."
        NCU_ARGS+=(--set basic)
    fi
fi

printf "[INFO] Command: ncu"
printf " %q" "${NCU_ARGS[@]}"
printf " "
printf "%q " "${APP[@]}"
printf "\n"

set +e
/usr/bin/time \
    -f "[PROFILE_TIME] real_seconds=%e
[PROFILE_TIME] user_seconds=%U
[PROFILE_TIME] sys_seconds=%S
[PROFILE_TIME] max_rss_kb=%M" \
    ncu "${NCU_ARGS[@]}" "${APP[@]}" \
    2>&1 | tee "${CONSOLE_LOG}"

STATUS=${PIPESTATUS[0]}
set -e

if [ "${STATUS}" -ne 0 ]; then
    echo "[ERROR] Nsight Compute failed with status ${STATUS}."
    echo "[ERROR] See ${CONSOLE_LOG}."
    echo ""
    echo "Common cluster-side causes include:"
    echo "  - GPU performance-counter permission is disabled;"
    echo "  - another profiler is using the performance monitor;"
    echo "  - the kernel regex matched no kernel;"
    echo "  - too many launches or sections were selected."
    exit "${STATUS}"
fi

if [ ! -f "${REPORT}" ]; then
    echo "[ERROR] Expected report not found: ${REPORT}"
    exit 1
fi

echo ""
echo "[INFO] Nsight Compute report: ${REPORT}"

# Section-oriented output with rule results.
ncu \
    --import "${REPORT}" \
    --page details \
    --csv \
    > "${BASE}_details.csv" 2>&1 || true

# All raw collected metrics.
ncu \
    --import "${REPORT}" \
    --page raw \
    --csv \
    > "${BASE}_raw.csv" 2>&1 || true

# Per-kernel summary.
ncu \
    --import "${REPORT}" \
    --page details \
    --print-summary per-kernel \
    > "${BASE}_summary.txt" 2>&1 || true

echo ""
echo "[INFO] Generated:"
echo "  ${BUILD_LOG}"
echo "  ${REPORT}"
echo "  ${BASE}_details.csv"
echo "  ${BASE}_raw.csv"
echo "  ${BASE}_summary.txt"

# ------------------------------------------------------------
# Final GPU state
# ------------------------------------------------------------

echo ""
echo "===== FINAL GPU STATE ====="

nvidia-smi \
    --query-gpu=index,name,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used \
    --format=csv || true

echo ""
echo "===== PROFILE COMPLETE ====="
echo "End time           : $(date --iso-8601=seconds)"
echo "Profile directory  : ${PROFILE_DIR}"
echo "Metadata           : ${METADATA_FILE}"
