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
# GPU profiling script for CSD3 Ampere GPU nodes.
#
# Default configuration:
#   N       = 8
#   CASE    = shock_bubble
#   SOLVER  = hllc
#   MODE    = nsys
#
# PROFILE_MODE options:
#   nsys : application-level timeline profiling
#   ncu  : detailed kernel-level profiling
#   both : run nsys, then ncu
#
# Recommended workflow:
#
# 1. First run Nsight Systems:
#
#   PROFILE_MODE=nsys \
#   CASE=shock_bubble \
#   SOLVER=hllc \
#   N=8 \
#   sbatch scripts/slurm/run_gpu_profile_n8.sh
#
# 2. Inspect the generated kernel summary and choose a hotspot.
#
# 3. Run Nsight Compute on that kernel:
#
#   PROFILE_MODE=ncu \
#   CASE=shock_bubble \
#   SOLVER=hllc \
#   N=8 \
#   NCU_KERNEL_REGEX='kernel_name_fragment' \
#   NCU_LAUNCH_COUNT=1 \
#   sbatch scripts/slurm/run_gpu_profile_n8.sh
#
# Optional:
#
#   PROFILE_MODE=both ...
#
#   NCU_SET=full ...
#
#   NCU_LAUNCH_SKIP=20 ...
#
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PROFILE_MODE="${PROFILE_MODE:-nsys}"

N="${N:-8}"
CASE="${CASE:-shock_bubble}"
SOLVER="${SOLVER:-hllc}"

# Run the program once without profiling before collection.
WARMUP="${WARMUP:-1}"

# Nsight Compute options.
#
# Leave NCU_KERNEL_REGEX empty only for an exploratory run.
# It is much better to obtain the exact kernel name from nsys first.
NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-0}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-3}"

# targeted: selected performance sections
# full:     Nsight Compute full section set, much slower
NCU_SET="${NCU_SET:-targeted}"

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${SLURM_SUBMIT_DIR}"

cd "${WORKDIR}"

mkdir -p logs outputs validation scaling profiling

PROFILE_DIR="${WORKDIR}/profiling/${SLURM_JOB_ID}_${CASE}_${SOLVER}_n${N}"
mkdir -p "${PROFILE_DIR}"

# ------------------------------------------------------------
# Validate configuration
# ------------------------------------------------------------

case "${PROFILE_MODE}" in
    nsys|ncu|both)
        ;;
    *)
        echo "[ERROR] PROFILE_MODE must be nsys, ncu, or both."
        exit 2
        ;;
esac

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

command_supported() {
    local command_name="$1"
    local option_name="$2"

    "${command_name}" --help 2>&1 | grep -q -- "${option_name}"
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
echo "PROFILE_MODE       : ${PROFILE_MODE}"
echo "N                  : ${N}"
echo "CASE               : ${CASE}"
echo "SOLVER             : ${SOLVER}"
echo "WARMUP             : ${WARMUP}"
echo "NCU_KERNEL_REGEX   : ${NCU_KERNEL_REGEX:-<not set>}"
echo "NCU_LAUNCH_SKIP    : ${NCU_LAUNCH_SKIP}"
echo "NCU_LAUNCH_COUNT   : ${NCU_LAUNCH_COUNT}"
echo "NCU_SET            : ${NCU_SET}"
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
    echo "PROFILE_MODE=${PROFILE_MODE}"
    echo "N=${N}"
    echo "CASE=${CASE}"
    echo "SOLVER=${SOLVER}"
    echo "NCU_KERNEL_REGEX=${NCU_KERNEL_REGEX}"
    echo "NCU_LAUNCH_SKIP=${NCU_LAUNCH_SKIP}"
    echo "NCU_LAUNCH_COUNT=${NCU_LAUNCH_COUNT}"
    echo "NCU_SET=${NCU_SET}"

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
    echo "===== PROFILERS ====="
    which nsys || true
    nsys --version || true

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

if [ "${MAKE_CLEAN:-0}" = "1" ]; then
    make clean
fi

make gpu

if [ ! -x "./main_gpu" ]; then
    echo "[ERROR] ./main_gpu was not produced or is not executable."
    exit 1
fi

echo "[INFO] Binary:"
ls -lh ./main_gpu

# Application command stored as an array to preserve arguments safely.
APP=(
    ./main_gpu
    "${N}"
    --case "${CASE}"
    --solver "${SOLVER}"
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
# Nsight Systems
# ------------------------------------------------------------

run_nsys_profile() {
    require_command nsys

    echo ""
    echo "============================================================"
    echo "===== NSIGHT SYSTEMS PROFILE ====="
    echo "============================================================"

    local base="${PROFILE_DIR}/nsys_${CASE}_${SOLVER}_n${N}"
    local console_log="${base}_console.log"
    local report="${base}.nsys-rep"

    local nsys_args=(
        profile
        --trace=cuda,nvtx,osrt
        --sample=none
        --cpuctxsw=none
        --force-overwrite=true
        --output="${base}"
    )

    # Some installed versions expose CUDA memory usage tracing.
    if command_supported "nsys profile" "--cuda-memory-usage"; then
        nsys_args+=(--cuda-memory-usage=true)
    fi

    printf "[INFO] Command: nsys"
    printf " %q" "${nsys_args[@]}"
    printf " "
    printf "%q " "${APP[@]}"
    printf "\n"

    set +e
    /usr/bin/time \
        -f "[PROFILE_TIME] real_seconds=%e
[PROFILE_TIME] user_seconds=%U
[PROFILE_TIME] sys_seconds=%S
[PROFILE_TIME] max_rss_kb=%M" \
        nsys "${nsys_args[@]}" "${APP[@]}" \
        2>&1 | tee "${console_log}"

    local status=${PIPESTATUS[0]}
    set -e

    if [ "${status}" -ne 0 ]; then
        echo "[ERROR] Nsight Systems failed with status ${status}."
        echo "[ERROR] See ${console_log}."
        return "${status}"
    fi

    if [ ! -f "${report}" ]; then
        echo "[ERROR] Expected report not found: ${report}"
        return 1
    fi

    echo ""
    echo "[INFO] Nsight Systems report: ${report}"

    # Full readable summary.
    nsys stats "${report}" \
        > "${base}_stats.txt" 2>&1 || true

    # Kernel-time summary.
    nsys stats \
        --report cuda_gpu_kern_sum \
        --format csv \
        --output - \
        "${report}" \
        > "${base}_cuda_gpu_kern_sum.csv" 2>&1 || true

    # CUDA API summary, useful for finding cudaDeviceSynchronize,
    # cudaMemcpy and other host-side blocking operations.
    nsys stats \
        --report cuda_api_sum \
        --format csv \
        --output - \
        "${report}" \
        > "${base}_cuda_api_sum.csv" 2>&1 || true

    # Memory operation timing.
    nsys stats \
        --report cuda_gpu_mem_time_sum \
        --format csv \
        --output - \
        "${report}" \
        > "${base}_cuda_gpu_mem_time_sum.csv" 2>&1 || true

    # Detailed GPU trace. This can be large but is useful for ordering.
    nsys stats \
        --report cuda_gpu_trace \
        --format csv \
        --output - \
        "${report}" \
        > "${base}_cuda_gpu_trace.csv" 2>&1 || true

    echo ""
    echo "===== TOP GPU KERNELS ====="
    head -n 25 "${base}_cuda_gpu_kern_sum.csv" || true

    echo ""
    echo "===== TOP CUDA API CALLS ====="
    head -n 25 "${base}_cuda_api_sum.csv" || true

    echo ""
    echo "[INFO] Generated:"
    echo "  ${report}"
    echo "  ${base}_stats.txt"
    echo "  ${base}_cuda_gpu_kern_sum.csv"
    echo "  ${base}_cuda_api_sum.csv"
    echo "  ${base}_cuda_gpu_mem_time_sum.csv"
    echo "  ${base}_cuda_gpu_trace.csv"
}

# ------------------------------------------------------------
# Nsight Compute
# ------------------------------------------------------------

run_ncu_profile() {
    require_command ncu

    echo ""
    echo "============================================================"
    echo "===== NSIGHT COMPUTE PROFILE ====="
    echo "============================================================"

    local base="${PROFILE_DIR}/ncu_${CASE}_${SOLVER}_n${N}"
    local console_log="${base}_console.log"
    local report="${base}.ncu-rep"

    local ncu_args=(
        --force-overwrite
        --target-processes all
        --replay-mode kernel
        --launch-skip "${NCU_LAUNCH_SKIP}"
        --launch-count "${NCU_LAUNCH_COUNT}"
        --export "${base}"
    )

    if [ -n "${NCU_KERNEL_REGEX}" ]; then
        ncu_args+=(
            --kernel-name-base demangled
            --kernel-name "regex:${NCU_KERNEL_REGEX}"
        )

        echo "[INFO] Kernel filter: regex:${NCU_KERNEL_REGEX}"
    else
        echo "[WARN] NCU_KERNEL_REGEX is empty."
        echo "[WARN] Nsight Compute will profile the first matching launches."
        echo "[WARN] These may be initialization or boundary kernels."
        echo "[WARN] Run nsys first and use a hotspot kernel name."
    fi

    if [ "${NCU_SET}" = "full" ]; then
        echo "[WARN] Using the full Nsight Compute section set."
        echo "[WARN] This can require many kernel replays."

        ncu_args+=(--set full)
    else
        echo "[INFO] Selecting targeted Nsight Compute sections."

        # Discover sections supported by the installed version.
        local available_sections
        available_sections="$(ncu --list-sections 2>&1 || true)"

        local requested_sections=(
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

        local selected_count=0
        local section

        for section in "${requested_sections[@]}"; do
            if grep -q "${section}" <<< "${available_sections}"; then
                ncu_args+=(--section "${section}")
                echo "[INFO] Enabled section: ${section}"
                selected_count=$((selected_count + 1))
            else
                echo "[INFO] Section unavailable in installed version: ${section}"
            fi
        done

        if [ "${selected_count}" -eq 0 ]; then
            echo "[WARN] Could not identify targeted sections."
            echo "[WARN] Falling back to Nsight Compute basic set."
            ncu_args+=(--set basic)
        fi
    fi

    printf "[INFO] Command: ncu"
    printf " %q" "${ncu_args[@]}"
    printf " "
    printf "%q " "${APP[@]}"
    printf "\n"

    set +e
    /usr/bin/time \
        -f "[PROFILE_TIME] real_seconds=%e
[PROFILE_TIME] user_seconds=%U
[PROFILE_TIME] sys_seconds=%S
[PROFILE_TIME] max_rss_kb=%M" \
        ncu "${ncu_args[@]}" "${APP[@]}" \
        2>&1 | tee "${console_log}"

    local status=${PIPESTATUS[0]}
    set -e

    if [ "${status}" -ne 0 ]; then
        echo "[ERROR] Nsight Compute failed with status ${status}."
        echo "[ERROR] See ${console_log}."
        echo ""
        echo "Common cluster-side causes include:"
        echo "  - GPU performance-counter permission is disabled;"
        echo "  - another profiler is using the performance monitor;"
        echo "  - the kernel regex matched no kernel;"
        echo "  - too many launches or sections were selected."
        return "${status}"
    fi

    if [ ! -f "${report}" ]; then
        echo "[ERROR] Expected report not found: ${report}"
        return 1
    fi

    echo ""
    echo "[INFO] Nsight Compute report: ${report}"

    # Section-oriented output with rule results.
    ncu \
        --import "${report}" \
        --page details \
        --csv \
        > "${base}_details.csv" 2>&1 || true

    # All raw collected metrics.
    ncu \
        --import "${report}" \
        --page raw \
        --csv \
        > "${base}_raw.csv" 2>&1 || true

    # Per-kernel summary.
    ncu \
        --import "${report}" \
        --page details \
        --print-summary per-kernel \
        > "${base}_summary.txt" 2>&1 || true

    echo ""
    echo "[INFO] Generated:"
    echo "  ${report}"
    echo "  ${base}_details.csv"
    echo "  ${base}_raw.csv"
    echo "  ${base}_summary.txt"
}

# ------------------------------------------------------------
# Execute requested profiling mode
# ------------------------------------------------------------

case "${PROFILE_MODE}" in
    nsys)
        run_nsys_profile
        ;;

    ncu)
        run_ncu_profile
        ;;

    both)
        run_nsys_profile
        run_ncu_profile
        ;;
esac

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
echo "Metadata            : ${METADATA_FILE}"