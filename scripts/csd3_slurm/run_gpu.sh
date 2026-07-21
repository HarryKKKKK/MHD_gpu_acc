#!/bin/bash -l
#SBATCH -J slurm_gpu
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# ============================================================
# GPU execution script
#
# This script:
#   1. Builds the GPU executable once.
#   2. Runs the same case and solver twice.
#   3. Keeps numerical output from both runs.
#   4. Records console logs and timing information separately.
#
# Default configuration:
#   N       = 8
#   CASE    = orszag_tang
#   SOLVER  = hlld
#
# Example:
#   sbatch scripts/csd3_slurm/run_gpu_twice.sh
#
# Override configuration:
#   CASE=rotor SOLVER=hlld N=1 \
#   sbatch scripts/csd3_slurm/run_gpu_twice.sh
# ============================================================

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

N="${N:-8}"
CASE="${CASE:-orszag_tang}"
SOLVER="${SOLVER:-hlld}"
NUM_RUNS="${NUM_RUNS:-2}"
MAKE_CLEAN="${MAKE_CLEAN:-0}"

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"

cd "${WORKDIR}"

# ------------------------------------------------------------
# Validate configuration
# ------------------------------------------------------------

if ! [[ "${N}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] N must be a positive integer."
    exit 2
fi

if ! [[ "${NUM_RUNS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] NUM_RUNS must be a positive integer."
    exit 2
fi

# ------------------------------------------------------------
# Output directories
# ------------------------------------------------------------

mkdir -p logs outputs validation

RUN_ROOT="${WORKDIR}/outputs/${SLURM_JOB_ID}_${CASE}_${SOLVER}_n${N}"
mkdir -p "${RUN_ROOT}"

SUMMARY_FILE="${WORKDIR}/validation/gpu_${SLURM_JOB_ID}_${CASE}_${SOLVER}_n${N}.csv"

echo \
"run,case,solver,n,real_seconds,user_seconds,sys_seconds,max_rss_kb,output_dir,git_branch,git_commit" \
> "${SUMMARY_FILE}"

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

# ------------------------------------------------------------
# Runtime environment
# ------------------------------------------------------------

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Avoid accidental synchronous CUDA debugging.
unset CUDA_LAUNCH_BLOCKING

# ------------------------------------------------------------
# Git information
# ------------------------------------------------------------

GIT_BRANCH=$(
    git branch --show-current 2>/dev/null || echo "unknown"
)

GIT_COMMIT=$(
    git rev-parse --short HEAD 2>/dev/null || echo "unknown"
)

# ------------------------------------------------------------
# Job information
# ------------------------------------------------------------

echo ""
echo "===== JOB CONFIGURATION ====="
echo "Job ID              : ${SLURM_JOB_ID}"
echo "Host                : $(hostname)"
echo "Start               : $(date --iso-8601=seconds)"
echo "Workdir             : ${WORKDIR}"
echo "Partition           : ${SLURM_JOB_PARTITION:-unknown}"
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "OMP_NUM_THREADS     : ${OMP_NUM_THREADS}"
echo "Git branch          : ${GIT_BRANCH}"
echo "Git commit          : ${GIT_COMMIT}"
echo "N                   : ${N}"
echo "CASE                : ${CASE}"
echo "SOLVER              : ${SOLVER}"
echo "Number of runs      : ${NUM_RUNS}"
echo "Run root            : ${RUN_ROOT}"
echo "Summary file        : ${SUMMARY_FILE}"

echo ""
echo "===== GPU INFO ====="
nvidia-smi || true

echo ""
echo "===== CPU INFO ====="
lscpu | grep -E \
    'CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)' \
    || true

# ------------------------------------------------------------
# Build
# ------------------------------------------------------------

echo ""
echo "===== BUILD ====="

make clean
make gpu

BIN="./bin/main_gpu"

if [ ! -x "${BIN}" ]; then
    echo "[ERROR] ${BIN} was not produced or is not executable."
    exit 1
fi

echo "[INFO] Binary:"
ls -lh "${BIN}"

# ------------------------------------------------------------
# Run function
# ------------------------------------------------------------

run_once() {
    local run_index="$1"

    local output_dir="${RUN_ROOT}/run${run_index}"
    local console_log="${RUN_ROOT}/run${run_index}.log"
    local timing_file="${RUN_ROOT}/run${run_index}_time.txt"

    mkdir -p "${output_dir}"

    local app=(
        "${BIN}"
        "${N}"
        --case "${CASE}"
        --solver "${SOLVER}"
        --out "${output_dir}"
    )

    echo ""
    echo "============================================================"
    echo "===== GPU RUN ${run_index}/${NUM_RUNS} ====="
    echo "============================================================"
    echo "Output directory : ${output_dir}"
    echo "Console log      : ${console_log}"
    echo "Timing file      : ${timing_file}"

    printf "Command          :"
    printf " %q" "${app[@]}"
    printf "\n"

    set +e

    /usr/bin/time \
        -f "real_seconds=%e
user_seconds=%U
sys_seconds=%S
max_rss_kb=%M" \
        -o "${timing_file}" \
        "${app[@]}" \
        2>&1 | tee "${console_log}"

    local run_status=${PIPESTATUS[0]}

    set -e

    if [ "${run_status}" -ne 0 ]; then
        echo "[ERROR] Run ${run_index} failed with status ${run_status}."
        echo "[ERROR] See log: ${console_log}"
        exit "${run_status}"
    fi

    local real_seconds
    local user_seconds
    local sys_seconds
    local max_rss_kb

    real_seconds=$(
        awk -F '=' '/^real_seconds=/{print $2; exit}' "${timing_file}"
    )

    user_seconds=$(
        awk -F '=' '/^user_seconds=/{print $2; exit}' "${timing_file}"
    )

    sys_seconds=$(
        awk -F '=' '/^sys_seconds=/{print $2; exit}' "${timing_file}"
    )

    max_rss_kb=$(
        awk -F '=' '/^max_rss_kb=/{print $2; exit}' "${timing_file}"
    )

    real_seconds="${real_seconds:-unknown}"
    user_seconds="${user_seconds:-unknown}"
    sys_seconds="${sys_seconds:-unknown}"
    max_rss_kb="${max_rss_kb:-unknown}"

    echo ""
    echo "------------------------------------------------------------"
    echo "[RUN COMPLETED] Run ${run_index}"
    echo "[TIMING] Real    : ${real_seconds} s"
    echo "[TIMING] User    : ${user_seconds} s"
    echo "[TIMING] System  : ${sys_seconds} s"
    echo "[MEMORY] Max RSS : ${max_rss_kb} KB"
    echo "[OUTPUT]         : ${output_dir}"
    echo "[LOG]            : ${console_log}"
    echo "------------------------------------------------------------"

    echo \
"${run_index},${CASE},${SOLVER},${N},${real_seconds},${user_seconds},${sys_seconds},${max_rss_kb},${output_dir},${GIT_BRANCH},${GIT_COMMIT}" \
    >> "${SUMMARY_FILE}"
}

# ------------------------------------------------------------
# Execute twice
# ------------------------------------------------------------

for ((RUN_INDEX = 1; RUN_INDEX <= NUM_RUNS; RUN_INDEX++)); do
    run_once "${RUN_INDEX}"
done

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "===== ALL RUNS COMPLETED ====="
echo "============================================================"

echo ""
echo "Summary:"
cat "${SUMMARY_FILE}"

echo ""
echo "Output root : ${RUN_ROOT}"
echo "Summary CSV : ${SUMMARY_FILE}"
echo "End time    : $(date --iso-8601=seconds)"

echo ""
echo "Final GPU state:"
nvidia-smi \
    --query-gpu=index,name,pstate,temperature.gpu,power.draw,memory.used \
    --format=csv \
    || true