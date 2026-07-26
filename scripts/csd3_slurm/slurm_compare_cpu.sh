#!/bin/bash -l
#SBATCH -J mhd_cmp_cpu
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-CPU
#SBATCH -p icelake
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --exclusive
#SBATCH --array=0-3%4
#SBATCH -t 36:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

# Final OpenMP CPU time-to-solution matrix on one 76-core CSD3 Ice Lake node.
# Each of the four array tasks owns one solver/node, builds once, and runs both
# cases and n={1,2,4} sequentially.  n=1/2 are repeated three times and n=4
# once.  n=8 is intentionally GPU-only.  --no-out suppresses field CSVs.
#
# If mybalance shows a different CPU project, override the account at submit:
#   mkdir -p logs
#   sbatch -A YOUR_PROJECT-CPU scripts/csd3_slurm/slurm_compare_cpu.sh

SUBMIT_ROOT="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
COMMON_SCRIPT="${SUBMIT_ROOT}/scripts/csd3_slurm/comparison_common.sh"
if [ ! -f "${COMMON_SCRIPT}" ]; then
    echo "[ERROR] Cannot find comparison helper: ${COMMON_SCRIPT}"
    echo "[ERROR] Submit this job from the MHD repository root, or set WORKDIR."
    exit 1
fi
# shellcheck source=comparison_common.sh
source "${COMMON_SCRIPT}"

BACKEND="cpu_openmp"
BUILD_VARIANT="repository_default_openmp"
FAILED_RUNS=0
RANKS_REPORTED=0

# OpenMP n=8 is intentionally omitted because its time-to-solution dominates
# the allocation.  GPU retains n=8 in slurm_compare_gpu.sh.
SCALES_STR="1 2 4"
comparison_init_config
comparison_load_module rhel8/default-icl
comparison_prepare_paths

THREADS_REPORTED="${OMP_THREADS:-${SLURM_CPUS_PER_TASK:-76}}"
export OMP_NUM_THREADS="${THREADS_REPORTED}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

if ! command -v g++ >/dev/null 2>&1; then
    echo "[ERROR] g++ is unavailable after loading rhel8/default-icl."
    exit 1
fi

COMPILER_INFO="$(g++ --version 2>&1)"
comparison_write_metadata "${COMPILER_INFO}"
{
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    echo "OMP_PROC_BIND=${OMP_PROC_BIND}"
    echo "OMP_PLACES=${OMP_PLACES}"
} >> "${METADATA_FILE}"

echo "===== BUILD CPU OPENMP ====="
BUILD_START_NS="$(date +%s%N)"
make -j "${MAKE_JOBS:-8}" cpu BUILD_DIR="${BUILD_ROOT}" BIN_DIR="${BIN_ROOT}"
BUILD_END_NS="$(date +%s%N)"
awk -v start="${BUILD_START_NS}" -v end="${BUILD_END_NS}" 'BEGIN {printf "build_wall_seconds=%.9f\n", (end-start)/1.0e9}' >> "${METADATA_FILE}"

BIN="${BIN_ROOT}/main_cpu"
if [ ! -x "${BIN}" ]; then
    echo "[ERROR] CPU binary was not produced: ${BIN}"
    exit 1
fi

for N_SCALE_VALUE in "${SCALES[@]}"; do
    for CASE_VALUE in "${CASES[@]}"; do
        comparison_set_case_scale "${CASE_VALUE}" "${N_SCALE_VALUE}"
        for ((repeat = 1; repeat <= NUM_REPEATS; ++repeat)); do
            comparison_run_once "${repeat}" \
                "${BIN}" "${CASE_NAME}" --n "${N_SCALE}" --solver "${SOLVER_NAME}" --no-out
        done
    done
done

comparison_finish
