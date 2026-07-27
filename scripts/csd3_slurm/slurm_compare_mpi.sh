#!/bin/bash -l
#SBATCH -J euler_cmp_mpi
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-CPU
#SBATCH -p icelake
#SBATCH -N 1
#SBATCH --ntasks=76
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --array=0-2%3
#SBATCH -t 36:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

# Final pure-MPI time-to-solution matrix on one 76-core CSD3 Ice Lake node.
# Each of the three array tasks owns one solver/node, builds once, and runs both
# cases and n={1,2,4} sequentially.  n=1/2 are repeated three times and n=4
# once.  n=8 is intentionally GPU-only.  --timing-only suppresses field
# gathers and field CSV output.
#
# If mybalance shows a different CPU project, override the account at submit:
#   mkdir -p logs
#   sbatch -A YOUR_PROJECT-CPU scripts/csd3_slurm/slurm_compare_mpi.sh

SUBMIT_ROOT="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
COMMON_SCRIPT="${SUBMIT_ROOT}/scripts/csd3_slurm/comparison_common.sh"
if [ ! -f "${COMMON_SCRIPT}" ]; then
    echo "[ERROR] Cannot find comparison helper: ${COMMON_SCRIPT}"
    echo "[ERROR] Submit this job from the Euler repository root, or set WORKDIR."
    exit 1
fi
# shellcheck source=comparison_common.sh
source "${COMMON_SCRIPT}"

BACKEND="mpi"
BUILD_VARIANT="repository_default_pure_mpi"
FAILED_RUNS=0
THREADS_REPORTED=1

# Pure-MPI n=8 is intentionally omitted because its time-to-solution dominates
# the allocation.  GPU retains n=8 in slurm_compare_gpu.sh.
SCALES_STR="1 2 4"
comparison_init_config
comparison_load_module rhel8/default-icl
comparison_prepare_paths

RANKS_REPORTED="${RANKS:-${SLURM_NTASKS:-76}}"
export OMP_NUM_THREADS=1
# rhel8/default-icl routes mpicxx through Intel classic icpc.  Its direct
# equivalent of GCC's -ffp-contract=off is -no-fma.
MPI_FP_FLAGS="${MPI_FP_FLAGS:--no-fma}"

for required in mpicxx mpirun; do
    if ! command -v "${required}" >/dev/null 2>&1; then
        echo "[ERROR] ${required} is unavailable after loading rhel8/default-icl."
        exit 1
    fi
done

COMPILER_INFO="$(mpicxx --version 2>&1)"
comparison_write_metadata "${COMPILER_INFO}"
{
    echo "mpirun=$(command -v mpirun)"
    echo "mpi_ranks=${RANKS_REPORTED}"
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    echo "MPI_FP_FLAGS=${MPI_FP_FLAGS}"
} >> "${METADATA_FILE}"

echo "===== BUILD PURE MPI ====="
BUILD_START_NS="$(date +%s%N)"
make -j "${MAKE_JOBS:-8}" mpi \
    BUILD_DIR="${BUILD_ROOT}" \
    BIN_DIR="${BIN_ROOT}" \
    MPI_FP_FLAGS="${MPI_FP_FLAGS}"
BUILD_END_NS="$(date +%s%N)"
awk -v start="${BUILD_START_NS}" -v end="${BUILD_END_NS}" 'BEGIN {printf "build_wall_seconds=%.9f\n", (end-start)/1.0e9}' >> "${METADATA_FILE}"

BIN="${BIN_ROOT}/main_mpi"
if [ ! -x "${BIN}" ]; then
    echo "[ERROR] MPI binary was not produced: ${BIN}"
    exit 1
fi

for N_SCALE_VALUE in "${SCALES[@]}"; do
    for CASE_VALUE in "${CASES[@]}"; do
        comparison_set_case_scale "${CASE_VALUE}" "${N_SCALE_VALUE}"
        for ((repeat = 1; repeat <= NUM_REPEATS; ++repeat)); do
            comparison_run_once "${repeat}" \
                mpirun -np "${RANKS_REPORTED}" "${BIN}" "${CASE_NAME}" --n "${N_SCALE}" --solver "${SOLVER_NAME}" --timing-only
        done
    done
done

comparison_finish
