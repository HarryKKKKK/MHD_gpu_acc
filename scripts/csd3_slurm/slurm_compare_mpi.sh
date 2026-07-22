#!/bin/bash -l
#SBATCH -J mhd_cmp_mpi
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-CPU
#SBATCH -p icelake
#SBATCH -N 1
#SBATCH --ntasks=76
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --array=0-31%4
#SBATCH -t 36:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

# Final pure-MPI time-to-solution matrix on one 76-core CSD3 Ice Lake node.
# Each of the 32 array tasks runs one case/solver/n configuration.  n=1/2 are
# repeated three times and n=4/8 once.  --timing-only suppresses field gathers
# and field CSV output.
#
# If mybalance shows a different CPU project, override the account at submit:
#   mkdir -p logs
#   sbatch -A YOUR_PROJECT-CPU scripts/csd3_slurm/slurm_compare_mpi.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=comparison_common.sh
source "${SCRIPT_DIR}/comparison_common.sh"

BACKEND="mpi"
BUILD_VARIANT="repository_default_pure_mpi"
FAILED_RUNS=0
THREADS_REPORTED=1

comparison_init_config
comparison_load_module rhel8/default-icl
comparison_prepare_paths

RANKS_REPORTED="${RANKS:-${SLURM_NTASKS:-76}}"
export OMP_NUM_THREADS=1

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
} >> "${METADATA_FILE}"

echo "===== BUILD PURE MPI ====="
BUILD_START_NS="$(date +%s%N)"
make -j "${MAKE_JOBS:-8}" mpi BUILD_DIR="${BUILD_ROOT}" BIN_DIR="${BIN_ROOT}"
BUILD_END_NS="$(date +%s%N)"
awk -v start="${BUILD_START_NS}" -v end="${BUILD_END_NS}" 'BEGIN {printf "build_wall_seconds=%.9f\n", (end-start)/1.0e9}' >> "${METADATA_FILE}"

BIN="${BIN_ROOT}/main_mpi"
if [ ! -x "${BIN}" ]; then
    echo "[ERROR] MPI binary was not produced: ${BIN}"
    exit 1
fi

for ((repeat = 1; repeat <= NUM_REPEATS; ++repeat)); do
    comparison_run_once "${repeat}" \
        mpirun -np "${RANKS_REPORTED}" "${BIN}" "${CASE_NAME}" --n "${N_SCALE}" --solver "${SOLVER_NAME}" --timing-only
done

comparison_finish
