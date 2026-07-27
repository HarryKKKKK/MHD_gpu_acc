#!/bin/bash -l
#SBATCH -J euler_speedup
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-CPU
#SBATCH -p icelake
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:15:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: sbatch $0 OMP_ARRAY_JOB_ID MPI_ARRAY_JOB_ID GPU_ARRAY_JOB_ID"
    exit 2
fi

OMP_JOB_ID="$1"
MPI_JOB_ID="$2"
GPU_JOB_ID="$3"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${WORKDIR}"

RESULT_DIR="${WORKDIR}/timing/euler_comparison/${OMP_JOB_ID}_${MPI_JOB_ID}_${GPU_JOB_ID}"
mkdir -p "${RESULT_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 is unavailable on the summary node."
    exit 1
fi

python3 scripts/summarize_euler_runtime.py \
    --root timing/final_comparison \
    --omp-job "${OMP_JOB_ID}" \
    --mpi-job "${MPI_JOB_ID}" \
    --gpu-job "${GPU_JOB_ID}" \
    --output-dir "${RESULT_DIR}"

{
    echo "omp_array_job_id=${OMP_JOB_ID}"
    echo "mpi_array_job_id=${MPI_JOB_ID}"
    echo "gpu_array_job_id=${GPU_JOB_ID}"
    echo "summary_job_id=${SLURM_JOB_ID:-manual}"
    echo "created_utc=$(date --utc --iso-8601=seconds)"
    echo "speedup_definition=median_reference_app_elapsed_s/median_target_app_elapsed_s"
    echo "cases=shock_bubble blast_wave"
    echo "solvers=hll hllc force"
    echo "omp_scales=1 2 4"
    echo "mpi_scales=1 2 4"
    echo "gpu_scales=1 2 4 8"
} > "${RESULT_DIR}/metadata.txt"

cat "${RESULT_DIR}/runtime_summary.csv"
echo "Runtime summary: ${RESULT_DIR}/runtime_summary.csv"
echo "Speedup summary: ${RESULT_DIR}/speedup_summary.csv"
