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

if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch $0 CPU_ARRAY_JOB_ID GPU_ARRAY_JOB_ID"
    exit 2
fi

CPU_JOB_ID="$1"
GPU_JOB_ID="$2"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${WORKDIR}"

RESULT_DIR="${WORKDIR}/timing/euler_comparison/${CPU_JOB_ID}_${GPU_JOB_ID}"
mkdir -p "${RESULT_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 is unavailable on the summary node."
    exit 1
fi

python3 scripts/summarize_gpu_speedup.py \
    --root timing/final_comparison \
    --cpu-job "${CPU_JOB_ID}" \
    --gpu-job "${GPU_JOB_ID}" \
    --output "${RESULT_DIR}/gpu_speedup_summary.csv"

{
    echo "cpu_array_job_id=${CPU_JOB_ID}"
    echo "gpu_array_job_id=${GPU_JOB_ID}"
    echo "summary_job_id=${SLURM_JOB_ID:-manual}"
    echo "created_utc=$(date --utc --iso-8601=seconds)"
    echo "speedup_definition=median_cpu_app_elapsed_s/median_gpu_app_elapsed_s"
    echo "cases=shock_bubble blast_wave"
    echo "solvers=hll hllc hlld force"
    echo "paired_speedup_scales=1 2 4"
    echo "gpu_timing_scales=1 2 4 8"
} > "${RESULT_DIR}/metadata.txt"

cat "${RESULT_DIR}/gpu_speedup_summary.csv"
echo "Summary: ${RESULT_DIR}/gpu_speedup_summary.csv"
