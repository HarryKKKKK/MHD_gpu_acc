#!/bin/bash

# Submit the Euler timing workflow using the same CPU/GPU timing jobs and
# result format as timing/final_comparison.  The summary job starts only after
# both four-task solver arrays have completed successfully.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
cd "${WORKDIR}"

if [ ! -f scripts/csd3_slurm/slurm_compare_cpu.sh ] ||
   [ ! -f scripts/csd3_slurm/slurm_compare_gpu.sh ]; then
    echo "[ERROR] Run this launcher from the MHD repository root, or set WORKDIR."
    exit 1
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "[ERROR] sbatch is unavailable."
    exit 1
fi

mkdir -p logs timing/euler_comparison

# comparison_common.sh requires exactly two cases, four solvers, and four
# scales.  These values select the two pure-hydrodynamic (Euler) cases while
# retaining the established solver-array mapping and repeat policy.
export CASES_STR="shock_bubble blast_wave"
export SOLVERS_STR="${SOLVERS_STR:-hll hllc hlld force}"
export SCALES_STR="${SCALES_STR:-1 2 4 8}"

CPU_SUBMISSION="$(sbatch --parsable --job-name=euler_cmp_cpu \
    scripts/csd3_slurm/slurm_compare_cpu.sh)"
CPU_JOB_ID="${CPU_SUBMISSION%%;*}"

GPU_SUBMISSION="$(sbatch --parsable --job-name=euler_cmp_gpu \
    scripts/csd3_slurm/slurm_compare_gpu.sh)"
GPU_JOB_ID="${GPU_SUBMISSION%%;*}"

SUMMARY_SUBMISSION="$(sbatch --parsable \
    --dependency="afterok:${CPU_JOB_ID}:${GPU_JOB_ID}" \
    scripts/csd3_slurm/slurm_summarize_euler.sh \
    "${CPU_JOB_ID}" "${GPU_JOB_ID}")"
SUMMARY_JOB_ID="${SUMMARY_SUBMISSION%%;*}"

echo "Submitted Euler timing workflow:"
echo "  CPU array : ${CPU_JOB_ID}"
echo "  GPU array : ${GPU_JOB_ID}"
echo "  Summary   : ${SUMMARY_JOB_ID}"
echo "After completion:"
echo "  timing/euler_comparison/${CPU_JOB_ID}_${GPU_JOB_ID}/gpu_speedup_summary.csv"
