#!/bin/bash -l
#SBATCH -J euler_submit
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-CPU
#SBATCH -p icelake
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:05:00
#SBATCH -o euler_submit_%j.out
#SBATCH -e euler_submit_%j.err

# CSD3 orchestration job for the Euler timing workflow.  This short job submits
# the established Ice Lake CPU and Ampere GPU timing arrays, then submits the
# summary job with an afterok dependency on both arrays.  Timing data retain
# the same format and hardware separation as timing/final_comparison.

set -euo pipefail

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
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

# These values select the two pure-hydrodynamic (Euler) cases while retaining
# the established solver-array mapping and repeat policy.  The CPU script
# fixes its scales to n={1,2,4}; the GPU script retains n={1,2,4,8}.
export CASES_STR="shock_bubble blast_wave"
export SOLVERS_STR="${SOLVERS_STR:-hll hllc hlld force}"

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
