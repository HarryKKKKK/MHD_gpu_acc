#!/bin/bash -l
#SBATCH -J mhd_valid
#SBATCH -A hk597
#SBATCH -p csc-mphil-gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Validation runs matching published benchmarks:
#
#  Test case      Solver  Reference
#  -----------    ------  -----------------------------------------
#  shock_bubble   hllc    Haas & Sturtevant (1987) Fig. 4, Ms=1.22
#                         9 snapshots at dimensionless t = 0.6..19.0
#  brio_wu        hlld    Brio & Wu (1988) Fig. 4; Mignone et al. (2010) §4.2
#                         γ=2, left(ρ=1,p=1,Bx=0.75,By=1), right(ρ=0.125,p=0.1,By=-1)
#  orszag_tang    hlld    Mignone et al. (2010) §4.5, Dahlburg–Picone ICs
#                         Snapshots at t=π (≡ Mignone t=0.5) and t=2π (≡ t=1)
#                         on [0,2π]² domain (same physics as their [0,1]²)

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
cd "$WORKDIR"

mkdir -p logs outputs

echo "===== JOB INFO ====="
echo "JobID  : ${SLURM_JOB_ID}"
echo "Host   : $(hostname)"
echo "Start  : $(date)"
echo ""

echo "===== BUILD ====="
make clean
make cpu gpu

# ── Validation matrix ───────────────────────────────────────────────────────
#   Parallel arrays: CASES[i] runs with SOLVERS[i]
declare -a CASES=("shock_bubble" "brio_wu"  "orszag_tang")
declare -a SOLVERS=("hllc"       "hlld"     "hlld")

# ── CPU runs ─────────────────────────────────────────────────────────────────
echo ""
echo "===== CPU RUNS ====="
for i in "${!CASES[@]}"; do
    case_name="${CASES[$i]}"
    solver="${SOLVERS[$i]}"
    out_dir="outputs/cpu_${case_name}_${solver}"

    echo ""
    echo "===== CPU | ${case_name} | solver=${solver} ====="
    echo "Output: ${out_dir}"

    OMP_NUM_THREADS=8 OMP_PROC_BIND=true OMP_PLACES=cores \
        ./bin/main_cpu "${case_name}" --solver "${solver}" --out "${out_dir}"
done

# ── GPU runs ─────────────────────────────────────────────────────────────────
echo ""
echo "===== GPU RUNS ====="
for i in "${!CASES[@]}"; do
    case_name="${CASES[$i]}"
    solver="${SOLVERS[$i]}"
    out_dir="outputs/gpu_${case_name}_${solver}_n1"

    echo ""
    echo "===== GPU | ${case_name} | solver=${solver} ====="
    echo "Output: ${out_dir}"

    ./bin/main_gpu 1 --case "${case_name}" --solver "${solver}" --out "${out_dir}"
done

echo ""
echo "===== VALIDATION COMPLETE ====="
echo "Output directories:"
for i in "${!CASES[@]}"; do
    echo "  outputs/cpu_${CASES[$i]}_${SOLVERS[$i]}/"
    echo "  outputs/gpu_${CASES[$i]}_${SOLVERS[$i]}_n1/"
done
echo ""
echo "Run visualisation:"
echo "  python visualization/plot_shock_bubble.py"
echo "  python visualization/plot_brio_wu.py"
echo "  python visualization/plot_orszag_tang.py"
date
