#!/bin/bash -l
#SBATCH -J mhd_diag
#SBATCH -A hansirui
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# ch_glm-ratchet diagnostic sweep: CPU(OMP) + GPU, n=1 only, hll (baseline)
# + hllc for orszag_tang and rotor. Not a performance benchmark — grid is
# tiny on purpose (n=1) so the diag CSVs (one row every DIAG_INTERVAL
# steps) stay small and step-by-step behaviour is easy to plot.
#
# --cpus-per-task=16: diagnostics only care about correctness, not
# throughput, so this does not need the full-node 112-core allocation the
# slurm_cpu_omp.sh benchmark uses.
#
# Produces, under diagnostics/<jobid>/:
#   diag_cpu_orszag_tang_hll_n1.csv   diag_gpu_orszag_tang_hll_n1.csv
#   diag_cpu_orszag_tang_hllc_n1.csv  diag_gpu_orszag_tang_hllc_n1.csv
#   diag_cpu_rotor_hll_n1.csv         diag_gpu_rotor_hll_n1.csv
#   diag_cpu_rotor_hllc_n1.csv        diag_gpu_rotor_hllc_n1.csv
# each with per-step columns:
#   arch,case,solver,step,t,dt,ch_glm,max_cf_x,max_cf_y,max_speed,
#   max_rho,min_rho,max_p,min_p,max_abs_Bx,max_abs_By,max_abs_Bz,max_abs_psi,
#   divB_L1,divB_max,n_floor_triggered
# plus a trailing "# exit_summary: final_step=...,final_t=...,
# t_end_target=...,exit_reason=..." comment line (skip it with
# pandas.read_csv(path, comment='#')).
#
# Override on the command line before sbatch, e.g.:
#   DIAG_INTERVAL=10 CASES_STR="rotor" sbatch scripts/dgx_slurm/slurm_diag.sh
# ============================================================

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
WORKDIR="${WORKDIR:-/aifs4su/hansirui_2nd/harry/MHD_gpu_acc}"
cd "$WORKDIR"

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT=$(git rev-parse --short HEAD   2>/dev/null || echo "unknown")

mkdir -p logs

read -r -a CASES   <<< "${CASES_STR:-orszag_tang rotor}"
read -r -a SOLVERS <<< "${SOLVERS_STR:-hll hllc}"
DIAG_INTERVAL="${DIAG_INTERVAL:-5}"
DIAG_DIR="diagnostics/${SLURM_JOB_ID}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "===== JOB INFO ====="
echo "JobID         : ${SLURM_JOB_ID}"
echo "Host          : $(hostname)"
echo "Start         : $(date)"
echo "Workdir       : ${WORKDIR}"
echo "Git Branch    : ${GIT_BRANCH}"
echo "Git Commit    : ${GIT_COMMIT}"
echo "Cases         : ${CASES[*]}"
echo "Solvers       : ${SOLVERS[*]}"
echo "Diag interval : ${DIAG_INTERVAL}"
echo "Diag dir      : ${DIAG_DIR}"
echo "OMP_THREADS   : ${OMP_NUM_THREADS}"
echo ""

module load cuda/12.2

echo "===== ENV CHECK ====="
echo "which g++ : $(which g++  || echo 'NOT FOUND')"
echo "which nvcc: $(which nvcc || echo 'NOT FOUND')"
if ! command -v g++ >/dev/null 2>&1; then
    echo "[ERROR] g++ not found in PATH."; exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[ERROR] nvcc not found even after 'module load cuda/12.2'."; exit 1
fi
echo ""

echo "===== BUILD ====="
make cpu gpu

mkdir -p "$DIAG_DIR"

run_cpu() {
    local case_name="$1" solver="$2"
    echo ""
    echo "===== CPU DIAG: case=${case_name}, solver=${solver}, n=1 ====="
    ./bin/main_cpu "$case_name" --n 1 --solver "$solver" --no-out \
        --diag-interval "$DIAG_INTERVAL" --diag-dir "$DIAG_DIR"
}

run_gpu() {
    local case_name="$1" solver="$2"
    echo ""
    echo "===== GPU DIAG: case=${case_name}, solver=${solver}, n=1 ====="
    ./bin/main_gpu 1 --case "$case_name" --solver "$solver" --no-out \
        --diag-interval "$DIAG_INTERVAL" --diag-dir "$DIAG_DIR"
}

for CASE in "${CASES[@]}"; do
    for SOLVER in "${SOLVERS[@]}"; do
        run_cpu "$CASE" "$SOLVER"
        run_gpu "$CASE" "$SOLVER"
    done
done

echo ""
echo "===== DIAG FILES ====="
ls -la "$DIAG_DIR"
echo ""
echo "===== EXIT SUMMARIES ====="
grep -H '^# exit_summary' "$DIAG_DIR"/*.csv || true
echo ""
echo "Saved diagnostics under: $DIAG_DIR"
echo "===== END ====="
date
