#!/bin/bash -l
#SBATCH -J mhd_full_compare
#SBATCH -A hansirui
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# Full CPU-vs-GPU field-output sweep: every solver x every base test case
# from `note` (Euler/shock_bubble, 1D/brio_wu, 2D/orszag_tang — the 4th
# entry "kl"/kelvin_helmholtz is deprecated, not run here), n=1. Unlike
# slurm_diag.sh (which only samples scalar diagnostics with --no-out),
# this writes the *full field* CSVs for both CPU and GPU and then diffs
# them cell-by-cell with compare_cpu_gpu.py.
#
# Produces, under outputs/<jobid>/:
#   cpu_<case>_<solver>/       full-field CSVs from main_cpu
#   gpu_<case>_<solver>_n1/    full-field CSVs from main_gpu
#   comparison_report.csv      one row per (case, solver): worst field,
#                               max|Δ|, max relative Δ, L2(Δ), PASS/FAIL
#
# Override on the command line before sbatch, e.g.:
#   CASES_STR="brio_wu orszag_tang" SOLVERS_STR="hllc" sbatch scripts/dgx_slurm/slurm_full_compare.sh
#
# Some (case, solver) combos are known to be numerically unstable (see
# debug_cpu branch investigation into HLLC on strong-field cases) and can
# take far more steps than others to reach t_end; each run still has its
# own dt-floor safety break so nothing hangs indefinitely, but the whole
# sweep can take a while. Increase --time above if you widen CASES/SOLVERS.
# ============================================================

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
WORKDIR="${WORKDIR:-/aifs4su/hansirui_2nd/harry/MHD_gpu_acc}"
cd "$WORKDIR"

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT=$(git rev-parse --short HEAD   2>/dev/null || echo "unknown")

mkdir -p logs

read -r -a CASES   <<< "${CASES_STR:-shock_bubble brio_wu orszag_tang}"
read -r -a SOLVERS <<< "${SOLVERS_STR:-hll hllc hlld force}"
OUT_ROOT="outputs/${SLURM_JOB_ID}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "===== JOB INFO ====="
echo "JobID       : ${SLURM_JOB_ID}"
echo "Host        : $(hostname)"
echo "Start       : $(date)"
echo "Workdir     : ${WORKDIR}"
echo "Git Branch  : ${GIT_BRANCH}"
echo "Git Commit  : ${GIT_COMMIT}"
echo "Cases       : ${CASES[*]}"
echo "Solvers     : ${SOLVERS[*]}"
echo "Output root : ${OUT_ROOT}"
echo "OMP_THREADS : ${OMP_NUM_THREADS}"
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
if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 not found in PATH."; exit 1
fi
echo ""

echo "===== BUILD ====="
make cpu gpu

mkdir -p "$OUT_ROOT"

run_cpu() {
    local case_name="$1" solver="$2"
    local out_dir="${OUT_ROOT}/cpu_${case_name}_${solver}"
    echo ""
    echo "===== CPU: case=${case_name}, solver=${solver}, n=1 ====="
    echo "Output: ${out_dir}"
    ./bin/main_cpu "$case_name" --n 1 --solver "$solver" --out "$out_dir"
}

run_gpu() {
    local case_name="$1" solver="$2"
    local out_dir="${OUT_ROOT}/gpu_${case_name}_${solver}_n1"
    echo ""
    echo "===== GPU: case=${case_name}, solver=${solver}, n=1 ====="
    echo "Output: ${out_dir}"
    ./bin/main_gpu 1 --case "$case_name" --solver "$solver" --out "$out_dir"
}

for CASE in "${CASES[@]}"; do
    for SOLVER in "${SOLVERS[@]}"; do
        run_cpu "$CASE" "$SOLVER"
        run_gpu "$CASE" "$SOLVER"
    done
done

echo ""
echo "===== COMPARISON ====="
python3 scripts/compare_cpu_gpu.py "$OUT_ROOT" \
    --cases   "$(IFS=,; echo "${CASES[*]}")" \
    --solvers "$(IFS=,; echo "${SOLVERS[*]}")" \
    || echo "[NOTE] compare_cpu_gpu.py exited non-zero — see FAIL/NO DATA rows above."

echo ""
echo "Saved outputs under : $OUT_ROOT"
echo "Comparison report   : $OUT_ROOT/comparison_report.csv"
echo "===== END ====="
date
