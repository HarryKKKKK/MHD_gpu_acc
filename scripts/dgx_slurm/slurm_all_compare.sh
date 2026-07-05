#!/bin/bash -l
#SBATCH -J mhd_all_compare
#SBATCH -A hansirui
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# Full CPU-vs-GPU-vs-MPI field-output sweep: every solver x every non-
# deprecated base test case (shock_bubble, brio_wu, orszag_tang, rotor —
# kelvin_helmholtz is deprecated, see slurm_full_compare.sh), n=1.
#
# Replaces slurm_diag.sh (deleted): that script only sampled scalar
# diagnostics for a 2-case/2-solver subset. This one writes the *full
# field* CSVs for CPU, GPU, and MPI on every case x solver combo and then
# diffs every pair of archs cell-by-cell with compare_multi_arch.py.
#
# Produces, under outputs/<jobid>/:
#   cpu_<case>_<solver>/       full-field CSVs from main_cpu
#   gpu_<case>_<solver>_n1/    full-field CSVs from main_gpu
#   mpi_<case>_<solver>_n1/    full-field CSVs from main_mpi (root rank only)
#   comparison_report.csv      one row per (case, solver, arch-pair): worst
#                               field, max|Δ|, max relative Δ, L2(Δ), PASS/FAIL
#
# Override on the command line before sbatch, e.g.:
#   CASES_STR="brio_wu rotor" SOLVERS_STR="hllc hlld" RANKS=4 \
#       sbatch scripts/dgx_slurm/slurm_all_compare.sh
#
# Some (case, solver) combos are known to be numerically unstable (see
# debug_cpu branch investigation into HLLC on strong-field cases) and can
# take far more steps than others to reach t_end; each run still has its
# own dt-floor safety break so nothing hangs indefinitely, but the whole
# sweep (4 cases x 4 solvers x 3 archs = 48 runs) can take a while.
# Increase --time above if you widen CASES/SOLVERS.
# ============================================================

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
WORKDIR="${WORKDIR:-/aifs4su/hansirui_2nd/harry/MHD_gpu_acc}"
cd "$WORKDIR"

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT=$(git rev-parse --short HEAD   2>/dev/null || echo "unknown")

mkdir -p logs

read -r -a CASES   <<< "${CASES_STR:-shock_bubble brio_wu orszag_tang rotor}"
read -r -a SOLVERS <<< "${SOLVERS_STR:-hll hllc hlld force}"
RANKS="${RANKS:-4}"
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
echo "MPI ranks   : ${RANKS}"
echo "Output root : ${OUT_ROOT}"
echo "OMP_THREADS : ${OMP_NUM_THREADS}"
echo ""

module load cuda/12.2

echo "===== ENV CHECK ====="
echo "which g++   : $(which g++    || echo 'NOT FOUND')"
echo "which nvcc  : $(which nvcc   || echo 'NOT FOUND')"
echo "which mpicxx: $(which mpicxx || echo 'NOT FOUND')"
echo "which mpirun: $(which mpirun || echo 'NOT FOUND')"
if ! command -v g++ >/dev/null 2>&1; then
    echo "[ERROR] g++ not found in PATH."; exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[ERROR] nvcc not found even after 'module load cuda/12.2'."; exit 1
fi
if ! command -v mpicxx >/dev/null 2>&1; then
    echo "[ERROR] mpicxx not found in PATH."; exit 1
fi
if ! command -v mpirun >/dev/null 2>&1; then
    echo "[ERROR] mpirun not found in PATH."; exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 not found in PATH."; exit 1
fi
echo ""

echo "===== BUILD ====="
make clean
make cpu gpu mpi

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

run_mpi() {
    local case_name="$1" solver="$2"
    local out_dir="${OUT_ROOT}/mpi_${case_name}_${solver}_n1"
    echo ""
    echo "===== MPI: case=${case_name}, solver=${solver}, n=1, ranks=${RANKS} ====="
    echo "Output: ${out_dir}"
    # --oversubscribe: bypass the slot-count check without disabling
    # slurm's cgroup CPU topology detection (see slurm_mpi.sh).
    mpirun -np "$RANKS" --host "$(hostname):${RANKS}" \
        --oversubscribe --bind-to core --map-by core \
        ./bin/main_mpi "$case_name" --n 1 --solver "$solver" --out "$out_dir"
}

for CASE in "${CASES[@]}"; do
    for SOLVER in "${SOLVERS[@]}"; do
        run_cpu "$CASE" "$SOLVER"
        run_gpu "$CASE" "$SOLVER"
        run_mpi "$CASE" "$SOLVER"
    done
done

echo ""
echo "===== COMPARISON (cpu vs gpu vs mpi, pairwise) ====="
python3 scripts/compare_multi_arch.py "$OUT_ROOT" \
    --cases   "$(IFS=,; echo "${CASES[*]}")" \
    --solvers "$(IFS=,; echo "${SOLVERS[*]}")" \
    --archs   cpu,gpu,mpi \
    || echo "[NOTE] compare_multi_arch.py exited non-zero — see FAIL/NO DATA rows above."

echo ""
echo "Saved outputs under : $OUT_ROOT"
echo "Comparison report   : $OUT_ROOT/comparison_report.csv"
echo "===== END ====="
date
