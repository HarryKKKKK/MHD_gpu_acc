#!/bin/bash -l
#SBATCH -J kh_cpu_omp
#SBATCH -A hk597
#SBATCH -p csc-mphil
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --time=05:59:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
cd "$WORKDIR"

mkdir -p logs validation outputs

# Override on the command line like:
# CASES_STR="kelvin_helmholtz" SOLVERS_STR="hll" ORDERS_STR="2" sbatch scripts/slurm_cpu.sh
read -r -a CASES   <<< "${CASES_STR:-kelvin_helmholtz}"
read -r -a SOLVERS <<< "${SOLVERS_STR:-hll hlld force}"
read -r -a ORDERS  <<< "${ORDERS_STR:-1 2}"

detect_cpus_on_node() {
    if [ -n "${SLURM_CPUS_ON_NODE:-}" ]; then
        echo "${SLURM_CPUS_ON_NODE}"
        return
    fi
    if [ -n "${SLURM_JOB_CPUS_PER_NODE:-}" ]; then
        echo "${SLURM_JOB_CPUS_PER_NODE}" | sed -E 's/\(x[0-9]+\)//'
        return
    fi
    nproc
}

CPUS_ON_NODE="$(detect_cpus_on_node)"
OMP_THREADS="${OMP_THREADS:-${CPUS_ON_NODE}}"

if [ "$OMP_THREADS" -gt "$CPUS_ON_NODE" ]; then
    echo "[ERROR] OMP_THREADS=${OMP_THREADS} > CPUS_ON_NODE=${CPUS_ON_NODE}"
    exit 1
fi

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS="${OMP_THREADS}"

echo "===== JOB INFO ====="
echo "JobID              : ${SLURM_JOB_ID}"
echo "Host               : $(hostname)"
echo "Start              : $(date)"
echo "Workdir            : ${WORKDIR}"
echo "Partition          : ${SLURM_JOB_PARTITION:-unknown}"
echo "SLURM_CPUS_ON_NODE : ${SLURM_CPUS_ON_NODE:-unset}"
echo "Detected CPUs/node : ${CPUS_ON_NODE}"
echo "OMP_THREADS        : ${OMP_THREADS}"
echo "CASES              : ${CASES[*]}"
echo "SOLVERS            : ${SOLVERS[*]}"
echo "ORDERS             : ${ORDERS[*]}"
echo ""

echo "===== CPU TOPOLOGY ====="
lscpu | grep -E 'CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)' || true

echo ""
echo "===== ENV CHECK ====="
echo "which g++: $(which g++ || echo 'NOT FOUND')"
if ! command -v g++ >/dev/null 2>&1; then
    echo "[ERROR] g++ not found in PATH."
    exit 1
fi

echo ""
echo "===== BUILD ====="
make clean
make cpu

SUMMARY="validation/cpu_omp_${SLURM_JOB_ID}.csv"
echo "arch,case,solver,order,threads,nx,ny,total_cells,total_steps,elapsed_s,real_seconds,user_seconds,sys_seconds,max_rss_kb,run_log,time_log" > "$SUMMARY"

# Parse fields from main_cpu stdout.
# Lines of interest:
#   "  Grid      : NX x NY"
#   "  Done: STEPS steps in ELAPSED s  (...)"
extract_cpu_fields() {
    local run_log="$1"
    local nx ny cells steps elapsed
    nx=$(awk '/Grid[[:space:]]*:/{print $3}' "$run_log")
    ny=$(awk '/Grid[[:space:]]*:/{print $5}' "$run_log")
    cells=$(( ${nx:-0} * ${ny:-0} ))
    steps=$(awk '/Done:/{print $2}' "$run_log")
    elapsed=$(awk '/Done:/{print $5}' "$run_log")
    echo "${nx},${ny},${cells},${steps},${elapsed}"
}

append_row() {
    local solver_name="$1" order="$2" case_name="$3" run_log="$4" time_log="$5"
    local fields real user sys rss
    fields=$(extract_cpu_fields "$run_log")
    real=$(awk -F '=' '/real_seconds/{print $2}' "$time_log")
    user=$(awk -F '=' '/user_seconds/{print $2}' "$time_log")
    sys=$(awk -F '=' '/sys_seconds/{print $2}' "$time_log")
    rss=$(awk -F '=' '/max_rss_kb/{print $2}' "$time_log")
    echo "cpu_omp,${case_name},${solver_name},${order},${OMP_THREADS},${fields},${real},${user},${sys},${rss},${run_log},${time_log}" >> "$SUMMARY"
}

for CASE in "${CASES[@]}"; do
    for SOLVER in "${SOLVERS[@]}"; do
        for ORDER in "${ORDERS[@]}"; do
            echo ""
            echo "===== CPU OMP RUN: case=${CASE}, solver=${SOLVER}, order=${ORDER}, threads=${OMP_THREADS} ====="

            OUT_DIR="outputs/${CASE}_${SOLVER}_o${ORDER}_t${OMP_THREADS}_${SLURM_JOB_ID}"
            RUN_LOG="logs/cpu_${CASE}_${SOLVER}_o${ORDER}_t${OMP_THREADS}_${SLURM_JOB_ID}.log"
            TIME_LOG="logs/cpu_${CASE}_${SOLVER}_o${ORDER}_t${OMP_THREADS}_${SLURM_JOB_ID}.time"

            /usr/bin/time -f "real_seconds=%e\nuser_seconds=%U\nsys_seconds=%S\nmax_rss_kb=%M" \
                -o "$TIME_LOG" \
                ./main_cpu "$CASE" --solver "$SOLVER" --order "$ORDER" --out "$OUT_DIR" \
                2>&1 | tee "$RUN_LOG"

            echo "----- TIME -----"
            cat "$TIME_LOG"

            append_row "$SOLVER" "$ORDER" "$CASE" "$RUN_LOG" "$TIME_LOG"
        done
    done
done

echo ""
echo "===== SUMMARY CSV ====="
cat "$SUMMARY"
echo ""
echo "Saved summary: $SUMMARY"
echo "===== END ====="
date
