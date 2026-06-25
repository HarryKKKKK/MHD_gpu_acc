#!/bin/bash -l
#SBATCH -J cpu_timing
#SBATCH -A hk597
#SBATCH -p csc-mphil
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# CPU timing script — Brio-Wu & Orszag-Tang, HLLD, n=2 (weak scaling)
# ============================================================

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
cd "$WORKDIR"

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT=$(git rev-parse --short HEAD   2>/dev/null || echo "unknown")

mkdir -p logs validation "outputs/${SLURM_JOB_ID}"

# Fixed parameters for this timing run
CASES=(brio_wu orszag_tang)
SOLVER="hlld"
N_SCALE=2

# Detect OMP thread count from Slurm or fall back to nproc
detect_cpus_on_node() {
    if [ -n "${SLURM_CPUS_ON_NODE:-}" ]; then echo "${SLURM_CPUS_ON_NODE}"; return; fi
    if [ -n "${SLURM_JOB_CPUS_PER_NODE:-}" ]; then
        echo "${SLURM_JOB_CPUS_PER_NODE}" | sed -E 's/\(x[0-9]+\)//'; return
    fi
    nproc
}

CPUS_ON_NODE="$(detect_cpus_on_node)"
OMP_THREADS="${OMP_THREADS:-${CPUS_ON_NODE}}"

if [ "$OMP_THREADS" -gt "$CPUS_ON_NODE" ]; then
    echo "[ERROR] OMP_THREADS=${OMP_THREADS} > CPUS_ON_NODE=${CPUS_ON_NODE}"; exit 1
fi

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS="${OMP_THREADS}"

echo "===== JOB INFO ====="
echo "JobID              : ${SLURM_JOB_ID}"
echo "Host               : $(hostname)"
echo "Start              : $(date)"
echo "Workdir            : ${WORKDIR}"
echo "Git Branch         : ${GIT_BRANCH}"
echo "Git Commit         : ${GIT_COMMIT}"
echo "Partition          : ${SLURM_JOB_PARTITION:-unknown}"
echo "Detected CPUs/node : ${CPUS_ON_NODE}"
echo "OMP_NUM_THREADS    : ${OMP_THREADS}"
echo "Cases              : ${CASES[*]}"
echo "Solver             : ${SOLVER}"
echo "Weak-scale n       : ${N_SCALE}"
echo ""

echo "===== CPU TOPOLOGY ====="
lscpu | grep -E 'CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)' || true
echo ""

echo "===== BUILD ====="
make clean
make cpu
echo ""

# ------------------------------------------------------------------
# Summary CSV
# ------------------------------------------------------------------
SUMMARY="validation/cpu_timing_${SLURM_JOB_ID}.csv"
echo "case,solver,n,threads,nx,ny,total_cells,steps,elapsed_s,steps_per_s,Mcell_updates_s,real_s,user_s,sys_s,max_rss_kb,git_branch,git_commit" \
    > "$SUMMARY"

parse_timing() { grep '^\[TIMING\]' "$1" | grep -oP "(?<=${2}=)[^ ]+" | tail -1; }

# ------------------------------------------------------------------
# Run one case and append a row to the summary
# ------------------------------------------------------------------
run_case() {
    local case_name="$1"

    local out_dir="outputs/${SLURM_JOB_ID}/${case_name}_${SOLVER}_n${N_SCALE}"
    mkdir -p "$out_dir"

    echo "===== RUN: ${case_name} | solver=${SOLVER} | n=${N_SCALE} | threads=${OMP_THREADS} ====="

    local temp_log temp_time
    temp_log=$(mktemp)
    temp_time=$(mktemp)

    /usr/bin/time -f "real_seconds=%e\nuser_seconds=%U\nsys_seconds=%S\nmax_rss_kb=%M" \
        -o "$temp_time" \
        ./bin/main_cpu "$case_name" --n "$N_SCALE" --solver "$SOLVER" --out "$out_dir" \
        2>&1 | tee "$temp_log"

    # Parse [TIMING] line from main_cpu output
    local n nx ny steps elapsed steps_per_s Mcell cells
    n=$(           parse_timing "$temp_log" "n")
    nx=$(          parse_timing "$temp_log" "nx")
    ny=$(          parse_timing "$temp_log" "ny")
    steps=$(       parse_timing "$temp_log" "steps")
    elapsed=$(     parse_timing "$temp_log" "elapsed_s")
    steps_per_s=$( parse_timing "$temp_log" "steps_per_s")
    Mcell=$(       parse_timing "$temp_log" "Mcell_updates_s")
    cells=$(( ${nx:-0} * ${ny:-0} ))

    # Parse /usr/bin/time output
    local real user sys rss
    real=$(awk -F= '/real_seconds/{print $2}' "$temp_time")
    user=$(awk -F= '/user_seconds/{print $2}' "$temp_time")
    sys=$( awk -F= '/sys_seconds/{print $2}'  "$temp_time")
    rss=$( awk -F= '/max_rss_kb/{print $2}'   "$temp_time")

    echo ""
    echo "----- TIMING SUMMARY: ${case_name} -----"
    printf "  %-22s %s\n"  "Grid (nx × ny):"       "${nx} × ${ny}  (${cells} cells)"
    printf "  %-22s %s\n"  "Steps:"                 "${steps}"
    printf "  %-22s %s s\n" "Solver wall time:"     "${elapsed}"
    printf "  %-22s %s steps/s\n" "Throughput:"     "${steps_per_s}"
    printf "  %-22s %s M cell-updates/s\n" "Cell throughput:" "${Mcell}"
    printf "  %-22s %s s\n" "real (wall):"          "${real}"
    printf "  %-22s %s s\n" "user (CPU):"           "${user}"
    printf "  %-22s %s s\n" "sys:"                  "${sys}"
    printf "  %-22s %s KB\n" "Max RSS:"             "${rss}"
    echo "----------------------------------------"
    echo ""

    echo "${case_name},${SOLVER},${n:-${N_SCALE}},${OMP_THREADS},${nx},${ny},${cells},${steps},${elapsed},${steps_per_s},${Mcell},${real},${user},${sys},${rss},${GIT_BRANCH},${GIT_COMMIT}" \
        >> "$SUMMARY"

    rm -f "$temp_log" "$temp_time"
}

# ------------------------------------------------------------------
# Run both cases
# ------------------------------------------------------------------
for CASE in "${CASES[@]}"; do
    run_case "$CASE"
done

# ------------------------------------------------------------------
# Final comparison table
# ------------------------------------------------------------------
echo "===== TIMING COMPARISON ====="
echo ""
printf "%-20s  %10s  %10s  %10s  %18s\n" \
    "Case" "nx×ny" "Steps" "Elapsed(s)" "Mcell-updates/s"
printf "%-20s  %10s  %10s  %10s  %18s\n" \
    "--------------------" "----------" "----------" "----------" "------------------"

# Re-read from CSV (skip header)
tail -n +2 "$SUMMARY" | while IFS=',' read -r \
    case solver n threads nx ny cells steps elapsed steps_per_s Mcell \
    real user sys rss branch commit; do
    printf "%-20s  %10s  %10s  %10s  %18s\n" \
        "$case" "${nx}x${ny}" "$steps" "$elapsed" "$Mcell"
done

echo ""
echo "Output dir: outputs/${SLURM_JOB_ID}/"
echo "Full CSV:   ${SUMMARY}"
echo "===== END ====="
date
