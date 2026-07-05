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

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT=$(git rev-parse --short HEAD   2>/dev/null || echo "unknown")

mkdir -p logs validation outputs

# Override any of these on the command line before sbatch, e.g.:
#   SCALES_STR="1 2 4" CASES_STR="kelvin_helmholtz" SOLVERS_STR="hlld" sbatch scripts/slurm_cpu.sh
#
# SCALES_STR : weak-scaling factors n (grid scaled n× in each dimension per run)
# CASES_STR  : space-separated test case names
# SOLVERS_STR: space-separated Riemann solver names (hll hlld force)
read -r -a SCALES  <<< "${SCALES_STR:-1}"
read -r -a CASES   <<< "${CASES_STR:-kelvin_helmholtz}"
read -r -a SOLVERS <<< "${SOLVERS_STR:-hll hlld force}"

# ------------------------------------------------------------------
# Detect how many physical cores are available on this node.
# ------------------------------------------------------------------
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
echo "Git Branch         : ${GIT_BRANCH}"
echo "Git Commit         : ${GIT_COMMIT}"
echo "Partition          : ${SLURM_JOB_PARTITION:-unknown}"
echo "SLURM_CPUS_ON_NODE : ${SLURM_CPUS_ON_NODE:-unset}"
echo "Detected CPUs/node : ${CPUS_ON_NODE}"
echo "OMP_THREADS        : ${OMP_THREADS}"
echo "SCALES             : ${SCALES[*]}"
echo "CASES              : ${CASES[*]}"
echo "SOLVERS            : ${SOLVERS[*]}"
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

# ------------------------------------------------------------------
# Summary CSV header
# ------------------------------------------------------------------
SUMMARY="validation/cpu_omp_${SLURM_JOB_ID}.csv"
echo "arch,case,solver,n,threads,nx,ny,total_cells,steps,elapsed_s,steps_per_s,Mcell_updates_s,real_s,user_s,sys_s,max_rss_kb,git_branch,git_commit" \
    > "$SUMMARY"

# ------------------------------------------------------------------
# Parse the machine-readable [TIMING] line emitted by main_cpu.
# Format: [TIMING] n=N nx=NX ny=NY steps=S elapsed_s=E steps_per_s=P Mcell_updates_s=M
# ------------------------------------------------------------------
parse_timing_line() {
    local log="$1"
    local field="$2"
    grep '^\[TIMING\]' "$log" | grep -oP "(?<=${field}=)[^ ]+" | tail -1
}

# ------------------------------------------------------------------
# One run: build a row in the summary CSV.
# ------------------------------------------------------------------
run_and_record() {
    local case_name="$1"
    local solver="$2"
    local n_scale="$3"

    local out_dir="outputs/cpu_${case_name}_${solver}_n${n_scale}_t${OMP_THREADS}"
    mkdir -p "$out_dir"

    echo ""
    echo "===== CPU OMP RUN: case=${case_name}, solver=${solver}, n=${n_scale}, threads=${OMP_THREADS} ====="

    local temp_log temp_time
    temp_log=$(mktemp)
    temp_time=$(mktemp)

    /usr/bin/time -f "real_seconds=%e\nuser_seconds=%U\nsys_seconds=%S\nmax_rss_kb=%M" \
        -o "$temp_time" \
        ./bin/main_cpu "$case_name" --n "$n_scale" --solver "$solver" --out "$out_dir" \
        2>&1 | tee "$temp_log"

    echo "----- TIME -----"
    cat "$temp_time"

    # Parse fields from the [TIMING] line
    local n nx ny steps elapsed steps_per_s Mcell
    n=$(        parse_timing_line "$temp_log" "n")
    nx=$(       parse_timing_line "$temp_log" "nx")
    ny=$(       parse_timing_line "$temp_log" "ny")
    steps=$(    parse_timing_line "$temp_log" "steps")
    elapsed=$(  parse_timing_line "$temp_log" "elapsed_s")
    steps_per_s=$(parse_timing_line "$temp_log" "steps_per_s")
    Mcell=$(    parse_timing_line "$temp_log" "Mcell_updates_s")
    local cells=$(( ${nx:-0} * ${ny:-0} ))

    # Parse /usr/bin/time fields
    local real user sys rss
    real=$(awk -F '=' '/real_seconds/{print $2}' "$temp_time")
    user=$(awk -F '=' '/user_seconds/{print $2}' "$temp_time")
    sys=$( awk -F '=' '/sys_seconds/{print $2}'  "$temp_time")
    rss=$( awk -F '=' '/max_rss_kb/{print $2}'   "$temp_time")

    echo "------------------------------------------------------------"
    echo "[TIMING RECORDED] elapsed=${elapsed}s | steps=${steps} | Mcell_updates/s=${Mcell}"
    echo "[WALL TIME]       real=${real}s | user=${user}s | sys=${sys}s | RSS=${rss}KB"
    echo "------------------------------------------------------------"

    echo "cpu_omp,${case_name},${solver},${n:-${n_scale}},${OMP_THREADS},${nx},${ny},${cells},${steps},${elapsed},${steps_per_s},${Mcell},${real},${user},${sys},${rss},${GIT_BRANCH},${GIT_COMMIT}" \
        >> "$SUMMARY"

    rm -f "$temp_log" "$temp_time"
}

# ------------------------------------------------------------------
# Main sweep: for each (n, case, solver) combination
# ------------------------------------------------------------------
for N in "${SCALES[@]}"; do
    for CASE in "${CASES[@]}"; do
        for SOLVER in "${SOLVERS[@]}"; do
            run_and_record "$CASE" "$SOLVER" "$N"
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
