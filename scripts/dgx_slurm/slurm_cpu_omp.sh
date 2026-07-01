#!/bin/bash -l
#SBATCH -J mhd_cpu_omp
#SBATCH -A hansirui
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# CPU OpenMP timing sweep on the DGX "debug" partition (dgx-126/dgx-127,
# 2 nodes shared by several groups). No --exclusive: `scontrol show
# node` on this cluster shows sub-node (cons_res-style) allocation, i.e.
# other jobs can share the rest of the node while this one runs, so an
# exclusive reservation would only block other groups for no benefit.
#
# --cpus-per-task=112 requests all *physical* cores on one node (112
# cores / 224 hardware threads across 2 sockets; hyperthreading is
# intentionally excluded from this benchmark).
#
# Cases   : orszag_tang, rotor
# Solvers : hll hllc hlld force
# Scales  : n = 1, 2, 4  (weak-scaling grid factor; see get_n_case_config
#           in src/test_cases.cpp — nx,ny are multiplied by n, unrelated
#           to core/rank count)
#
# Override any of these on the command line before sbatch, e.g.:
#   SCALES_STR="1 2 4 8" sbatch scripts/dgx_slurm/slurm_cpu_omp.sh
# ============================================================

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
WORKDIR="${WORKDIR:-/aifs4su/hansirui_2nd/harry/MHD_gpu_acc}"
cd "$WORKDIR"

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT=$(git rev-parse --short HEAD   2>/dev/null || echo "unknown")

mkdir -p logs validation outputs

read -r -a CASES   <<< "${CASES_STR:-orszag_tang rotor}"
read -r -a SOLVERS <<< "${SOLVERS_STR:-hll hllc hlld force}"
read -r -a SCALES  <<< "${SCALES_STR:-1 2 4}"

OMP_THREADS="${SLURM_CPUS_PER_TASK:-112}"
export OMP_NUM_THREADS="${OMP_THREADS}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "===== JOB INFO ====="
echo "JobID       : ${SLURM_JOB_ID}"
echo "Host        : $(hostname)"
echo "Start       : $(date)"
echo "Workdir     : ${WORKDIR}"
echo "Git Branch  : ${GIT_BRANCH}"
echo "Git Commit  : ${GIT_COMMIT}"
echo "Partition   : ${SLURM_JOB_PARTITION:-unknown}"
echo "OMP_THREADS : ${OMP_THREADS}"
echo "Cases       : ${CASES[*]}"
echo "Solvers     : ${SOLVERS[*]}"
echo "Scales (n)  : ${SCALES[*]}"
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
# No `make clean`: build/cpu and bin/main_cpu are private to the `cpu`
# target, so this can run concurrently with the MPI/GPU sbatch scripts
# without clobbering their build artefacts.
make cpu

SUMMARY="validation/cpu_omp_${SLURM_JOB_ID}.csv"
echo "arch,case,solver,n,threads,nx,ny,total_cells,steps,elapsed_s,steps_per_s,Mcell_updates_s,real_s,user_s,sys_s,max_rss_kb,git_branch,git_commit" \
    > "$SUMMARY"

parse_timing_line() {
    grep '^\[TIMING\]' "$1" | grep -oP "(?<=${2}=)[^ ]+" | tail -1
}

run_and_record() {
    local case_name="$1" solver="$2" n_scale="$3"
    echo ""
    echo "===== CPU OMP RUN: case=${case_name}, solver=${solver}, n=${n_scale}, threads=${OMP_THREADS} ====="

    local temp_log temp_time
    temp_log=$(mktemp); temp_time=$(mktemp)

    /usr/bin/time -f "real_seconds=%e\nuser_seconds=%U\nsys_seconds=%S\nmax_rss_kb=%M" \
        -o "$temp_time" \
        ./bin/main_cpu "$case_name" --n "$n_scale" --solver "$solver" --no-out \
        2>&1 | tee "$temp_log"

    local n nx ny steps elapsed steps_per_s Mcell
    n=$(           parse_timing_line "$temp_log" "n")
    nx=$(          parse_timing_line "$temp_log" "nx")
    ny=$(          parse_timing_line "$temp_log" "ny")
    steps=$(       parse_timing_line "$temp_log" "steps")
    elapsed=$(     parse_timing_line "$temp_log" "elapsed_s")
    steps_per_s=$( parse_timing_line "$temp_log" "steps_per_s")
    Mcell=$(       parse_timing_line "$temp_log" "Mcell_updates_s")
    local cells=$(( ${nx:-0} * ${ny:-0} ))

    local real user sys rss
    real=$(awk -F= '/real_seconds/{print $2}' "$temp_time")
    user=$(awk -F= '/user_seconds/{print $2}' "$temp_time")
    sys=$( awk -F= '/sys_seconds/{print $2}'  "$temp_time")
    rss=$( awk -F= '/max_rss_kb/{print $2}'   "$temp_time")

    echo "------------------------------------------------------------"
    echo "[TIMING RECORDED] elapsed=${elapsed}s | steps=${steps} | Mcell_updates/s=${Mcell}"
    echo "[WALL TIME]       real=${real}s | user=${user}s | sys=${sys}s | RSS=${rss}KB"
    echo "------------------------------------------------------------"

    echo "cpu_omp,${case_name},${solver},${n:-${n_scale}},${OMP_THREADS},${nx},${ny},${cells},${steps},${elapsed},${steps_per_s},${Mcell},${real},${user},${sys},${rss},${GIT_BRANCH},${GIT_COMMIT}" \
        >> "$SUMMARY"

    rm -f "$temp_log" "$temp_time"
}

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
