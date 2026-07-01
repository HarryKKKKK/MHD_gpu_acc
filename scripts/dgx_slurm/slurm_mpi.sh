#!/bin/bash -l
#SBATCH -J mhd_mpi
#SBATCH -A hansirui
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# Pure-MPI timing sweep on the DGX "debug" partition. No --exclusive
# (see slurm_cpu_omp.sh for rationale — allocation here is sub-node,
# not whole-node).
#
# --ntasks=1 --cpus-per-task=112 (NOT --ntasks=112 --cpus-per-task=1):
# mpirun is forked directly from the sbatch *batch script*, not launched
# via `srun`. Slurm only gives the batch script itself the CPU affinity
# of ONE task's worth of cpus-per-task; --ntasks is reserved for
# separate srun-launched steps. With --ntasks=112 --cpus-per-task=1 the
# batch script (and therefore mpirun and everything it forks) is only
# bound to 1 CPU, and `mpirun --bind-to core` fails with "would result
# in binding more processes than cpus on a resource". Requesting the
# same shape as the OMP script (one task, 112 cpus) gives the batch
# script — and mpirun's local fork of 112 ranks inside it — the full
# 112-cpu affinity mask instead.
#
# 112 single-threaded MPI ranks, one per physical core, all on one node
# (mirrors the OMP script's core count so the two are comparable;
# hyperthreads unused).
#
# mpicxx/mpirun (OpenMPI 4.1.7a1) are already on PATH by default on
# this cluster — no `module load` needed.
#
# Cases   : orszag_tang, rotor
# Solvers : hll hllc hlld force
# Scales  : n = 1, 2, 4
#
# Override on the command line before sbatch, e.g.:
#   RANKS=56 SCALES_STR="1 2" sbatch scripts/dgx_slurm/slurm_mpi.sh
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

RANKS="${RANKS:-${SLURM_CPUS_PER_TASK:-112}}"
# Pure MPI: one thread per rank, no OpenMP oversubscription within a rank.
export OMP_NUM_THREADS=1

echo "===== JOB INFO ====="
echo "JobID      : ${SLURM_JOB_ID}"
echo "Host       : $(hostname)"
echo "Start      : $(date)"
echo "Workdir    : ${WORKDIR}"
echo "Git Branch : ${GIT_BRANCH}"
echo "Git Commit : ${GIT_COMMIT}"
echo "Partition  : ${SLURM_JOB_PARTITION:-unknown}"
echo "MPI ranks  : ${RANKS}"
echo "Cases      : ${CASES[*]}"
echo "Solvers    : ${SOLVERS[*]}"
echo "Scales (n) : ${SCALES[*]}"
echo ""

echo "===== CPU TOPOLOGY ====="
lscpu | grep -E 'CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)' || true
echo ""

echo "===== ENV CHECK ====="
echo "which mpicxx: $(which mpicxx || echo 'NOT FOUND')"
echo "which mpirun: $(which mpirun || echo 'NOT FOUND')"
if ! command -v mpicxx >/dev/null 2>&1; then echo "[ERROR] mpicxx not found in PATH."; exit 1; fi
if ! command -v mpirun >/dev/null 2>&1; then echo "[ERROR] mpirun not found in PATH."; exit 1; fi

echo ""
echo "===== BUILD ====="
# No `make clean`: build/mpi and bin/main_mpi are private to the `mpi`
# target — safe to build concurrently with the CPU/GPU sbatch scripts.
make mpi

SUMMARY="validation/mpi_${SLURM_JOB_ID}.csv"
echo "arch,case,solver,n,ranks,nx,ny,total_cells,steps,elapsed_s,steps_per_s,Mcell_updates_s,real_s,user_s,sys_s,max_rss_kb,git_branch,git_commit" \
    > "$SUMMARY"

parse_timing_line() {
    grep '^\[TIMING\]' "$1" | grep -oP "(?<=${2}=)[^ ]+" | tail -1
}

run_and_record() {
    local case_name="$1" solver="$2" n_scale="$3"
    echo ""
    echo "===== MPI RUN: case=${case_name}, solver=${solver}, n=${n_scale}, ranks=${RANKS} ====="

    local temp_log temp_time
    temp_log=$(mktemp); temp_time=$(mktemp)

    /usr/bin/time -f "real_seconds=%e\nuser_seconds=%U\nsys_seconds=%S\nmax_rss_kb=%M" \
        -o "$temp_time" \
        mpirun -np "$RANKS" --bind-to core --map-by core \
        ./bin/main_mpi "$case_name" --n "$n_scale" --solver "$solver" --no-out \
        2>&1 | tee "$temp_log"

    local n nx ny ranks_reported steps elapsed steps_per_s Mcell
    n=$(              parse_timing_line "$temp_log" "n")
    nx=$(             parse_timing_line "$temp_log" "nx")
    ny=$(             parse_timing_line "$temp_log" "ny")
    ranks_reported=$( parse_timing_line "$temp_log" "ranks")
    steps=$(          parse_timing_line "$temp_log" "steps")
    elapsed=$(        parse_timing_line "$temp_log" "elapsed_s")
    steps_per_s=$(    parse_timing_line "$temp_log" "steps_per_s")
    Mcell=$(          parse_timing_line "$temp_log" "Mcell_updates_s")
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

    echo "mpi,${case_name},${solver},${n:-${n_scale}},${ranks_reported:-${RANKS}},${nx},${ny},${cells},${steps},${elapsed},${steps_per_s},${Mcell},${real},${user},${sys},${rss},${GIT_BRANCH},${GIT_COMMIT}" \
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
