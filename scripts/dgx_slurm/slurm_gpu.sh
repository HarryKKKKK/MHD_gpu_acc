#!/bin/bash -l
#SBATCH -J mhd_gpu
#SBATCH -A hansirui
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# GPU (CUDA) timing sweep on the DGX "debug" partition. Only dgx-127 in
# this partition has GPUs (gpu:3, i.e. 3x H800); --gres=gpu:1 requests
# one, leaving the other two free for other jobs/groups. main_gpu is
# single-GPU code (no NCCL/MPI), so it never needs more than 1.
# --cpus-per-task=8: the host side here is only I/O/bookkeeping, not a
# hot loop, so (unlike the OMP/MPI scripts) it doesn't need many cores.
#
# nvcc is NOT on PATH by default on this cluster — `module load
# cuda/12.2` is required first. 12.2 is picked because `nvidia-smi`
# reports this node's driver (535.129.03) supports CUDA runtime up to
# 12.2; a newer toolkit risks "CUDA driver version is insufficient for
# CUDA runtime version" at run time. GPU compute capability is 9.0
# (H800), which already matches Makefile's `CUDA_ARCH := -arch=sm_90`,
# so no Makefile change is needed on this cluster.
#
# Cases   : orszag_tang, rotor
# Solvers : hll hllc hlld force
# Scales  : n = 1, 2, 4
#
# Override on the command line before sbatch, e.g.:
#   SOLVERS_STR="hlld" sbatch scripts/dgx_slurm/slurm_gpu.sh
# ============================================================

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
WORKDIR="${WORKDIR:-/aifs4su/hansirui_2nd/harry/MHD_gpu_acc}"
cd "$WORKDIR"

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT=$(git rev-parse --short HEAD   2>/dev/null || echo "unknown")

mkdir -p logs validation outputs

module load cuda/12.2

read -r -a CASES   <<< "${CASES_STR:-orszag_tang rotor}"
read -r -a SOLVERS <<< "${SOLVERS_STR:-hll hllc hlld force}"
read -r -a SCALES  <<< "${SCALES_STR:-1 2 4}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

MAKEFILE_CUDA_ARCH=$(awk -F ':=' '/^CUDA_ARCH[ \t]*:=/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' Makefile 2>/dev/null || echo "unknown")

echo "===== JOB INFO ====="
echo "JobID               : ${SLURM_JOB_ID}"
echo "Host                : $(hostname)"
echo "Start               : $(date)"
echo "Workdir             : ${WORKDIR}"
echo "Git Branch          : ${GIT_BRANCH}"
echo "Git Commit          : ${GIT_COMMIT}"
echo "Partition           : ${SLURM_JOB_PARTITION:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Cases               : ${CASES[*]}"
echo "Solvers             : ${SOLVERS[*]}"
echo "Scales (n)          : ${SCALES[*]}"
echo "Makefile CUDA_ARCH  : ${MAKEFILE_CUDA_ARCH}"
echo ""

echo "===== GPU INFO ====="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv || true
echo ""

echo "===== ENV CHECK ====="
echo "which nvcc: $(which nvcc || echo 'NOT FOUND')"
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[ERROR] nvcc not found even after 'module load cuda/12.2'. Run 'module avail cuda' to check the exact module name on this cluster."
    exit 1
fi
nvcc --version

echo ""
echo "===== BUILD ====="
# No `make clean`: build/gpu and bin/main_gpu are private to the `gpu`
# target — safe to build concurrently with the CPU/MPI sbatch scripts.
make gpu

SUMMARY="validation/gpu_${SLURM_JOB_ID}.csv"
echo "arch,case,solver,n,nx,ny,total_cells,total_steps,real_seconds,user_seconds,sys_seconds,max_rss_kb,git_branch,git_commit,cuda_arch" > "$SUMMARY"

run_and_record() {
    local case_name="$1" solver="$2" n_scale="$3"
    echo ""
    echo "===== GPU RUN: case=${case_name}, solver=${solver}, n=${n_scale} ====="

    local temp_log temp_time
    temp_log=$(mktemp); temp_time=$(mktemp)

    /usr/bin/time -f "real_seconds=%e\nuser_seconds=%U\nsys_seconds=%S\nmax_rss_kb=%M" \
        -o "$temp_time" \
        ./bin/main_gpu "$n_scale" --case "$case_name" --solver "$solver" --no-out \
        2>&1 | tee "$temp_log"

    local nx ny cells steps real user sys rss
    nx=$(   awk -F ':' '/\[GPU\] nx/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$temp_log")
    ny=$(   awk -F ':' '/\[GPU\] ny/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$temp_log")
    cells=$(awk -F ':' '/\[GPU\] total_cells/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$temp_log")
    steps=$(awk -F '=' '/\[GPU\] Total steps/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$temp_log")
    real=$(awk -F '=' '/real_seconds/{print $2; exit}' "$temp_time")
    user=$(awk -F '=' '/user_seconds/{print $2; exit}' "$temp_time")
    sys=$( awk -F '=' '/sys_seconds/{print $2; exit}'  "$temp_time")
    rss=$( awk -F '=' '/max_rss_kb/{print $2; exit}'   "$temp_time")

    echo "------------------------------------------------------------"
    echo "[TIMING RECORDED] Real: ${real}s | User: ${user}s | Sys: ${sys}s | Max RSS: ${rss} KB"
    echo "------------------------------------------------------------"

    echo "gpu,${case_name},${solver},${n_scale},${nx},${ny},${cells},${steps},${real},${user},${sys},${rss},${GIT_BRANCH},${GIT_COMMIT},${MAKEFILE_CUDA_ARCH}" >> "$SUMMARY"

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
