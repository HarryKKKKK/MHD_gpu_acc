#!/bin/bash -l
#SBATCH -J kh_gpu
#SBATCH -A hk597
#SBATCH -p csc-mphil-gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=05:59:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
cd "$WORKDIR"

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

MAKEFILE_CUDA_ARCH=$(awk -F ':=' '/^CUDA_ARCH[ \t]*:=/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' Makefile 2>/dev/null || echo "unknown")
GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo "unknown")

mkdir -p logs validation outputs

read -r -a SCALES  <<< "${SCALES_STR:-1}"
read -r -a CASES   <<< "${CASES_STR:-kelvin_helmholtz}"
read -r -a SOLVERS <<< "${SOLVERS_STR:-hll}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "===== JOB INFO ====="
echo "JobID               : ${SLURM_JOB_ID}"
echo "Host                : $(hostname)"
echo "Start               : $(date)"
echo "Workdir             : ${WORKDIR}"
echo "Git Branch          : ${GIT_BRANCH}"
echo "Git Commit          : ${GIT_COMMIT}"
echo "Partition           : ${SLURM_JOB_PARTITION:-unknown}"
echo "SLURM_NTASKS        : ${SLURM_NTASKS:-unset}"
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-unset}"
echo "SLURM_CPUS_ON_NODE  : ${SLURM_CPUS_ON_NODE:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "OMP_NUM_THREADS     : ${OMP_NUM_THREADS}"
echo "SCALES              : ${SCALES[*]}"
echo "CASES               : ${CASES[*]}"
echo "SOLVERS             : ${SOLVERS[*]}"
echo "Makefile CUDA_ARCH  : ${MAKEFILE_CUDA_ARCH}"
echo "GPU compute cap     : ${GPU_COMPUTE_CAP}"
echo ""

echo "===== CPU TOPOLOGY ====="
lscpu | grep -E 'CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)' || true

echo ""
echo "===== GPU INFO ====="
nvidia-smi || true

echo ""
echo "===== BUILD ====="
make clean
make gpu

SUMMARY="validation/gpu_${SLURM_JOB_ID}.csv"
echo "arch,case,solver,n,nx,ny,total_cells,total_steps,real_seconds,user_seconds,sys_seconds,max_rss_kb,output_dir,git_branch,git_commit,cuda_arch" > "$SUMMARY"

run_and_record() {
    local case_name="$1"
    local solver_name="$2"
    local n_scale="$3"

    local out_dir="outputs/gpu_${case_name}_${solver_name}_n${n_scale}_${SLURM_JOB_ID}"
    mkdir -p "$out_dir"

    echo ""
    echo "===== GPU RUN: case=${case_name}, solver=${solver_name}, n=${n_scale} ====="
    echo "Output dir: ${out_dir}"

    local temp_log temp_time
    temp_log=$(mktemp)
    temp_time=$(mktemp)

    /usr/bin/time -f "real_seconds=%e\nuser_seconds=%U\nsys_seconds=%S\nmax_rss_kb=%M" \
        -o "$temp_time" \
        ./main_gpu "$n_scale" --case "$case_name" --solver "$solver_name" --out "$out_dir" \
        2>&1 | tee "$temp_log"

    local nx ny cells steps real user sys rss
    nx=$(awk    -F ':'  '/\[GPU\] nx/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$temp_log")
    ny=$(awk    -F ':'  '/\[GPU\] ny/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$temp_log")
    cells=$(awk -F ':'  '/\[GPU\] total_cells/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$temp_log")
    steps=$(awk -F '='  '/\[GPU\] Total steps/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$temp_log")
    real=$(awk  -F '='  '/real_seconds/{print $2; exit}' "$temp_time")
    user=$(awk  -F '='  '/user_seconds/{print $2; exit}' "$temp_time")
    sys=$(awk   -F '='  '/sys_seconds/{print $2; exit}'  "$temp_time")
    rss=$(awk   -F '='  '/max_rss_kb/{print $2; exit}'   "$temp_time")

    echo "------------------------------------------------------------"
    echo "[TIMING RECORDED] Real: ${real}s | User: ${user}s | Sys: ${sys}s | Max RSS: ${rss} KB"
    echo "[OUTPUT SAVED] ${out_dir}"
    echo "------------------------------------------------------------"

    echo "gpu,${case_name},${solver_name},${n_scale},${nx},${ny},${cells},${steps},${real},${user},${sys},${rss},${out_dir},${GIT_BRANCH},${GIT_COMMIT},${MAKEFILE_CUDA_ARCH}" >> "$SUMMARY"

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