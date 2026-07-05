#!/bin/bash -l
#SBATCH -J mhd_profile_gpu
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
# Nsight Systems + Nsight Compute profiling on the DGX "debug" partition.
# Produces one .nsys-rep (timeline) and one .ncu-rep (kernel metrics) per
# case/solver/n, under profiling/, to be pulled down and opened in the
# Nsight Systems / Nsight Compute GUI locally.
#
# Only ONE case/solver/n combination is profiled per submission (unlike
# the timing sweep scripts) — ncu's `--set full` replays every captured
# kernel launch many times to collect all metrics, so sweeping the same
# grid as slurm_gpu.sh would take far too long. Override on the command
# line, e.g.:
#   CASE=rotor SOLVER=hlld N_SCALE=2 sbatch scripts/dgx_slurm/slurm_profile_gpu.sh
#
# ncu only replays the first NCU_LAUNCH_COUNT kernel launches (default
# 100, skipping NCU_LAUNCH_SKIP first) rather than the whole run: the
# solver launches the same handful of kernels every timestep, so the
# first N launches are already representative of the per-step hot loop.
#
# nvcc/nsys/ncu are NOT on PATH by default on this cluster — `module load
# cuda/12.2` is required first (same toolkit version as slurm_gpu.sh, for
# the same driver-compatibility reason).
# ============================================================

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
WORKDIR="${WORKDIR:-/aifs4su/hansirui_2nd/harry/MHD_gpu_acc}"
cd "$WORKDIR"

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT=$(git rev-parse --short HEAD   2>/dev/null || echo "unknown")

mkdir -p logs profiling

module load cuda/12.2

CASE="${CASE:-orszag_tang}"
SOLVER="${SOLVER:-hlld}"
N_SCALE="${N_SCALE:-2}"

NCU_SET="${NCU_SET:-full}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-100}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-0}"

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
echo "Case                : ${CASE}"
echo "Solver              : ${SOLVER}"
echo "N scale             : ${N_SCALE}"
echo "ncu --set           : ${NCU_SET}"
echo "ncu launch count    : ${NCU_LAUNCH_COUNT} (skip ${NCU_LAUNCH_SKIP})"
echo "Makefile CUDA_ARCH  : ${MAKEFILE_CUDA_ARCH}"
echo ""

echo "===== GPU INFO ====="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv || true
echo ""

echo "===== ENV CHECK ====="
for tool in nvcc nsys ncu; do
    echo "which ${tool}: $(which "${tool}" 2>/dev/null || echo 'NOT FOUND')"
done
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[ERROR] nvcc not found even after 'module load cuda/12.2'. Run 'module avail cuda' to check the exact module name on this cluster."
    exit 1
fi
if ! command -v nsys >/dev/null 2>&1 || ! command -v ncu >/dev/null 2>&1; then
    echo "[ERROR] nsys/ncu not found. They normally ship inside the CUDA toolkit module; check 'module avail nsight' if this toolkit doesn't include them."
    exit 1
fi
nvcc --version
nsys --version
ncu --version

echo ""
echo "===== BUILD (with -lineinfo for ncu source correlation) ====="
make clean
make gpu NVCC_EXTRA_FLAGS=-lineinfo

TAG="${CASE}_${SOLVER}_n${N_SCALE}_${SLURM_JOB_ID}"

echo ""
echo "===== NSIGHT SYSTEMS: nsys profile ====="
NSYS_OUT="profiling/nsys_${TAG}"
nsys profile \
    --force-overwrite=true \
    --trace=cuda,osrt,nvtx \
    --sample=cpu \
    --output="${NSYS_OUT}" \
    --stats=true \
    ./bin/main_gpu "$N_SCALE" --case "$CASE" --solver "$SOLVER" --no-out
echo "Saved: ${NSYS_OUT}.nsys-rep"

echo ""
echo "===== NSIGHT COMPUTE: ncu profile ====="
NCU_OUT="profiling/ncu_${TAG}"
ncu \
    --force-overwrite \
    --set "${NCU_SET}" \
    --launch-skip "${NCU_LAUNCH_SKIP}" \
    --launch-count "${NCU_LAUNCH_COUNT}" \
    --export "${NCU_OUT}" \
    ./bin/main_gpu "$N_SCALE" --case "$CASE" --solver "$SOLVER" --no-out
echo "Saved: ${NCU_OUT}.ncu-rep"

echo ""
echo "===== DONE ====="
echo "Pull these down and open locally with:"
echo "  nsys-ui ${NSYS_OUT}.nsys-rep"
echo "  ncu-ui  ${NCU_OUT}.ncu-rep"
date
