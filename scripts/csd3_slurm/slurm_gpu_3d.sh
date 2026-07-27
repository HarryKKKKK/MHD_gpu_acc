#!/bin/bash -l
#SBATCH --job-name=euler3d_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${WORKDIR}"
RESOLUTION="${RESOLUTION:-128}"
SOLVER="${SOLVER:-hllc}"
CFL="${CFL:-0.20}"
T_END="${T_END:-0.10}"
CUDA_ARCH="${CUDA_ARCH:--arch=sm_80}"
make gpu_3d test_gpu_3d CUDA_ARCH="${CUDA_ARCH}"
srun --ntasks=1 ./bin/main_gpu_3d --case blast \
    --resolution "${RESOLUTION}" --solver "${SOLVER}" \
    --cfl "${CFL}" --t-end "${T_END}" --no-out
