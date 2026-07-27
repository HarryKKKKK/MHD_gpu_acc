#!/bin/bash -l
#SBATCH --job-name=euler3d_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

set -euo pipefail
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${WORKDIR}"
RESOLUTION="${RESOLUTION:-64}"
SOLVER="${SOLVER:-hllc}"
CFL="${CFL:-0.20}"
T_END="${T_END:-0.10}"
OMP_THREADS="${OMP_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export OMP_NUM_THREADS="${OMP_THREADS}" OMP_PROC_BIND=close OMP_PLACES=cores
make cpu_3d test_3d
srun --ntasks=1 --cpus-per-task="${OMP_THREADS}" ./bin/main_cpu_3d \
    --case blast --resolution "${RESOLUTION}" --solver "${SOLVER}" \
    --cfl "${CFL}" --t-end "${T_END}" --no-out
