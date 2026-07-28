#!/bin/bash -l
#SBATCH -J mhd_gpu3d_base
#SBATCH -A hansirui
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
WORKDIR="${WORKDIR:-/aifs4su/hansirui_2nd/harry/MHD_gpu_acc}"
cd "$WORKDIR"
mkdir -p logs validation

module load cuda/12.2

read -r -a CASES <<< "${CASES_STR:-blast imtg}"
read -r -a SOLVERS <<< "${SOLVERS_STR:-hlld}"
read -r -a RESOLUTIONS <<< "${RESOLUTIONS_STR:-128}"
MAX_STEPS="${MAX_STEPS:-0}"
BLAST_T_END="${BLAST_T_END:-0.10}"
IMTG_T_END="${IMTG_T_END:-5.809475019311126}"
CFL="${CFL:-0.20}"
SNAPSHOTS="${SNAPSHOTS:-5}"

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

echo "===== 3D BASELINE JOB ====="
echo "Host: $(hostname)"
echo "Branch: ${GIT_BRANCH}"
echo "Commit: ${GIT_COMMIT}"
echo "Cases: ${CASES[*]}"
echo "Solvers: ${SOLVERS[*]}"
echo "Resolutions: ${RESOLUTIONS[*]}"
echo "Max steps: ${MAX_STEPS} (0 means run to case t_end)"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
nvcc --version

make gpu_3d_baseline

# Compile/run gate before expensive measurements. Both supported cases must
# complete two steps and exercise their different x/y boundary conditions.
for case_name in "${CASES[@]}"; do
    echo "===== SMOKE: ${case_name} ====="
    if [[ "${case_name}" == "imtg" ]]; then
        case_t_end="${IMTG_T_END}"
    else
        case_t_end="${BLAST_T_END}"
    fi
    ./bin/main_gpu_3d_baseline \
        --case "$case_name" --solver hlld \
        --resolution 16 --t-end "${case_t_end}" --cfl "${CFL}" \
        --snapshots "${SNAPSHOTS}" \
        --max-steps 2 --no-out
done

SUMMARY="validation/gpu_3d_baseline_${SLURM_JOB_ID}.csv"
echo "arch,case,solver,nx,ny,nz,total_cells,total_steps,final_time,gpu_seconds,wall_seconds,git_branch,git_commit" > "$SUMMARY"

for resolution in "${RESOLUTIONS[@]}"; do
    for case_name in "${CASES[@]}"; do
        for solver in "${SOLVERS[@]}"; do
            echo "===== RUN: case=${case_name} solver=${solver} N=${resolution} ====="
            if [[ "${case_name}" == "imtg" ]]; then
                case_t_end="${IMTG_T_END}"
            else
                case_t_end="${BLAST_T_END}"
            fi
            temp_log=$(mktemp)
            ./bin/main_gpu_3d_baseline \
                --case "$case_name" --solver "$solver" \
                --resolution "$resolution" --t-end "${case_t_end}" \
                --cfl "${CFL}" --snapshots "${SNAPSHOTS}" \
                --max-steps "$MAX_STEPS" --no-out \
                2>&1 | tee "$temp_log"

            nx=$(awk -F ':' '/\[GPU3D\] nx/{gsub(/[ \t]/,"",$2);print $2;exit}' "$temp_log")
            ny=$(awk -F ':' '/\[GPU3D\] ny/{gsub(/[ \t]/,"",$2);print $2;exit}' "$temp_log")
            nz=$(awk -F ':' '/\[GPU3D\] nz/{gsub(/[ \t]/,"",$2);print $2;exit}' "$temp_log")
            cells=$(awk -F ':' '/\[GPU3D\] total_cells/{gsub(/[ \t]/,"",$2);print $2;exit}' "$temp_log")
            steps=$(awk -F '=' '/\[GPU3D\] Total steps/{gsub(/[ \t]/,"",$2);print $2;exit}' "$temp_log")
            final_time=$(awk -F '=' '/\[GPU3D\] Final time/{gsub(/[ \t]/,"",$2);print $2;exit}' "$temp_log")
            gpu_seconds=$(awk -F '=' '/\[GPU3D\] GPU elapsed/{gsub(/[ s\t]/,"",$2);print $2;exit}' "$temp_log")
            wall_seconds=$(awk -F '=' '/\[GPU3D\] Wall elapsed/{gsub(/[ s\t]/,"",$2);print $2;exit}' "$temp_log")

            echo "gpu3d_baseline,${case_name},${solver},${nx},${ny},${nz},${cells},${steps},${final_time},${gpu_seconds},${wall_seconds},${GIT_BRANCH},${GIT_COMMIT}" >> "$SUMMARY"
            rm -f "$temp_log"
        done
    done
done

echo "===== SUMMARY ====="
cat "$SUMMARY"
