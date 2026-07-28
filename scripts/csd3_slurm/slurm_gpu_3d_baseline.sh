#!/bin/bash -l
#SBATCH -J mhd3d_base
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# CSD3 timing job for the unoptimised 3D CUDA baseline.
#
# Submit from the repository root after creating the Slurm log directory:
#   mkdir -p logs
#   sbatch scripts/csd3_slurm/slurm_gpu_3d_baseline.sh
#
# Example report sweep:
#   export RESOLUTIONS_STR="64 96 128" SOLVERS_STR="hlld"
#   sbatch --export=ALL,RESOLUTIONS_STR,SOLVERS_STR \
#     scripts/csd3_slurm/slurm_gpu_3d_baseline.sh

set -euo pipefail

JOB_ID="${SLURM_JOB_ID:-manual}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SUBMIT_DIR}}"
cd "${WORKDIR}"

if [[ ! -f Makefile ]]; then
    echo "[ERROR] Submit from the MHD repository root or set WORKDIR."
    exit 1
fi

mkdir -p logs timing/gpu_3d_baseline

echo "===== MODULE SETUP ====="
if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash \
                /usr/local/Modules/init/bash; do
        if [[ -f "${init}" ]]; then
            source "${init}"
            break
        fi
    done
fi
if ! command -v module >/dev/null 2>&1; then
    echo "[ERROR] Environment Modules is unavailable."
    exit 1
fi

module purge
module load rhel8/default-amp
module list 2>&1 || true

for tool in nvcc nvidia-smi make; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "[ERROR] ${tool} is unavailable after loading rhel8/default-amp."
        exit 1
    fi
done

read -r -a CASES <<< "${CASES_STR:-orszag_tang rotor}"
read -r -a SOLVERS <<< "${SOLVERS_STR:-hlld}"
read -r -a RESOLUTIONS <<< "${RESOLUTIONS_STR:-64}"
MAX_STEPS="${MAX_STEPS:-0}"
SMOKE_RESOLUTION="${SMOKE_RESOLUTION:-16}"
SMOKE_STEPS="${SMOKE_STEPS:-2}"
MAKE_JOBS="${MAKE_JOBS:-8}"

if (( ${#CASES[@]} == 0 || ${#SOLVERS[@]} == 0 ||
      ${#RESOLUTIONS[@]} == 0 )); then
    echo "[ERROR] CASES_STR, SOLVERS_STR, and RESOLUTIONS_STR must not be empty."
    exit 2
fi

for case_name in "${CASES[@]}"; do
    if [[ "${case_name}" != "orszag_tang" && "${case_name}" != "rotor" ]]; then
        echo "[ERROR] Unsupported case '${case_name}'. Use orszag_tang or rotor."
        exit 2
    fi
done

for resolution in "${RESOLUTIONS[@]}"; do
    if ! [[ "${resolution}" =~ ^[1-9][0-9]*$ ]] || (( resolution < 8 )); then
        echo "[ERROR] Every resolution must be an integer >= 8: '${resolution}'."
        exit 2
    fi
done

GPU_NAME="$(
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null |
    head -1 | xargs
)"
COMPUTE_CAP="$(
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
    head -1 | tr -d '[:space:]' || true
)"
if [[ "${COMPUTE_CAP}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    CUDA_SM="${COMPUTE_CAP/./}"
elif [[ "${GPU_NAME}" == *A100* || "${GPU_NAME}" == *A30* ]]; then
    CUDA_SM=80
elif [[ "${GPU_NAME}" == *A10* || "${GPU_NAME}" == *A40* ]]; then
    CUDA_SM=86
else
    CUDA_SM="${CUDA_SM:-80}"
fi
CUDA_ARCH_FLAG="${CUDA_ARCH_FLAG:--arch=sm_${CUDA_SM}}"

GIT_BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_COMMIT_MSG="$(git log -1 --pretty=%s 2>/dev/null || echo unknown)"
BUILD_ROOT="${BUILD_ROOT:-build/csd3_gpu3d_baseline_${JOB_ID}}"
BIN_ROOT="${BIN_ROOT:-bin/csd3_gpu3d_baseline_${JOB_ID}}"
BIN="${BIN_ROOT}/main_gpu_3d_baseline"

echo "===== CSD3 3D GPU BASELINE JOB ====="
echo "Job ID              : ${JOB_ID}"
echo "Host                : $(hostname)"
echo "Start               : $(date --iso-8601=seconds)"
echo "Workdir             : ${WORKDIR}"
echo "Git branch          : ${GIT_BRANCH}"
echo "Git commit          : ${GIT_COMMIT}"
echo "GPU                 : ${GPU_NAME:-unknown}"
echo "Compute capability  : ${COMPUTE_CAP:-unknown}"
echo "CUDA architecture   : ${CUDA_ARCH_FLAG}"
echo "Cases               : ${CASES[*]}"
echo "Solvers             : ${SOLVERS[*]}"
echo "Resolutions         : ${RESOLUTIONS[*]}"
echo "Maximum steps       : ${MAX_STEPS} (0 = case t_end)"
echo "Build root          : ${BUILD_ROOT}"
echo "Binary root         : ${BIN_ROOT}"
echo
nvidia-smi
nvcc --version

echo "===== BUILD ====="
make -j "${MAKE_JOBS}" gpu_3d_baseline \
    CUDA_ARCH="${CUDA_ARCH_FLAG}" \
    BUILD_DIR="${BUILD_ROOT}" \
    BIN_DIR="${BIN_ROOT}"

if [[ ! -x "${BIN}" ]]; then
    echo "[ERROR] ${BIN} was not produced or is not executable."
    exit 1
fi

echo "===== TWO-CASE SMOKE GATE ====="
for case_name in "${CASES[@]}"; do
    echo "----- ${case_name}: ${SMOKE_RESOLUTION}^3, ${SMOKE_STEPS} steps -----"
    srun --ntasks=1 "${BIN}" \
        --case "${case_name}" \
        --solver hlld \
        --resolution "${SMOKE_RESOLUTION}" \
        --max-steps "${SMOKE_STEPS}" \
        --no-out
done

SUMMARY="timing/gpu_3d_baseline/csd3_gpu3d_baseline_${JOB_ID}.csv"
{
    echo "# commit_message: ${GIT_COMMIT_MSG}"
    echo "arch,case,solver,nx,ny,nz,total_cells,total_steps,final_time,gpu_seconds,wall_seconds,process_real_seconds,max_rss_kb,git_branch,git_commit,cuda_arch"
} > "${SUMMARY}"

run_and_record() {
    local case_name="$1"
    local solver="$2"
    local resolution="$3"
    local temp_log
    local temp_time
    local run_status
    temp_log="$(mktemp)"
    temp_time="$(mktemp)"

    echo
    echo "============================================================"
    echo "3D BASELINE: case=${case_name} solver=${solver} N=${resolution}"
    echo "============================================================"

    local app=(
        "${BIN}"
        --case "${case_name}"
        --solver "${solver}"
        --resolution "${resolution}"
        --max-steps "${MAX_STEPS}"
        --no-out
    )
    printf "Command: srun --ntasks=1"
    printf " %q" "${app[@]}"
    printf "\n"

    set +e
    if command -v /usr/bin/time >/dev/null 2>&1; then
        /usr/bin/time \
            -f "real_seconds=%e\nmax_rss_kb=%M" \
            -o "${temp_time}" \
            srun --ntasks=1 "${app[@]}" 2>&1 | tee "${temp_log}"
        run_status="${PIPESTATUS[0]}"
    else
        srun --ntasks=1 "${app[@]}" 2>&1 | tee "${temp_log}"
        run_status="${PIPESTATUS[0]}"
        printf "real_seconds=unknown\nmax_rss_kb=unknown\n" > "${temp_time}"
    fi
    set -e

    if [[ "${run_status}" -ne 0 ]]; then
        echo "[ERROR] Run failed with status ${run_status}."
        rm -f "${temp_log}" "${temp_time}"
        exit "${run_status}"
    fi

    local nx ny nz cells steps final_time gpu_seconds wall_seconds
    local process_real rss
    nx="$(awk -F ':' '/\[GPU3D\] nx/{gsub(/[ \t]/,"",$2);print $2;exit}' "${temp_log}")"
    ny="$(awk -F ':' '/\[GPU3D\] ny/{gsub(/[ \t]/,"",$2);print $2;exit}' "${temp_log}")"
    nz="$(awk -F ':' '/\[GPU3D\] nz/{gsub(/[ \t]/,"",$2);print $2;exit}' "${temp_log}")"
    cells="$(awk -F ':' '/\[GPU3D\] total_cells/{gsub(/[ \t]/,"",$2);print $2;exit}' "${temp_log}")"
    steps="$(awk -F '=' '/\[GPU3D\] Total steps/{gsub(/[ \t]/,"",$2);print $2;exit}' "${temp_log}")"
    final_time="$(awk -F '=' '/\[GPU3D\] Final time/{gsub(/[ \t]/,"",$2);print $2;exit}' "${temp_log}")"
    gpu_seconds="$(awk -F '=' '/\[GPU3D\] GPU elapsed/{gsub(/[ s\t]/,"",$2);print $2;exit}' "${temp_log}")"
    wall_seconds="$(awk -F '=' '/\[GPU3D\] Wall elapsed/{gsub(/[ s\t]/,"",$2);print $2;exit}' "${temp_log}")"
    process_real="$(awk -F '=' '/^real_seconds=/{print $2;exit}' "${temp_time}")"
    rss="$(awk -F '=' '/^max_rss_kb=/{print $2;exit}' "${temp_time}")"

    echo "gpu3d_baseline,${case_name},${solver},${nx},${ny},${nz},${cells},${steps},${final_time},${gpu_seconds},${wall_seconds},${process_real:-unknown},${rss:-unknown},${GIT_BRANCH},${GIT_COMMIT},${CUDA_ARCH_FLAG}" >> "${SUMMARY}"
    rm -f "${temp_log}" "${temp_time}"
}

for resolution in "${RESOLUTIONS[@]}"; do
    for case_name in "${CASES[@]}"; do
        for solver in "${SOLVERS[@]}"; do
            run_and_record "${case_name}" "${solver}" "${resolution}"
        done
    done
done

echo
echo "===== SUMMARY CSV ====="
cat "${SUMMARY}"
echo
echo "Saved summary: ${WORKDIR}/${SUMMARY}"
echo "End: $(date --iso-8601=seconds)"
