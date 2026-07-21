#!/bin/bash -l
#SBATCH -J slurm_gpu
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# CSD3 Ampere GPU timing sweep.
# Before submission, ensure the Slurm log directory exists:
#   mkdir -p logs

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
MAKE_CLEAN="${MAKE_CLEAN:-1}"

cd "${WORKDIR}"
mkdir -p logs timing/gpu_timing outputs

echo "===== MODULE SETUP ====="

if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -f /usr/share/Modules/init/bash ]; then
        source /usr/share/Modules/init/bash
    elif [ -f /usr/local/Modules/init/bash ]; then
        source /usr/local/Modules/init/bash
    fi
fi

if ! command -v module >/dev/null 2>&1; then
    echo "[ERROR] The 'module' command is unavailable."
    exit 1
fi

module purge

if ! module load rhel8/default-amp; then
    echo "[ERROR] Failed to load rhel8/default-amp."
    echo "[INFO] Relevant available modules:"
    module avail 2>&1 | grep -Ei 'default-amp|cuda|nvhpc' || true
    exit 1
fi

echo "[INFO] Loaded modules:"
module list 2>&1 || true

if ! command -v nvcc >/dev/null 2>&1; then
    echo "[ERROR] nvcc is unavailable after loading rhel8/default-amp."
    module avail 2>&1 | grep -Ei 'cuda|nvhpc' || true
    exit 1
fi

echo "[INFO] nvcc: $(command -v nvcc)"
nvcc --version

read -r -a CASES   <<< "${CASES_STR:-orszag_tang rotor}"
read -r -a SOLVERS <<< "${SOLVERS_STR:-hll hllc hlld force}"
read -r -a SCALES  <<< "${SCALES_STR:-1 2 4 8}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
unset CUDA_LAUNCH_BLOCKING

GIT_BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_COMMIT_MSG="$(git log -1 --pretty=%s 2>/dev/null || echo unknown)"

GPU_NAME="$(
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null |
    head -n 1 |
    sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
)"

COMPUTE_CAP="$(
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
    head -n 1 |
    tr -d '[:space:]' || true
)"

if [[ "${COMPUTE_CAP}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    CUDA_SM="${COMPUTE_CAP/./}"
else
    case "${GPU_NAME}" in
        *A100*|*A30*) CUDA_SM="80" ;;
        *A10*|*A40*) CUDA_SM="86" ;;
        *H100*|*H800*) CUDA_SM="90" ;;
        *)
            echo "[ERROR] Could not determine CUDA architecture for GPU: ${GPU_NAME:-unknown}"
            echo "[ERROR] Set it manually, for example:"
            echo "        CUDA_ARCH_FLAG='-arch=sm_80' sbatch $0"
            exit 1
            ;;
    esac
fi

CUDA_ARCH_FLAG="${CUDA_ARCH_FLAG:--arch=sm_${CUDA_SM}}"

echo ""
echo "===== JOB INFO ====="
echo "Job ID               : ${SLURM_JOB_ID}"
echo "Host                 : $(hostname)"
echo "Start                : $(date --iso-8601=seconds)"
echo "Workdir              : ${WORKDIR}"
echo "Partition            : ${SLURM_JOB_PARTITION:-unknown}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU name             : ${GPU_NAME:-unknown}"
echo "Compute capability   : ${COMPUTE_CAP:-unknown}"
echo "Build CUDA arch      : ${CUDA_ARCH_FLAG}"
echo "Git branch           : ${GIT_BRANCH}"
echo "Git commit           : ${GIT_COMMIT}"
echo "Cases                : ${CASES[*]}"
echo "Solvers              : ${SOLVERS[*]}"
echo "Scales               : ${SCALES[*]}"
echo "OMP threads          : ${OMP_NUM_THREADS}"

echo ""
echo "===== GPU INFO ====="
nvidia-smi || true

echo ""
echo "===== BUILD ====="

if [ "${MAKE_CLEAN}" = "1" ]; then
    make clean
fi

make gpu CUDA_ARCH="${CUDA_ARCH_FLAG}"

BIN="./bin/main_gpu"

if [ ! -x "${BIN}" ]; then
    echo "[ERROR] ${BIN} was not produced or is not executable."
    exit 1
fi

echo "[INFO] Built binary:"
ls -lh "${BIN}"

SUMMARY="timing/gpu_timing/gpu_${SLURM_JOB_ID}.csv"

{
    echo "# commit_message: ${GIT_COMMIT_MSG}"
    echo "arch,case,solver,n,nx,ny,total_cells,total_steps,real_seconds,user_seconds,sys_seconds,max_rss_kb,git_branch,git_commit,cuda_arch"
} > "${SUMMARY}"

run_and_record() {
    local case_name="$1"
    local solver="$2"
    local n_scale="$3"

    local temp_log
    local temp_time
    local run_status

    temp_log="$(mktemp)"
    temp_time="$(mktemp)"

    echo ""
    echo "============================================================"
    echo "GPU RUN: case=${case_name}, solver=${solver}, n=${n_scale}"
    echo "============================================================"

    local app=(
        "${BIN}"
        "${n_scale}"
        --case "${case_name}"
        --solver "${solver}"
        --no-out
    )

    printf "Command:"
    printf " %q" "${app[@]}"
    printf "\n"

    set +e
    /usr/bin/time \
        -f "real_seconds=%e
user_seconds=%U
sys_seconds=%S
max_rss_kb=%M" \
        -o "${temp_time}" \
        "${app[@]}" \
        2>&1 | tee "${temp_log}"

    run_status="${PIPESTATUS[0]}"
    set -e

    if [ "${run_status}" -ne 0 ]; then
        echo "[ERROR] Run failed with status ${run_status}."
        echo "[ERROR] Configuration: case=${case_name}, solver=${solver}, n=${n_scale}"
        rm -f "${temp_log}" "${temp_time}"
        exit "${run_status}"
    fi

    local nx ny cells steps real user sys rss

    nx="$(awk -F ':' '/\[GPU\] nx/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "${temp_log}")"
    ny="$(awk -F ':' '/\[GPU\] ny/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "${temp_log}")"
    cells="$(awk -F ':' '/\[GPU\] total_cells/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "${temp_log}")"
    steps="$(awk -F '=' '/\[GPU\] Total steps/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "${temp_log}")"

    real="$(awk -F '=' '/^real_seconds=/{print $2; exit}' "${temp_time}")"
    user="$(awk -F '=' '/^user_seconds=/{print $2; exit}' "${temp_time}")"
    sys="$(awk -F '=' '/^sys_seconds=/{print $2; exit}' "${temp_time}")"
    rss="$(awk -F '=' '/^max_rss_kb=/{print $2; exit}' "${temp_time}")"

    nx="${nx:-unknown}"
    ny="${ny:-unknown}"
    cells="${cells:-unknown}"
    steps="${steps:-unknown}"
    real="${real:-unknown}"
    user="${user:-unknown}"
    sys="${sys:-unknown}"
    rss="${rss:-unknown}"

    echo "------------------------------------------------------------"
    echo "[TIMING] Real=${real}s User=${user}s Sys=${sys}s MaxRSS=${rss}KB"
    echo "------------------------------------------------------------"

    echo "gpu,${case_name},${solver},${n_scale},${nx},${ny},${cells},${steps},${real},${user},${sys},${rss},${GIT_BRANCH},${GIT_COMMIT},${CUDA_ARCH_FLAG}" >> "${SUMMARY}"

    rm -f "${temp_log}" "${temp_time}"
}

for N in "${SCALES[@]}"; do
    for CASE_NAME in "${CASES[@]}"; do
        for SOLVER_NAME in "${SOLVERS[@]}"; do
            run_and_record "${CASE_NAME}" "${SOLVER_NAME}" "${N}"
        done
    done
done

echo ""
echo "===== SUMMARY CSV ====="
cat "${SUMMARY}"

echo ""
echo "Saved summary: ${SUMMARY}"
echo "End time     : $(date --iso-8601=seconds)"
echo "===== END ====="