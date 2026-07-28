#!/bin/bash -l
#SBATCH -J hlld_baseline
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# CSD3 timing for the current branch's unmodified 2D CUDA HLLD code.
# It runs rotor and Orszag-Tang at n={1,2,4,8}. n=1/2 are repeated
# three times; n=4/8 are run once. Field output is disabled.
#
# Submit from the repository root:
#   mkdir -p logs
#   sbatch scripts/csd3_slurm/slurm_baseline_hlld_timing.sh
#
# Optional:
#   sbatch --export=ALL,SMALL_N_REPEATS=5 \
#     scripts/csd3_slurm/slurm_baseline_hlld_timing.sh
#
# Quick numerical step-count check before the complete sweep:
#   sbatch --export=ALL,SCALES_STR=1,SMALL_N_REPEATS=1 \
#     scripts/csd3_slurm/slurm_baseline_hlld_timing.sh

set -euo pipefail

JOB_ID="${SLURM_JOB_ID:-manual}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SUBMIT_DIR}}"
cd "${WORKDIR}"

if [[ ! -f Makefile ]]; then
    echo "[ERROR] Submit from the MHD repository root or set WORKDIR."
    exit 1
fi

# Slurm opens stdout/stderr before the script starts, so create logs before
# sbatch. The mkdir remains useful for manual execution.
mkdir -p logs

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

for tool in nvcc nvidia-smi make /usr/bin/time; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "[ERROR] Required tool is unavailable: ${tool}"
        exit 1
    fi
done

read -r -a CASES <<< "${CASES_STR:-orszag_tang rotor}"
read -r -a SCALES <<< "${SCALES_STR:-1 2 4 8}"
SOLVER="hlld"
SMALL_N_REPEATS="${SMALL_N_REPEATS:-3}"
LARGE_N_REPEATS="${LARGE_N_REPEATS:-1}"
MAKE_JOBS="${MAKE_JOBS:-8}"

if ! [[ "${SMALL_N_REPEATS}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${LARGE_N_REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] Repeat counts must be positive integers."
    exit 2
fi

GPU_NAME="$(
    nvidia-smi --query-gpu=name --format=csv,noheader |
    head -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
)"
COMPUTE_CAP="$(
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
    head -n 1 | tr -d '[:space:]' || true
)"
if [[ "${COMPUTE_CAP}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    CUDA_SM="${COMPUTE_CAP/./}"
else
    case "${GPU_NAME}" in
        *A100*|*A30*) CUDA_SM=80 ;;
        *A10*|*A40*) CUDA_SM=86 ;;
        *H100*|*H800*) CUDA_SM=90 ;;
        *)
            echo "[ERROR] Cannot determine CUDA architecture for ${GPU_NAME}."
            echo "[ERROR] Set CUDA_ARCH_FLAG explicitly."
            exit 1
            ;;
    esac
fi
CUDA_ARCH_FLAG="${CUDA_ARCH_FLAG:--arch=sm_${CUDA_SM}}"

GIT_BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_STATUS="$(git status --short 2>/dev/null || true)"
RESULT_DIR="${WORKDIR}/timing/baseline_hlld/${JOB_ID}/gpu"
BUILD_ROOT="${WORKDIR}/build/baseline_hlld/${JOB_ID}/gpu"
BIN_ROOT="${WORKDIR}/bin/baseline_hlld/${JOB_ID}/gpu"
RUN_DIR="${RESULT_DIR}/runs"
RAW_CSV="${RESULT_DIR}/gpu_hlld.csv"
MEAN_CSV="${RESULT_DIR}/gpu_hlld_means.csv"
METADATA="${RESULT_DIR}/metadata.txt"
BIN="${BIN_ROOT}/main_gpu"
mkdir -p "${RUN_DIR}" "${BUILD_ROOT}" "${BIN_ROOT}"

echo "===== CSD3 2D BASELINE HLLD TIMING ====="
echo "Job ID            : ${JOB_ID}"
echo "Host              : $(hostname)"
echo "Workdir           : ${WORKDIR}"
echo "Git branch        : ${GIT_BRANCH}"
echo "Git commit        : ${GIT_COMMIT}"
echo "GPU               : ${GPU_NAME}"
echo "CUDA architecture : ${CUDA_ARCH_FLAG}"
echo "Cases             : ${CASES[*]}"
echo "Scales            : ${SCALES[*]}"
echo "Small-n repeats   : ${SMALL_N_REPEATS}"
echo "Large-n repeats   : ${LARGE_N_REPEATS}"
echo "Result directory  : ${RESULT_DIR}"
echo
nvidia-smi
nvcc --version

{
    echo "job_id=${JOB_ID}"
    echo "hostname=$(hostname)"
    echo "start_utc=$(date --utc --iso-8601=seconds)"
    echo "workdir=${WORKDIR}"
    echo "git_branch=${GIT_BRANCH}"
    echo "git_commit=${GIT_COMMIT}"
    echo "git_status_begin"
    printf '%s\n' "${GIT_STATUS}"
    echo "git_status_end"
    echo "cases=${CASES[*]}"
    echo "solver=${SOLVER}"
    echo "scales=${SCALES[*]}"
    echo "small_n_repeats=${SMALL_N_REPEATS}"
    echo "large_n_repeats=${LARGE_N_REPEATS}"
    echo "build_variant=baseline_current_commit_no_extra_flags"
    echo "gpu_name=${GPU_NAME}"
    echo "compute_capability=${COMPUTE_CAP:-unknown}"
    echo "cuda_arch_flag=${CUDA_ARCH_FLAG}"
    echo "nvcc_extra_flags="
    echo "nvcc_begin"
    nvcc --version
    echo "nvcc_end"
    echo "nvidia_smi_begin"
    nvidia-smi
    echo "nvidia_smi_end"
} > "${METADATA}"

echo "===== BUILD CURRENT BRANCH BASELINE ====="
BUILD_START_NS="$(date +%s%N)"
make -j "${MAKE_JOBS}" gpu \
    BUILD_DIR="${BUILD_ROOT}" \
    BIN_DIR="${BIN_ROOT}" \
    CUDA_ARCH="${CUDA_ARCH_FLAG}" \
    NVCC_EXTRA_FLAGS=""
BUILD_END_NS="$(date +%s%N)"
awk -v start="${BUILD_START_NS}" -v end="${BUILD_END_NS}" \
    'BEGIN {printf "build_wall_seconds=%.9f\n", (end-start)/1.0e9}' \
    >> "${METADATA}"

if [[ ! -x "${BIN}" ]]; then
    echo "[ERROR] GPU executable was not produced: ${BIN}"
    exit 1
fi

echo "backend,case,solver,n,repeat,repeats_requested,nx,ny,total_cells,steps,app_elapsed_s,wall_seconds,user_seconds,sys_seconds,max_rss_kb,start_utc,end_utc,exit_status,hostname,git_branch,git_commit" \
    > "${RAW_CSV}"

FAILED_RUNS=0

run_once() {
    local case_name="$1"
    local n_scale="$2"
    local repeat="$3"
    local repeats_requested="$4"
    local stem="${RUN_DIR}/${case_name}_${SOLVER}_n${n_scale}_repeat_${repeat}"
    local console_file="${stem}.log"
    local time_file="${stem}.time"
    local start_utc end_utc start_ns end_ns wall_seconds status

    echo
    echo "============================================================"
    echo "RUN case=${case_name} solver=${SOLVER} n=${n_scale} repeat=${repeat}/${repeats_requested}"
    echo "============================================================"

    start_utc="$(date --utc --iso-8601=ns)"
    start_ns="$(date +%s%N)"
    set +e
    /usr/bin/time \
        -f $'user_seconds=%U\nsys_seconds=%S\nmax_rss_kb=%M\ntime_exit_status=%x' \
        -o "${time_file}" \
        "${BIN}" "${n_scale}" \
        --case "${case_name}" --solver "${SOLVER}" --no-out \
        2>&1 | tee "${console_file}"
    status="${PIPESTATUS[0]}"
    set -e
    end_ns="$(date +%s%N)"
    end_utc="$(date --utc --iso-8601=ns)"
    wall_seconds="$(
        awk -v start="${start_ns}" -v end="${end_ns}" \
            'BEGIN {printf "%.9f", (end-start)/1.0e9}'
    )"

    local nx ny cells steps app_elapsed user_seconds sys_seconds max_rss
    nx="$(awk -F: '/^\[GPU\] nx:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "${console_file}")"
    ny="$(awk -F: '/^\[GPU\] ny:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "${console_file}")"
    cells="$(awk -F: '/^\[GPU\] total_cells:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "${console_file}")"
    steps="$(awk -F= '/^\[GPU\] Total steps=/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "${console_file}")"
    app_elapsed="$(awk '/^[[:space:]]*Elapsed[[:space:]]*:/ {print $3; exit}' "${console_file}")"
    user_seconds="$(awk -F= '$1=="user_seconds" {print $2; exit}' "${time_file}")"
    sys_seconds="$(awk -F= '$1=="sys_seconds" {print $2; exit}' "${time_file}")"
    max_rss="$(awk -F= '$1=="max_rss_kb" {print $2; exit}' "${time_file}")"

    printf '%s\n' \
        "gpu,${case_name},${SOLVER},${n_scale},${repeat},${repeats_requested},${nx:-unknown},${ny:-unknown},${cells:-unknown},${steps:-unknown},${app_elapsed:-unknown},${wall_seconds},${user_seconds:-unknown},${sys_seconds:-unknown},${max_rss:-unknown},${start_utc},${end_utc},${status},$(hostname),${GIT_BRANCH},${GIT_COMMIT}" \
        >> "${RAW_CSV}"

    echo "[RESULT] status=${status} app_elapsed_s=${app_elapsed:-unknown} wall_seconds=${wall_seconds}"
    if [[ "${status}" -ne 0 ]]; then
        FAILED_RUNS=$((FAILED_RUNS + 1))
    fi
}

unset CUDA_LAUNCH_BLOCKING
for n_scale in "${SCALES[@]}"; do
    if (( n_scale <= 2 )); then
        repeats="${SMALL_N_REPEATS}"
    else
        repeats="${LARGE_N_REPEATS}"
    fi
    for case_name in "${CASES[@]}"; do
        for ((repeat = 1; repeat <= repeats; ++repeat)); do
            run_once "${case_name}" "${n_scale}" "${repeat}" "${repeats}"
        done
    done
done

awk -F, '
    BEGIN {
        OFS=","
        print "backend,case,solver,n,runs_successful,runs_requested,mean_app_elapsed_s,mean_wall_seconds"
    }
    NR == 1 { next }
    {
        key=$1 SUBSEP $2 SUBSEP $3 SUBSEP $4
        if (!(key in seen)) {
            seen[key]=++count
            keys[count]=key
            backend[key]=$1
            case_name[key]=$2
            solver[key]=$3
            scale[key]=$4
            requested[key]=$6
        }
        if ($18 == 0) {
            successful[key]++
            wall_sum[key]+=$12
            if ($11 != "unknown") {
                app_count[key]++
                app_sum[key]+=$11
            }
        }
    }
    END {
        for (i=1; i<=count; ++i) {
            key=keys[i]
            app_mean=(app_count[key] > 0) \
                ? sprintf("%.9f", app_sum[key]/app_count[key]) : "unknown"
            wall_mean=(successful[key] > 0) \
                ? sprintf("%.9f", wall_sum[key]/successful[key]) : "unknown"
            print backend[key],case_name[key],solver[key],scale[key], \
                  successful[key]+0,requested[key],app_mean,wall_mean
        }
    }
' "${RAW_CSV}" > "${MEAN_CSV}"

{
    echo "end_utc=$(date --utc --iso-8601=seconds)"
    echo "failed_runs=${FAILED_RUNS}"
} >> "${METADATA}"

echo
echo "===== RAW TIMINGS ====="
cat "${RAW_CSV}"
echo
echo "===== MEAN TIMINGS ====="
cat "${MEAN_CSV}"
echo
echo "Raw CSV  : ${RAW_CSV}"
echo "Mean CSV : ${MEAN_CSV}"
echo "Metadata : ${METADATA}"
echo "Failures : ${FAILED_RUNS}"

if [[ "${FAILED_RUNS}" -ne 0 ]]; then
    exit 1
fi
