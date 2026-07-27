#!/bin/bash -l
#SBATCH -J mhd_gpu_n1_out
#SBATCH -A hansirui
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# DGX GPU full-output sweep for plotting and validation:
#   cases   = Orszag-Tang, rotor, Brio-Wu
#   solvers = HLL, HLLC, HLLD, FORCE
#   n       = 1
#
# This is the DGX counterpart of:
#   scripts/csd3_slurm/slurm_gpu_n1_all_outputs.sh
#
# The executable is built once in a job-private directory, then reused for
# all 12 runs. Every run writes complete field snapshots to its own directory.
#
# Submit from the MHD repository root after making sure logs/ exists:
#   mkdir -p logs
#   sbatch scripts/dgx_slurm/slurm_gpu_n1_all_outputs.sh
#
# Optional overrides:
#   CASES_STR="orszag_tang rotor brio_wu" \
#   SOLVERS_STR="hll hllc hlld force" \
#   sbatch scripts/dgx_slurm/slurm_gpu_n1_all_outputs.sh
#
# To reproduce a non-default GPU build:
#   NVCC_EXTRA_FLAGS="-DMHD_HLLD_CANONICALIZE_Y=1 ..." \
#   sbatch scripts/dgx_slurm/slurm_gpu_n1_all_outputs.sh

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
WORKDIR="${WORKDIR:-/aifs4su/hansirui_2nd/harry/MHD_gpu_acc}"
read -r -a CASES <<< "${CASES_STR:-orszag_tang rotor brio_wu}"
read -r -a SOLVERS <<< "${SOLVERS_STR:-hll hllc hlld force}"

if [ ! -f "${WORKDIR}/Makefile" ] || [ ! -f "${WORKDIR}/scripts/gpu/main_gpu.cu" ]; then
    echo "[ERROR] WORKDIR is not the MHD repository root: ${WORKDIR}"
    echo "[ERROR] Set WORKDIR to the repository root before submitting."
    exit 1
fi
if [ "${#CASES[@]}" -eq 0 ] || [ "${#SOLVERS[@]}" -eq 0 ]; then
    echo "[ERROR] CASES_STR and SOLVERS_STR must not be empty."
    exit 2
fi
for case_name in "${CASES[@]}"; do
    case "${case_name}" in
        orszag_tang|rotor|brio_wu) ;;
        *) echo "[ERROR] Unknown MHD case '${case_name}'."; exit 2 ;;
    esac
done
for solver in "${SOLVERS[@]}"; do
    case "${solver}" in
        hll|hllc|hlld|force) ;;
        *) echo "[ERROR] Unknown solver '${solver}'."; exit 2 ;;
    esac
done

cd "${WORKDIR}"

OUT_ROOT="${WORKDIR}/outputs/dgx_gpu_n1_plots/${SLURM_JOB_ID}"
BUILD_ROOT="${WORKDIR}/build/dgx_gpu_n1_plots/${SLURM_JOB_ID}"
BIN_ROOT="${WORKDIR}/bin/dgx_gpu_n1_plots/${SLURM_JOB_ID}"
RUN_LOG_ROOT="${OUT_ROOT}/run_logs"
MANIFEST="${OUT_ROOT}/output_manifest.csv"
METADATA="${OUT_ROOT}/metadata.txt"
mkdir -p logs "${OUT_ROOT}" "${BUILD_ROOT}" "${BIN_ROOT}" "${RUN_LOG_ROOT}"

echo "===== MODULE SETUP ====="
module load cuda/12.2
module list 2>&1 || true

for required in nvcc nvidia-smi make; do
    if ! command -v "${required}" >/dev/null 2>&1; then
        echo "[ERROR] ${required} is unavailable after loading cuda/12.2."
        exit 1
    fi
done

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
if [[ "${COMPUTE_CAP}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    CUDA_SM="${COMPUTE_CAP/./}"
else
    case "${GPU_NAME}" in
        *H100*|*H800*) CUDA_SM=90 ;;
        *)
            echo "[ERROR] Cannot determine CUDA architecture for '${GPU_NAME}'."
            echo "[ERROR] Set CUDA_ARCH_FLAG explicitly, for example --arch=sm_90."
            exit 1
            ;;
    esac
fi
CUDA_ARCH_FLAG="${CUDA_ARCH_FLAG:---arch=sm_${CUDA_SM}}"
NVCC_EXTRA_FLAGS="${NVCC_EXTRA_FLAGS:-}"

GIT_BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_STATUS="$(git status --short 2>/dev/null || true)"

{
    echo "job_id=${SLURM_JOB_ID}"
    echo "host=$(hostname)"
    echo "start_utc=$(date --utc --iso-8601=seconds)"
    echo "workdir=${WORKDIR}"
    echo "output_root=${OUT_ROOT}"
    echo "cases=${CASES[*]}"
    echo "solvers=${SOLVERS[*]}"
    echo "n=1"
    echo "gpu_name=${GPU_NAME}"
    echo "compute_capability=${COMPUTE_CAP}"
    echo "cuda_arch_flag=${CUDA_ARCH_FLAG}"
    echo "nvcc_extra_flags=${NVCC_EXTRA_FLAGS}"
    echo "git_branch=${GIT_BRANCH}"
    echo "git_commit=${GIT_COMMIT}"
    echo "git_status_begin"
    printf '%s\n' "${GIT_STATUS}"
    echo "git_status_end"
    echo "nvcc_begin"
    nvcc --version
    echo "nvcc_end"
    echo "modules_begin"
    module list 2>&1 || true
    echo "modules_end"
    echo "nvidia_smi_begin"
    nvidia-smi
    echo "nvidia_smi_end"
} > "${METADATA}"

echo "===== JOB CONFIGURATION ====="
cat "${METADATA}"

echo "===== BUILD GPU EXECUTABLE ====="
make -j "${MAKE_JOBS:-8}" gpu \
    BUILD_DIR="${BUILD_ROOT}" \
    BIN_DIR="${BIN_ROOT}" \
    CUDA_ARCH="${CUDA_ARCH_FLAG}" \
    NVCC_EXTRA_FLAGS="${NVCC_EXTRA_FLAGS}"

GPU_BIN="${BIN_ROOT}/main_gpu"
if [ ! -x "${GPU_BIN}" ]; then
    echo "[ERROR] GPU binary was not produced: ${GPU_BIN}"
    exit 1
fi

printf '%s\n' "case,solver,n,output_dir,csv_files,wall_seconds,exit_status" > "${MANIFEST}"
FAILED_RUNS=0

for case_name in "${CASES[@]}"; do
    for solver in "${SOLVERS[@]}"; do
        label="gpu_${case_name}_${solver}_n1"
        output_dir="${OUT_ROOT}/${label}"
        log_file="${RUN_LOG_ROOT}/${label}.log"
        time_file="${RUN_LOG_ROOT}/${label}.time"
        mkdir -p "${output_dir}"

        echo ""
        echo "============================================================"
        echo "RUN case=${case_name} solver=${solver} n=1"
        echo "Output: ${output_dir}"
        echo "============================================================"

        set +e
        /usr/bin/time \
            -f $'wall_seconds=%e\nuser_seconds=%U\nsys_seconds=%S\nmax_rss_kb=%M\nexit_status=%x' \
            -o "${time_file}" \
            "${GPU_BIN}" 1 \
                --case "${case_name}" \
                --solver "${solver}" \
                --out "${output_dir}" \
            2>&1 | tee "${log_file}"
        status="${PIPESTATUS[0]}"
        set -e

        wall_seconds="$(awk -F= '$1 == "wall_seconds" {print $2}' "${time_file}")"
        csv_files="$(find "${output_dir}" -maxdepth 1 -type f -name '*.csv' | wc -l | tr -d '[:space:]')"
        relative_output="outputs/dgx_gpu_n1_plots/${SLURM_JOB_ID}/${label}"
        printf '%s,%s,1,%s,%s,%s,%s\n' \
            "${case_name}" "${solver}" "${relative_output}" \
            "${csv_files}" "${wall_seconds:-unknown}" "${status}" >> "${MANIFEST}"

        if [ "${status}" -ne 0 ]; then
            echo "[ERROR] ${label} failed with status ${status}; see ${log_file}."
            FAILED_RUNS=$((FAILED_RUNS + 1))
        elif [ "${csv_files}" -eq 0 ]; then
            echo "[ERROR] ${label} completed but produced no CSV files."
            FAILED_RUNS=$((FAILED_RUNS + 1))
        else
            echo "[OK] ${label}: ${csv_files} CSV files"
        fi
    done
done

{
    echo "end_utc=$(date --utc --iso-8601=seconds)"
    echo "failed_runs=${FAILED_RUNS}"
} >> "${METADATA}"

ARCHIVE="${OUT_ROOT}.tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${OUT_ROOT}")" "$(basename "${OUT_ROOT}")"

echo ""
echo "===== FINAL SUMMARY ====="
echo "Full outputs : ${OUT_ROOT}"
echo "Manifest     : ${MANIFEST}"
echo "Run logs     : ${RUN_LOG_ROOT}"
echo "Archive      : ${ARCHIVE}"
echo "Failed runs  : ${FAILED_RUNS}"

if [ "${FAILED_RUNS}" -ne 0 ]; then
    exit 1
fi
