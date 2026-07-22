#!/bin/bash -l
#SBATCH -J mhd_n1_parity
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

# CSD3 numerical-parity job.  It builds CPU, GPU and pure-MPI executables
# from the same checkout, runs n=1 full-field outputs for Orszag-Tang, rotor
# and Brio-Wu, then compares every backend pair cell-by-cell.
#
# Default: HLLD only.  Override without editing the file, for example:
#   SOLVERS_STR="hll hllc hlld force" RANKS=4 \
#     sbatch scripts/csd3_slurm/slurm_n1_output_parity.sh
#
# Submit from the repository root after creating the Slurm log directory:
#   mkdir -p logs
#   sbatch scripts/csd3_slurm/slurm_n1_output_parity.sh

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"

CASES=(orszag_tang rotor brio_wu)
read -r -a SOLVERS <<< "${SOLVERS_STR:-hlld}"
RANKS="${RANKS:-4}"
ATOL="${ATOL:-1e-12}"
RTOL="${RTOL:-1e-12}"

if ! [[ "${RANKS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] RANKS must be a positive integer; got '${RANKS}'."
    exit 2
fi
if [ "${#SOLVERS[@]}" -eq 0 ]; then
    echo "[ERROR] SOLVERS_STR did not contain a solver."
    exit 2
fi
for solver in "${SOLVERS[@]}"; do
    case "${solver}" in
        hll|hllc|hlld|force) ;;
        *) echo "[ERROR] Unknown solver '${solver}'."; exit 2 ;;
    esac
done

cd "${WORKDIR}"

OUT_ROOT="${WORKDIR}/outputs/csd3_n1_parity/${SLURM_JOB_ID}"
BUILD_ROOT="${WORKDIR}/build/csd3_n1_parity/${SLURM_JOB_ID}"
BIN_ROOT="${WORKDIR}/bin/csd3_n1_parity/${SLURM_JOB_ID}"
RUN_LOG_ROOT="${OUT_ROOT}/run_logs"
mkdir -p logs "${OUT_ROOT}" "${BUILD_ROOT}" "${BIN_ROOT}" "${RUN_LOG_ROOT}"

echo "===== MODULE SETUP ====="
if ! command -v module >/dev/null 2>&1; then
    for init_file in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/local/Modules/init/bash; do
        if [ -f "${init_file}" ]; then
            # shellcheck disable=SC1090
            source "${init_file}"
            break
        fi
    done
fi
if ! command -v module >/dev/null 2>&1; then
    echo "[ERROR] The module command is unavailable."
    exit 1
fi

module purge
module load rhel8/default-amp
module list 2>&1 || true

for required in g++ nvcc mpicxx mpirun python3 nvidia-smi; do
    if ! command -v "${required}" >/dev/null 2>&1; then
        echo "[ERROR] ${required} is unavailable after loading rhel8/default-amp."
        exit 1
    fi
done

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
if [[ "${COMPUTE_CAP}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    CUDA_SM="${COMPUTE_CAP/./}"
else
    case "${GPU_NAME}" in
        *A100*|*A30*) CUDA_SM=80 ;;
        *A10*|*A40*) CUDA_SM=86 ;;
        *H100*|*H800*) CUDA_SM=90 ;;
        *) echo "[ERROR] Cannot determine CUDA architecture for '${GPU_NAME}'. Set CUDA_ARCH_FLAG."; exit 1 ;;
    esac
fi
CUDA_ARCH_FLAG="${CUDA_ARCH_FLAG:--arch=sm_${CUDA_SM}}"

export OMP_NUM_THREADS="${OMP_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
unset CUDA_LAUNCH_BLOCKING

GIT_BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_STATUS="$(git status --short 2>/dev/null || true)"

METADATA="${OUT_ROOT}/metadata.txt"
{
    echo "job_id=${SLURM_JOB_ID}"
    echo "host=$(hostname)"
    echo "start_utc=$(date --utc --iso-8601=seconds)"
    echo "workdir=${WORKDIR}"
    echo "cases=${CASES[*]}"
    echo "solvers=${SOLVERS[*]}"
    echo "n=1"
    echo "mpi_ranks=${RANKS}"
    echo "omp_threads=${OMP_NUM_THREADS}"
    echo "atol=${ATOL}"
    echo "rtol=${RTOL}"
    echo "gpu_name=${GPU_NAME}"
    echo "compute_capability=${COMPUTE_CAP}"
    echo "cuda_arch=${CUDA_ARCH_FLAG}"
    echo "git_branch=${GIT_BRANCH}"
    echo "git_commit=${GIT_COMMIT}"
    echo "git_status_begin"
    printf '%s\n' "${GIT_STATUS}"
    echo "git_status_end"
    echo "gxx_begin"
    g++ --version
    echo "gxx_end"
    echo "mpicxx_begin"
    mpicxx --version
    echo "mpicxx_end"
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

echo "===== BUILD CPU / GPU / MPI ====="
make -j "${MAKE_JOBS:-8}" cpu mpi gpu \
    BUILD_DIR="${BUILD_ROOT}" \
    BIN_DIR="${BIN_ROOT}" \
    CUDA_ARCH="${CUDA_ARCH_FLAG}"

CPU_BIN="${BIN_ROOT}/main_cpu"
GPU_BIN="${BIN_ROOT}/main_gpu"
MPI_BIN="${BIN_ROOT}/main_mpi"
for binary in "${CPU_BIN}" "${GPU_BIN}" "${MPI_BIN}"; do
    if [ ! -x "${binary}" ]; then
        echo "[ERROR] Expected binary was not produced: ${binary}"
        exit 1
    fi
done

FAILED_RUNS=0

run_logged() {
    local label="$1"
    shift
    local log_file="${RUN_LOG_ROOT}/${label}.log"
    local time_file="${RUN_LOG_ROOT}/${label}.time"
    local status

    echo ""
    echo "============================================================"
    echo "RUN ${label}"
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
    echo "============================================================"

    set +e
    /usr/bin/time \
        -f $'real_seconds=%e\nuser_seconds=%U\nsys_seconds=%S\nmax_rss_kb=%M\nexit_status=%x' \
        -o "${time_file}" \
        "$@" 2>&1 | tee "${log_file}"
    status="${PIPESTATUS[0]}"
    set -e

    if [ "${status}" -ne 0 ]; then
        echo "[ERROR] ${label} failed with status ${status}; see ${log_file}."
        FAILED_RUNS=$((FAILED_RUNS + 1))
    fi
}

for case_name in "${CASES[@]}"; do
    for solver in "${SOLVERS[@]}"; do
        cpu_out="${OUT_ROOT}/cpu_${case_name}_${solver}"
        gpu_out="${OUT_ROOT}/gpu_${case_name}_${solver}_n1"
        mpi_out="${OUT_ROOT}/mpi_${case_name}_${solver}_n1"
        mkdir -p "${cpu_out}" "${gpu_out}" "${mpi_out}"

        run_logged "cpu_${case_name}_${solver}" \
            "${CPU_BIN}" "${case_name}" --n 1 --solver "${solver}" --out "${cpu_out}"

        run_logged "gpu_${case_name}_${solver}" \
            "${GPU_BIN}" 1 --case "${case_name}" --solver "${solver}" --out "${gpu_out}"

        run_logged "mpi_${case_name}_${solver}" \
            mpirun -np "${RANKS}" --host "$(hostname):${RANKS}" \
                --oversubscribe --bind-to core --map-by core \
                "${MPI_BIN}" "${case_name}" --n 1 --solver "${solver}" \
                --out "${mpi_out}" --output
    done
done

echo ""
echo "===== PAIRWISE FIELD COMPARISON ====="
CASES_CSV="$(IFS=,; echo "${CASES[*]}")"
SOLVERS_CSV="$(IFS=,; echo "${SOLVERS[*]}")"
set +e
python3 scripts/compare_multi_arch.py "${OUT_ROOT}" \
    --cases "${CASES_CSV}" \
    --solvers "${SOLVERS_CSV}" \
    --archs cpu,gpu,mpi \
    --tol "${ATOL}" \
    --rtol "${RTOL}" \
    --verbose 2>&1 | tee "${OUT_ROOT}/comparison.log"
COMPARE_STATUS="${PIPESTATUS[0]}"
set -e

{
    echo "end_utc=$(date --utc --iso-8601=seconds)"
    echo "failed_runs=${FAILED_RUNS}"
    echo "comparison_exit_status=${COMPARE_STATUS}"
} >> "${METADATA}"

ARCHIVE="${OUT_ROOT}.tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${OUT_ROOT}")" "$(basename "${OUT_ROOT}")"

echo ""
echo "===== FINAL SUMMARY ====="
echo "Full outputs       : ${OUT_ROOT}"
echo "Summary comparison : ${OUT_ROOT}/comparison_report.csv"
echo "Per-field report   : ${OUT_ROOT}/comparison_report_full.csv"
echo "Comparison log     : ${OUT_ROOT}/comparison.log"
echo "Download archive   : ${ARCHIVE}"
echo "Failed runs        : ${FAILED_RUNS}"
echo "Comparison status  : ${COMPARE_STATUS}"

if [ "${FAILED_RUNS}" -ne 0 ] || [ "${COMPARE_STATUS}" -ne 0 ]; then
    exit 1
fi
