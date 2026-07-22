#!/bin/bash -l
#SBATCH -J sys_eq_gpu_nsys
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Nsight Systems timeline profiling for the GPU solver on CSD3.
# The default run matches the current Orszag-Tang/HLLD/n=8 experiment and
# uses a bounded number of steps so the timeline remains small enough to
# inspect.  Override settings before sbatch when needed, for example:
#
#   WARMUP_STEPS=20 BENCHMARK_STEPS=100 \
#     sbatch scripts/csd3_slurm/profile_gpu_systems.sh

set -euo pipefail

N="${N:-8}"
CASE="${CASE:-orszag_tang}"
SOLVER="${SOLVER:-hlld}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-50}"
PROFILE_NVCC_FLAGS="${PROFILE_NVCC_FLAGS:--lineinfo -Xptxas=-v}"
ADVANCE_X_MIN_BLOCKS_PER_SM="${ADVANCE_X_MIN_BLOCKS_PER_SM:-3}"
ADVANCE_Y_MIN_BLOCKS_PER_SM="${ADVANCE_Y_MIN_BLOCKS_PER_SM:-0}"
HLLD_CANONICALIZE_Y="${HLLD_CANONICALIZE_Y:-0}"
MAKE_CLEAN="${MAKE_CLEAN:-1}"

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"

cd "${WORKDIR}"
mkdir -p logs profiling

for value_name in N WARMUP_STEPS BENCHMARK_STEPS \
                  ADVANCE_X_MIN_BLOCKS_PER_SM ADVANCE_Y_MIN_BLOCKS_PER_SM; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${value_name} must be a non-negative integer."
        exit 2
    fi
done

if [[ "${HLLD_CANONICALIZE_Y}" != "0" && "${HLLD_CANONICALIZE_Y}" != "1" ]]; then
    echo "[ERROR] HLLD_CANONICALIZE_Y must be 0 or 1."
    exit 2
fi

if [ "${N}" -eq 0 ] || [ "${BENCHMARK_STEPS}" -eq 0 ]; then
    echo "[ERROR] N and BENCHMARK_STEPS must be positive."
    exit 2
fi

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

if command -v module >/dev/null 2>&1; then
    module purge
    module load rhel8/default-amp
fi

if ! command -v nsys >/dev/null 2>&1; then
    echo "[ERROR] nsys was not found after loading rhel8/default-amp."
    module spider nsight 2>&1 || true
    module spider cuda 2>&1 || true
    exit 127
fi

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
unset CUDA_LAUNCH_BLOCKING

PROFILE_DIR="${WORKDIR}/profiling/${SLURM_JOB_ID}_${CASE}_${SOLVER}_n${N}_nsys"
mkdir -p "${PROFILE_DIR}"

BASE="${PROFILE_DIR}/nsys_${CASE}_${SOLVER}_n${N}"
REPORT="${BASE}.nsys-rep"
SQLITE_REPORT="${BASE}.sqlite"
STATS_FILE="${BASE}_stats.txt"
CONSOLE_LOG="${BASE}_console.log"
METADATA_FILE="${PROFILE_DIR}/metadata.txt"

{
    echo "date=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "workdir=${WORKDIR}"
    echo "N=${N}"
    echo "case=${CASE}"
    echo "solver=${SOLVER}"
    echo "warmup_steps=${WARMUP_STEPS}"
    echo "benchmark_steps=${BENCHMARK_STEPS}"
    echo "advance_x_min_blocks_per_sm=${ADVANCE_X_MIN_BLOCKS_PER_SM}"
    echo "advance_y_min_blocks_per_sm=${ADVANCE_Y_MIN_BLOCKS_PER_SM}"
    echo "hlld_canonicalize_y=${HLLD_CANONICALIZE_Y}"
    echo "git_branch=$(git branch --show-current 2>/dev/null || echo unknown)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "nsys=$(nsys --version 2>&1 | tr '\n' ' ')"
    nvidia-smi --query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem \
        --format=csv,noheader 2>/dev/null || true
} | tee "${METADATA_FILE}"

echo "===== BUILD ====="
BUILD_NVCC_FLAGS="${PROFILE_NVCC_FLAGS}"
BUILD_NVCC_FLAGS+=" -DMHD_ADVANCE_X_MIN_BLOCKS_PER_SM=${ADVANCE_X_MIN_BLOCKS_PER_SM}"
BUILD_NVCC_FLAGS+=" -DMHD_ADVANCE_Y_MIN_BLOCKS_PER_SM=${ADVANCE_Y_MIN_BLOCKS_PER_SM}"
BUILD_NVCC_FLAGS+=" -DMHD_HLLD_CANONICALIZE_Y=${HLLD_CANONICALIZE_Y}"

if [ "${MAKE_CLEAN}" = "1" ]; then
    make clean
fi
make gpu NVCC_EXTRA_FLAGS="${BUILD_NVCC_FLAGS}" \
    2>&1 | tee "${PROFILE_DIR}/build.log"

APP=(
    ./bin/main_gpu
    "${N}"
    --case "${CASE}"
    --solver "${SOLVER}"
    --no-out
    --warmup-steps "${WARMUP_STEPS}"
    --benchmark-steps "${BENCHMARK_STEPS}"
)

printf "[INFO] Application command:"
printf " %q" "${APP[@]}"
printf "\n"

echo "===== NSIGHT SYSTEMS PROFILE ====="
set +e
nsys profile \
    --force-overwrite=true \
    --trace=cuda,osrt,nvtx \
    --sample=none \
    --output="${BASE}" \
    "${APP[@]}" 2>&1 | tee "${CONSOLE_LOG}"
STATUS=${PIPESTATUS[0]}
set -e

if [ "${STATUS}" -ne 0 ]; then
    echo "[ERROR] nsys profile failed with status ${STATUS}."
    exit "${STATUS}"
fi

if [ ! -f "${REPORT}" ]; then
    # Nsight Systems bundled with CUDA 11.x may use .qdrep, or may leave the
    # intermediate .qdstrm when its host-side importer is not on PATH.
    if [ -f "${BASE}.qdrep" ]; then
        REPORT="${BASE}.qdrep"
    elif [ -f "${BASE}.qdstrm" ]; then
        RAW_REPORT="${BASE}.qdstrm"
        IMPORTER="$(command -v QdstrmImporter 2>/dev/null || true)"

        if [ -z "${IMPORTER}" ]; then
            NSYS_BIN="$(readlink -f "$(command -v nsys)")"
            NSYS_SEARCH_ROOT="$(dirname "$(dirname "${NSYS_BIN}")")"
            IMPORTER="$(find "${NSYS_SEARCH_ROOT}" -type f \
                -name QdstrmImporter -perm -u+x -print -quit 2>/dev/null || true)"
        fi

        if [ -n "${IMPORTER}" ]; then
            echo "[INFO] Converting ${RAW_REPORT} with ${IMPORTER}."
            "${IMPORTER}" \
                -i "${RAW_REPORT}" \
                -o "${BASE}.qdrep"
            REPORT="${BASE}.qdrep"
        else
            echo "[WARN] Timeline capture succeeded, but this CUDA module only"
            echo "[WARN] provides the target-side collector. QdstrmImporter was"
            echo "[WARN] not found, so stats and SQLite cannot be generated here."
            echo "[WARN] Preserve and download: ${RAW_REPORT}"
            echo "[WARN] It must be imported with matching Nsight Systems 2021.2.4."
            echo "===== GENERATED FILES ====="
            ls -lh "${PROFILE_DIR}"
            exit 0
        fi
    else
        echo "[ERROR] Expected report was not produced:"
        echo "[ERROR] ${BASE}.nsys-rep/.qdrep/.qdstrm"
        exit 1
    fi
fi

echo "===== NSIGHT SYSTEMS STATS ====="
# CUDA 11.4 ships an older Nsight Systems whose report aliases differ from
# current releases.  Try the legacy aliases first, then fall back to its
# default report collection.
set +e
nsys stats \
    --report cudaapisum,gpukernsum,gpumemtimesum \
    "${REPORT}" >"${STATS_FILE}" 2>&1
STATS_STATUS=$?
if [ "${STATS_STATUS}" -ne 0 ]; then
    nsys stats "${REPORT}" >"${STATS_FILE}" 2>&1
fi

nsys export \
    --type sqlite \
    --force-overwrite=true \
    --output="${SQLITE_REPORT}" \
    "${REPORT}" >>"${STATS_FILE}" 2>&1
EXPORT_STATUS=$?
set -e

if [ "${EXPORT_STATUS}" -ne 0 ]; then
    echo "[WARN] SQLite export failed; the .nsys-rep is still valid."
fi

echo "===== GENERATED FILES ====="
ls -lh "${PROFILE_DIR}"
echo "Profile directory: ${PROFILE_DIR}"
echo "Report           : ${REPORT}"
echo "SQLite           : ${SQLITE_REPORT}"
echo "Stats            : ${STATS_FILE}"
