#!/bin/bash -l
#SBATCH --job-name=euler_gpu_profile
#SBATCH --account=MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# End-to-end profiling workflow for the 2D Euler CUDA solver:
#   1. Build with source line information and ptxas resource reporting.
#   2. Run repeated, unprofiled timing baselines.
#   3. Capture an Nsight Systems CUDA timeline.
#   4. Profile one steady-state advance_x and advance_y kernel with
#      Nsight Compute.
#
# Typical submission:
#   mkdir -p logs
#   sbatch scripts/csd3_slurm/profile_gpu_systems.sh
#
# Larger representative workload:
#   sbatch --export=ALL,N=4,CASE=blast_wave,SOLVER=hllc \
#       scripts/csd3_slurm/profile_gpu_systems.sh
#
# Smoke test:
#   sbatch --export=ALL,N=1,BASELINE_REPEATS=2,BASELINE_STEPS=20,\
#NSYS_STEPS=10,NCU_SET=basic,NCU_LAUNCH_SKIP=2 \
#       scripts/csd3_slurm/profile_gpu_systems.sh

set -euo pipefail

JOB_ID="${SLURM_JOB_ID:-manual}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
N="${N:-4}"
CASE="${CASE:-blast_wave}"
SOLVER="${SOLVER:-hllc}"

BASELINE_REPEATS="${BASELINE_REPEATS:-5}"
BASELINE_WARMUP="${BASELINE_WARMUP:-50}"
BASELINE_STEPS="${BASELINE_STEPS:-200}"
NSYS_WARMUP="${NSYS_WARMUP:-20}"
NSYS_STEPS="${NSYS_STEPS:-50}"
NCU_WARMUP="${NCU_WARMUP:-20}"
NCU_STEPS="${NCU_STEPS:-20}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-10}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"
NCU_SET="${NCU_SET:-full}"

CUDA_ARCH="${CUDA_ARCH:--arch=sm_80}"
PROFILE_NVCC_FLAGS="${PROFILE_NVCC_FLAGS:--lineinfo -Xptxas=-v}"
MAKE_JOBS="${MAKE_JOBS:-8}"

RESULT_DIR="${RESULT_DIR:-profiling/euler_${JOB_ID}_${CASE}_${SOLVER}_n${N}}"
BUILD_ROOT="${BUILD_ROOT:-build/euler_profile_${JOB_ID}}"
BIN_ROOT="${BIN_ROOT:-bin/euler_profile_${JOB_ID}}"
APP="${BIN_ROOT}/main_gpu"

cd "${WORKDIR}"
mkdir -p logs "${RESULT_DIR}" "${BUILD_ROOT}" "${BIN_ROOT}"

positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

for value in "${N}" "${BASELINE_REPEATS}" "${BASELINE_STEPS}" \
             "${NSYS_STEPS}" "${NCU_STEPS}" "${NCU_LAUNCH_COUNT}"; do
    if ! positive_integer "${value}"; then
        echo "[ERROR] Expected a positive integer, got '${value}'."
        exit 2
    fi
done

for value in "${BASELINE_WARMUP}" "${NSYS_WARMUP}" "${NCU_WARMUP}" \
             "${NCU_LAUNCH_SKIP}"; do
    if ! nonnegative_integer "${value}"; then
        echo "[ERROR] Expected a non-negative integer, got '${value}'."
        exit 2
    fi
done

case "${CASE}" in
    blast_wave|kelvin_helmholtz|shock_bubble) ;;
    *)
        echo "[ERROR] CASE must be blast_wave, kelvin_helmholtz, or shock_bubble."
        exit 2
        ;;
esac

case "${SOLVER}" in
    hll|hllc|force) ;;
    *)
        echo "[ERROR] SOLVER must be hll, hllc, or force."
        exit 2
        ;;
esac

echo "===== MODULE SETUP ====="
if ! command -v module >/dev/null 2>&1; then
    for init_file in /etc/profile.d/modules.sh \
                     /usr/share/Modules/init/bash \
                     /usr/local/Modules/init/bash; do
        if [ -f "${init_file}" ]; then
            # shellcheck disable=SC1090
            source "${init_file}"
            break
        fi
    done
fi

if command -v module >/dev/null 2>&1; then
    module purge
    module load rhel8/default-amp
fi

for tool in make nvcc nvidia-smi nsys ncu awk; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "[ERROR] '${tool}' is unavailable after module setup."
        if command -v module >/dev/null 2>&1; then
            module spider nsight 2>&1 || true
            module spider cuda 2>&1 || true
        fi
        exit 127
    fi
done

unset CUDA_LAUNCH_BLOCKING
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

GPU_INFO="$(
    nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version,\
clocks.current.sm,clocks.current.memory \
        --format=csv,noheader 2>/dev/null | head -1 || true
)"

{
    echo "date=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "job_id=${JOB_ID}"
    echo "workdir=${WORKDIR}"
    echo "git_branch=$(git branch --show-current 2>/dev/null || echo unknown)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "case=${CASE}"
    echo "solver=${SOLVER}"
    echo "n=${N}"
    echo "baseline_repeats=${BASELINE_REPEATS}"
    echo "baseline_warmup=${BASELINE_WARMUP}"
    echo "baseline_steps=${BASELINE_STEPS}"
    echo "nsys_warmup=${NSYS_WARMUP}"
    echo "nsys_steps=${NSYS_STEPS}"
    echo "ncu_warmup=${NCU_WARMUP}"
    echo "ncu_steps=${NCU_STEPS}"
    echo "ncu_set=${NCU_SET}"
    echo "ncu_launch_skip=${NCU_LAUNCH_SKIP}"
    echo "ncu_launch_count=${NCU_LAUNCH_COUNT}"
    echo "cuda_arch=${CUDA_ARCH}"
    echo "nvcc_extra_flags=${PROFILE_NVCC_FLAGS}"
    echo "gpu=${GPU_INFO}"
    echo "nvcc=$(nvcc --version 2>&1 | tr '\n' ' ')"
    echo "nsys=$(nsys --version 2>&1 | tr '\n' ' ')"
    echo "ncu=$(ncu --version 2>&1 | tr '\n' ' ')"
} | tee "${RESULT_DIR}/metadata.txt"

echo "===== BUILD EULER CUDA SOLVER ====="
make -j "${MAKE_JOBS}" gpu \
    BUILD_DIR="${BUILD_ROOT}" \
    BIN_DIR="${BIN_ROOT}" \
    CUDA_ARCH="${CUDA_ARCH}" \
    NVCC_EXTRA_FLAGS="${PROFILE_NVCC_FLAGS}" \
    2>&1 | tee "${RESULT_DIR}/build.txt"

if [ ! -x "${APP}" ]; then
    echo "[ERROR] Build did not produce ${APP}."
    exit 1
fi

COMMON_ARGS=(
    "${N}"
    --case "${CASE}"
    --solver "${SOLVER}"
    --no-out
)

echo "===== UNPROFILED BASELINE ====="
BASELINE_CSV="${RESULT_DIR}/baseline.csv"
echo "repeat,measured_steps,wall_ms,ms_per_step,x_ms_per_step,y_ms_per_step,state_hash" \
    > "${BASELINE_CSV}"

for ((repeat=1; repeat<=BASELINE_REPEATS; ++repeat)); do
    log="${RESULT_DIR}/baseline_${repeat}.txt"
    echo "----- baseline ${repeat}/${BASELINE_REPEATS} -----"
    "${APP}" "${COMMON_ARGS[@]}" \
        --warmup-steps "${BASELINE_WARMUP}" \
        --benchmark-steps "${BASELINE_STEPS}" \
        2>&1 | tee "${log}"

    line="$(grep '^TUNING_CSV,' "${log}" | tail -1 || true)"
    if [ -z "${line}" ]; then
        echo "[ERROR] Missing TUNING_CSV in ${log}."
        exit 1
    fi
    echo "${repeat},${line#TUNING_CSV,}" >> "${BASELINE_CSV}"
done

BASELINE_SUMMARY="${RESULT_DIR}/baseline_summary.txt"
awk -F, '
    NR > 1 {
        wall[++n]=$3
        step[n]=$4
        x[n]=$5
        y[n]=$6
    }
    function sort(a,count, i,j,t) {
        for(i=1;i<=count;i++) for(j=i+1;j<=count;j++)
            if(a[j]<a[i]) { t=a[i];a[i]=a[j];a[j]=t }
    }
    function median(a,count) {
        sort(a,count)
        return count%2 ? a[(count+1)/2] : (a[count/2]+a[count/2+1])/2
    }
    END {
        printf "repeats=%d\n",n
        printf "median_wall_ms=%.9f\n",median(wall,n)
        printf "median_ms_per_step=%.9f\n",median(step,n)
        printf "median_x_ms_per_step=%.9f\n",median(x,n)
        printf "median_y_ms_per_step=%.9f\n",median(y,n)
    }
' "${BASELINE_CSV}" | tee "${BASELINE_SUMMARY}"

echo "===== NSIGHT SYSTEMS TIMELINE ====="
NSYS_BASE="${RESULT_DIR}/nsys_${CASE}_${SOLVER}_n${N}"
nsys profile \
    --force-overwrite=true \
    --trace=cuda,osrt \
    --sample=none \
    --output="${NSYS_BASE}" \
    "${APP}" "${COMMON_ARGS[@]}" \
        --warmup-steps "${NSYS_WARMUP}" \
        --benchmark-steps "${NSYS_STEPS}" \
    2>&1 | tee "${RESULT_DIR}/nsys_console.txt"

NSYS_REPORT=""
for candidate in "${NSYS_BASE}.nsys-rep" "${NSYS_BASE}.qdrep"; do
    if [ -f "${candidate}" ]; then
        NSYS_REPORT="${candidate}"
        break
    fi
done

if [ -n "${NSYS_REPORT}" ]; then
    set +e
    nsys stats \
        --report cudaapisum,gpukernsum,gpumemtimesum \
        "${NSYS_REPORT}" > "${RESULT_DIR}/nsys_stats.txt" 2>&1
    stats_status=$?
    if [ "${stats_status}" -ne 0 ]; then
        nsys stats "${NSYS_REPORT}" > "${RESULT_DIR}/nsys_stats.txt" 2>&1
    fi
    set -e
else
    echo "[WARN] No finalized .nsys-rep/.qdrep was found."
    echo "[WARN] Check for a .qdstrm file and import it with a matching nsys version."
fi

echo "===== NSIGHT COMPUTE KERNEL PROFILES ====="
for axis in x y; do
    NCU_BASE="${RESULT_DIR}/ncu_advance_${axis}_${CASE}_${SOLVER}_n${N}"
    echo "----- advance_${axis}_kernel -----"
    set +e
    ncu \
        --force-overwrite \
        --set "${NCU_SET}" \
        --kernel-name-base function \
        --kernel-name "regex:.*advance_${axis}_kernel.*" \
        --launch-skip "${NCU_LAUNCH_SKIP}" \
        --launch-count "${NCU_LAUNCH_COUNT}" \
        --export "${NCU_BASE}" \
        "${APP}" "${COMMON_ARGS[@]}" \
            --warmup-steps "${NCU_WARMUP}" \
            --benchmark-steps "${NCU_STEPS}" \
        2>&1 | tee "${RESULT_DIR}/ncu_advance_${axis}_console.txt"
    ncu_status=${PIPESTATUS[0]}
    set -e

    if [ "${ncu_status}" -ne 0 ]; then
        echo "[ERROR] Nsight Compute failed for advance_${axis}."
        echo "[HINT] If the log contains ERR_NVGPUCTRPERM, GPU performance"
        echo "[HINT] counters are restricted and the cluster administrator must"
        echo "[HINT] enable profiling access for this partition."
        exit "${ncu_status}"
    fi
done

echo "===== PROFILE COMPLETE ====="
echo "Result directory: ${WORKDIR}/${RESULT_DIR}"
echo "Baseline CSV    : ${WORKDIR}/${BASELINE_CSV}"
echo "Baseline summary: ${WORKDIR}/${BASELINE_SUMMARY}"
if [ -n "${NSYS_REPORT}" ]; then
    echo "Systems report  : ${WORKDIR}/${NSYS_REPORT}"
fi
echo "Compute X report: ${WORKDIR}/${RESULT_DIR}/ncu_advance_x_${CASE}_${SOLVER}_n${N}.ncu-rep"
echo "Compute Y report: ${WORKDIR}/${RESULT_DIR}/ncu_advance_y_${CASE}_${SOLVER}_n${N}.ncu-rep"
ls -lh "${RESULT_DIR}"
