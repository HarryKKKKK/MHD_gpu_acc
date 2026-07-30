#!/bin/bash -l
#SBATCH -J mhd_fmad_ab
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Controlled FMA-contraction A/B test for the final Chapter 3 HLLD build.
# Two separate binaries are built with --fmad=false and --fmad=true.  Runs
# alternate their order to reduce thermal and clock-drift bias.

set -euo pipefail

N="${N:-8}"
CASE="${CASE:-orszag_tang}"
REPEATS="${REPEATS:-5}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-500}"

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
cd "${WORKDIR}"

is_positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_nonnegative_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }

if ! is_positive_integer "${N}" || \
   ! is_positive_integer "${REPEATS}" || \
   ! is_positive_integer "${BENCHMARK_STEPS}" || \
   ! is_nonnegative_integer "${WARMUP_STEPS}"; then
    echo "[ERROR] N, REPEATS and BENCHMARK_STEPS must be positive integers;"
    echo "[ERROR] WARMUP_STEPS must be a non-negative integer."
    exit 2
fi

if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -f /usr/share/Modules/init/bash ]; then
        source /usr/share/Modules/init/bash
    fi
fi

if command -v module >/dev/null 2>&1; then
    module purge
    module load rhel8/default-amp
fi

unset CUDA_LAUNCH_BLOCKING
export OMP_NUM_THREADS=1

RESULT_REL="tuning/${SLURM_JOB_ID}_${CASE}_hlld_n${N}_fmad_ab"
RESULT_DIR="${WORKDIR}/${RESULT_REL}"
mkdir -p logs "${RESULT_DIR}/build_logs" "${RESULT_DIR}/run_logs"

RAW_CSV="${RESULT_DIR}/fmad_ab_raw.csv"
SUMMARY_CSV="${RESULT_DIR}/fmad_ab_summary.csv"
METADATA="${RESULT_DIR}/metadata.txt"

echo "fmad,repeat,run_order,measured_steps,wall_ms,ms_per_step,x_ms_per_step,y_ms_per_step,state_hash,validation" > "${RAW_CSV}"
echo "fmad,repeats,measured_steps,median_wall_ms,median_ms_per_step,median_x_ms_per_step,median_y_ms_per_step,state_hash,validation" > "${SUMMARY_CSV}"

median() {
    printf '%s\n' "$@" | sort -n | awk '
        { value[NR] = $1 }
        END {
            if (NR == 0) exit 1
            if (NR % 2 == 1) printf "%.9f", value[(NR + 1) / 2]
            else printf "%.9f", (value[NR / 2] + value[NR / 2 + 1]) / 2.0
        }
    '
}

declare -A BIN
declare -A WALL_VALUES
declare -A STEP_VALUES
declare -A X_VALUES
declare -A Y_VALUES
declare -A MODE_HASH
declare -A VALIDATION
declare -A MEDIAN_STEP
declare -A MEDIAN_X
declare -A MEDIAN_Y

for mode in false true; do
    build_rel="${RESULT_REL}/build/fmad_${mode}"
    bin_rel="${RESULT_REL}/bin/fmad_${mode}"
    BIN["${mode}"]="${WORKDIR}/${bin_rel}/main_gpu"
    build_log="${RESULT_DIR}/build_logs/fmad_${mode}.log"

    flags="-Xptxas=-v"
    flags+=" -DMHD_HLLD_CANONICALIZE_Y=1"
    flags+=" -DMHD_ADVANCE_X_BLOCK_X=16 -DMHD_ADVANCE_X_BLOCK_Y=8"
    flags+=" -DMHD_ADVANCE_Y_BLOCK_X=16 -DMHD_ADVANCE_Y_BLOCK_Y=8"
    flags+=" -DMHD_ADVANCE_X_MIN_BLOCKS_PER_SM=3"
    flags+=" -DMHD_ADVANCE_Y_MIN_BLOCKS_PER_SM=3"

    echo "===== BUILD fmad=${mode} ====="
    set +e
    make gpu \
        BUILD_DIR="${build_rel}" \
        BIN_DIR="${bin_rel}" \
        FMAD="${mode}" \
        NVCC_EXTRA_FLAGS="${flags}" \
        2>&1 | tee "${build_log}"
    build_status=${PIPESTATUS[0]}
    set -e

    if [ "${build_status}" -ne 0 ] || [ ! -x "${BIN[${mode}]}" ]; then
        echo "[ERROR] Build failed for fmad=${mode}; see ${build_log}."
        exit 1
    fi
    if ! grep -q -- "--fmad=${mode}" "${build_log}"; then
        echo "[ERROR] Build log does not confirm --fmad=${mode}."
        exit 1
    fi

    WALL_VALUES["${mode}"]=""
    STEP_VALUES["${mode}"]=""
    X_VALUES["${mode}"]=""
    Y_VALUES["${mode}"]=""
    MODE_HASH["${mode}"]=""
    VALIDATION["${mode}"]="PASS"
done

{
    echo "date=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "slurm_job_id=${SLURM_JOB_ID}"
    echo "git_branch=$(git branch --show-current 2>/dev/null || echo unknown)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "case=${CASE}"
    echo "solver=hlld"
    echo "n=${N}"
    echo "repeats=${REPEATS}"
    echo "warmup_steps=${WARMUP_STEPS}"
    echo "benchmark_steps=${BENCHMARK_STEPS}"
    echo "configuration=canonical_y,x_lb3,y_lb3,x16x8,y16x8"
    nvidia-smi --query-gpu=name,driver_version,clocks.current.sm,clocks.current.memory --format=csv
    nvcc --version
} > "${METADATA}"

run_once() {
    local mode="$1"
    local repeat="$2"
    local run_order="$3"
    local run_log="${RESULT_DIR}/run_logs/fmad_${mode}_r${repeat}.log"

    echo "===== RUN repeat=${repeat}/${REPEATS} order=${run_order} fmad=${mode} ====="
    set +e
    "${BIN[${mode}]}" "${N}" \
        --case "${CASE}" \
        --solver hlld \
        --no-out \
        --warmup-steps "${WARMUP_STEPS}" \
        --benchmark-steps "${BENCHMARK_STEPS}" \
        >"${run_log}" 2>&1
    run_status=$?
    set -e
    if [ "${run_status}" -ne 0 ]; then
        echo "[ERROR] fmad=${mode} repeat=${repeat} failed; see ${run_log}."
        tail -n 50 "${run_log}" || true
        exit "${run_status}"
    fi

    tuning_line="$(grep '^TUNING_CSV,' "${run_log}" | tail -n 1 || true)"
    if [ -z "${tuning_line}" ]; then
        echo "[ERROR] Missing TUNING_CSV output in ${run_log}."
        exit 1
    fi

    IFS=',' read -r marker measured_steps wall_ms ms_per_step x_ms y_ms state_hash <<< "${tuning_line}"
    if [ "${marker}" != "TUNING_CSV" ]; then
        echo "[ERROR] Malformed tuning output: ${tuning_line}"
        exit 1
    fi

    if [ -z "${MODE_HASH[${mode}]}" ]; then
        MODE_HASH["${mode}"]="${state_hash}"
    elif [ "${MODE_HASH[${mode}]}" != "${state_hash}" ]; then
        VALIDATION["${mode}"]="FAIL_REPEAT_HASH"
    fi

    WALL_VALUES["${mode}"]+="${wall_ms} "
    STEP_VALUES["${mode}"]+="${ms_per_step} "
    X_VALUES["${mode}"]+="${x_ms} "
    Y_VALUES["${mode}"]+="${y_ms} "

    echo "${mode},${repeat},${run_order},${measured_steps},${wall_ms},${ms_per_step},${x_ms},${y_ms},${state_hash},${VALIDATION[${mode}]}" >> "${RAW_CSV}"
    echo "[RESULT] fmad=${mode}: step=${ms_per_step} ms, x=${x_ms} ms, y=${y_ms} ms, hash=${state_hash}, ${VALIDATION[${mode}]}"
}

for ((repeat = 1; repeat <= REPEATS; ++repeat)); do
    if (( repeat % 2 == 1 )); then
        run_once false "${repeat}" 1
        run_once true  "${repeat}" 2
    else
        run_once true  "${repeat}" 1
        run_once false "${repeat}" 2
    fi
done

for mode in false true; do
    median_wall="$(median ${WALL_VALUES[${mode}]})"
    MEDIAN_STEP["${mode}"]="$(median ${STEP_VALUES[${mode}]})"
    MEDIAN_X["${mode}"]="$(median ${X_VALUES[${mode}]})"
    MEDIAN_Y["${mode}"]="$(median ${Y_VALUES[${mode}]})"

    echo "${mode},${REPEATS},${BENCHMARK_STEPS},${median_wall},${MEDIAN_STEP[${mode}]},${MEDIAN_X[${mode}]},${MEDIAN_Y[${mode}]},${MODE_HASH[${mode}]},${VALIDATION[${mode}]}" >> "${SUMMARY_CSV}"
done

speedup_pct() {
    awk -v strict="$1" -v contracted="$2" \
        'BEGIN { printf "%.6f", 100.0 * (strict - contracted) / strict }'
}

echo ""
echo "===== FMA A/B SUMMARY ====="
echo "fmad=false: step=${MEDIAN_STEP[false]} ms, x=${MEDIAN_X[false]} ms, y=${MEDIAN_Y[false]} ms, ${VALIDATION[false]}"
echo "fmad=true : step=${MEDIAN_STEP[true]} ms, x=${MEDIAN_X[true]} ms, y=${MEDIAN_Y[true]} ms, ${VALIDATION[true]}"
echo "speed-up from enabling contraction:"
echo "  step=$(speedup_pct "${MEDIAN_STEP[false]}" "${MEDIAN_STEP[true]}")%"
echo "  x=$(speedup_pct "${MEDIAN_X[false]}" "${MEDIAN_X[true]}")%"
echo "  y=$(speedup_pct "${MEDIAN_Y[false]}" "${MEDIAN_Y[true]}")%"

if [ "${MODE_HASH[false]}" = "${MODE_HASH[true]}" ]; then
    echo "cross-mode state hashes: identical"
else
    echo "cross-mode state hashes: different, as permitted for different rounding modes"
fi

echo "Raw results    : ${RAW_CSV}"
echo "Median summary : ${SUMMARY_CSV}"
echo "Metadata       : ${METADATA}"
echo "Completed      : $(date --iso-8601=seconds)"
