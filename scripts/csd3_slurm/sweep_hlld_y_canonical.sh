#!/bin/bash -l
#SBATCH -J hlld_y_canonical
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# A/B experiment for canonicalizing HLLD's y-direction Riemann solve through
# the x-direction implementation.  The native baseline and all experimental
# binaries are built into separate directories and checked against one state
# hash.  Canonical variants also test whether y launch bounds become useful
# after any register-pressure reduction.

set -euo pipefail

N="${N:-8}"
CASE="${CASE:-orszag_tang}"
REPEATS="${REPEATS:-5}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-500}"
TUNE_NVCC_FLAGS="${TUNE_NVCC_FLAGS:--Xptxas=-v}"

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
    echo "[ERROR] N, REPEATS and BENCHMARK_STEPS must be positive;"
    echo "[ERROR] WARMUP_STEPS must be non-negative."
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

RESULT_REL="tuning/${SLURM_JOB_ID}_${CASE}_hlld_n${N}_y_canonical"
RESULT_DIR="${WORKDIR}/${RESULT_REL}"
mkdir -p logs "${RESULT_DIR}/build_logs" "${RESULT_DIR}/run_logs"

RAW_CSV="${RESULT_DIR}/y_canonical_raw.csv"
SUMMARY_CSV="${RESULT_DIR}/y_canonical_summary.csv"
echo "variant,canonicalize_y,y_lb,registers,stack_bytes,spill_store_bytes,spill_load_bytes,repeat,measured_steps,wall_ms,ms_per_step,x_ms_per_step,y_ms_per_step,state_hash,correctness" > "${RAW_CSV}"
echo "variant,canonicalize_y,y_lb,registers,stack_bytes,spill_store_bytes,spill_load_bytes,repeats,measured_steps,median_wall_ms,median_ms_per_step,median_x_ms_per_step,median_y_ms_per_step,state_hash,correctness" > "${SUMMARY_CSV}"

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

extract_y_resources() {
    awk '
        /advance_y_kernelIL13RiemannSolver2EEE/ { target = 1; next }
        target && /bytes stack frame/ {
            stack = $1; stores = $5; loads = $9
        }
        target && /Used [0-9]+ registers/ {
            for (i = 1; i <= NF; ++i) {
                if ($i == "Used") {
                    print $(i + 1), stack, stores, loads
                    exit
                }
            }
        }
    ' "$1"
}

REFERENCE_HASH=""

run_variant() {
    local variant="$1"
    local canonicalize_y="$2"
    local y_lb="$3"

    local build_rel="${RESULT_REL}/build/${variant}"
    local bin_rel="${RESULT_REL}/bin/${variant}"
    local bin="${WORKDIR}/${bin_rel}/main_gpu"
    local build_log="${RESULT_DIR}/build_logs/${variant}.log"

    local flags="${TUNE_NVCC_FLAGS}"
    flags+=" -DMHD_HLLD_CANONICALIZE_Y=${canonicalize_y}"
    flags+=" -DMHD_ADVANCE_X_BLOCK_X=16 -DMHD_ADVANCE_X_BLOCK_Y=8"
    flags+=" -DMHD_ADVANCE_Y_BLOCK_X=16 -DMHD_ADVANCE_Y_BLOCK_Y=8"
    flags+=" -DMHD_ADVANCE_X_MIN_BLOCKS_PER_SM=3"
    flags+=" -DMHD_ADVANCE_Y_MIN_BLOCKS_PER_SM=${y_lb}"

    echo ""
    echo "===== BUILD ${variant} ====="
    set +e
    make gpu \
        BUILD_DIR="${build_rel}" \
        BIN_DIR="${bin_rel}" \
        NVCC_EXTRA_FLAGS="${flags}" \
        2>&1 | tee "${build_log}"
    local build_status=${PIPESTATUS[0]}
    set -e
    if [ "${build_status}" -ne 0 ] || [ ! -x "${bin}" ]; then
        echo "[ERROR] Build failed for ${variant}; see ${build_log}."
        exit 1
    fi

    local resources
    resources="$(extract_y_resources "${build_log}")"
    if [ -z "${resources}" ]; then
        echo "[ERROR] Could not extract HLLD advance_y resources from ${build_log}."
        exit 1
    fi

    local registers stack_bytes spill_store_bytes spill_load_bytes
    read -r registers stack_bytes spill_store_bytes spill_load_bytes <<< "${resources}"
    echo "[RESOURCE] y registers=${registers}, stack=${stack_bytes}, spill stores=${spill_store_bytes}, spill loads=${spill_load_bytes}"

    local wall_values=()
    local step_values=()
    local x_values=()
    local y_values=()
    local variant_hash=""
    local correctness="PASS"
    local measured_steps=""

    for ((repeat = 1; repeat <= REPEATS; ++repeat)); do
        local run_log="${RESULT_DIR}/run_logs/${variant}_r${repeat}.log"
        "${bin}" "${N}" \
            --case "${CASE}" \
            --solver hlld \
            --no-out \
            --warmup-steps "${WARMUP_STEPS}" \
            --benchmark-steps "${BENCHMARK_STEPS}" \
            >"${run_log}" 2>&1

        local tuning_line
        tuning_line="$(grep '^TUNING_CSV,' "${run_log}" | tail -n 1 || true)"
        if [ -z "${tuning_line}" ]; then
            echo "[ERROR] Missing TUNING_CSV output in ${run_log}."
            tail -n 50 "${run_log}" || true
            exit 1
        fi

        local marker wall_ms ms_per_step x_ms y_ms state_hash
        IFS=',' read -r marker measured_steps wall_ms ms_per_step x_ms y_ms state_hash <<< "${tuning_line}"

        if [ -z "${variant_hash}" ]; then
            variant_hash="${state_hash}"
        elif [ "${variant_hash}" != "${state_hash}" ]; then
            correctness="FAIL_REPEAT_HASH"
        fi

        if [ -z "${REFERENCE_HASH}" ]; then
            REFERENCE_HASH="${state_hash}"
        elif [ "${REFERENCE_HASH}" != "${state_hash}" ]; then
            correctness="FAIL_BASELINE_HASH"
        fi

        wall_values+=("${wall_ms}")
        step_values+=("${ms_per_step}")
        x_values+=("${x_ms}")
        y_values+=("${y_ms}")

        echo "${variant},${canonicalize_y},${y_lb},${registers},${stack_bytes},${spill_store_bytes},${spill_load_bytes},${repeat},${measured_steps},${wall_ms},${ms_per_step},${x_ms},${y_ms},${state_hash},${correctness}" >> "${RAW_CSV}"
        echo "[RESULT] ${variant} r${repeat}: step=${ms_per_step} ms, x=${x_ms} ms, y=${y_ms} ms, hash=${state_hash}, ${correctness}"
    done

    local median_wall median_step median_x median_y
    median_wall="$(median "${wall_values[@]}")"
    median_step="$(median "${step_values[@]}")"
    median_x="$(median "${x_values[@]}")"
    median_y="$(median "${y_values[@]}")"

    echo "${variant},${canonicalize_y},${y_lb},${registers},${stack_bytes},${spill_store_bytes},${spill_load_bytes},${REPEATS},${measured_steps},${median_wall},${median_step},${median_x},${median_y},${variant_hash},${correctness}" >> "${SUMMARY_CSV}"
    echo "[MEDIAN] ${variant}: step=${median_step} ms, x=${median_x} ms, y=${median_y} ms, ${correctness}"
}

echo "===== HLLD Y CANONICALIZATION SWEEP ====="
echo "job=${SLURM_JOB_ID} host=$(hostname)"
echo "git_branch=$(git branch --show-current 2>/dev/null || echo unknown)"
echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "case=${CASE} n=${N} repeats=${REPEATS}"
echo "warmup_steps=${WARMUP_STEPS} benchmark_steps=${BENCHMARK_STEPS}"
nvidia-smi --query-gpu=name,driver_version,clocks.current.sm,clocks.current.memory --format=csv || true

# Keep native first: its first repeat establishes the required state hash.
run_variant native_y_lb0       0 0
run_variant canonical_y_lb0    1 0
run_variant canonical_y_lb2    1 2
run_variant canonical_y_lb3    1 3

echo ""
echo "===== RANKING BY MEDIAN Y TIME ====="
awk -F, 'NR > 1 { print $13 "," $1 ",regs=" $4 "," $15 }' "${SUMMARY_CSV}" | sort -t, -k1,1n

echo ""
echo "Raw results    : ${RAW_CSV}"
echo "Median summary : ${SUMMARY_CSV}"
echo "Reference hash : ${REFERENCE_HASH}"
echo "Completed      : $(date --iso-8601=seconds)"
