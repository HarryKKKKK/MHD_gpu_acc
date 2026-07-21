#!/bin/bash -l
#SBATCH -J mhd_launch_sweep
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

# Compile-time launch-configuration sweep for advance_x/advance_y.
#
# Defaults run two coordinate sweeps:
#   x phase: vary x while y stays at 16x8/LB=0
#   y phase: vary y while x stays at 16x8/LB=3
#
# Example:
#   sbatch scripts/csd3_slurm/sweep_gpu_launch.sh
#
# Smaller smoke test:
#   SWEEP_PHASE=x X_BLOCK_SHAPES="16x8 32x4" X_LAUNCH_BOUNDS="2 3" \
#   REPEATS=2 WARMUP_STEPS=20 BENCHMARK_STEPS=50 \
#   sbatch scripts/csd3_slurm/sweep_gpu_launch.sh

N="${N:-8}"
CASE="${CASE:-orszag_tang}"
SOLVER="${SOLVER:-hlld}"
SWEEP_PHASE="${SWEEP_PHASE:-both}"
REPEATS="${REPEATS:-5}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-500}"
RUN_COMBINED_WINNER="${RUN_COMBINED_WINNER:-1}"

X_BLOCK_SHAPES="${X_BLOCK_SHAPES:-16x8 32x4 8x16}"
Y_BLOCK_SHAPES="${Y_BLOCK_SHAPES:-16x8 32x4 8x16}"
X_LAUNCH_BOUNDS="${X_LAUNCH_BOUNDS:-0 2 3}"
Y_LAUNCH_BOUNDS="${Y_LAUNCH_BOUNDS:-0 2 3}"

FIXED_X_SHAPE="${FIXED_X_SHAPE:-16x8}"
FIXED_X_LB="${FIXED_X_LB:-3}"
FIXED_Y_SHAPE="${FIXED_Y_SHAPE:-16x8}"
FIXED_Y_LB="${FIXED_Y_LB:-0}"

TUNE_NVCC_FLAGS="${TUNE_NVCC_FLAGS:--Xptxas=-v}"

SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
cd "${WORKDIR}"

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

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

parse_shape() {
    local shape="$1"
    if ! [[ "${shape}" =~ ^([1-9][0-9]*)x([1-9][0-9]*)$ ]]; then
        echo "[ERROR] Invalid block shape: ${shape}; expected BXxBY." >&2
        return 2
    fi
    SHAPE_BX="${BASH_REMATCH[1]}"
    SHAPE_BY="${BASH_REMATCH[2]}"
    if [ $((SHAPE_BX * SHAPE_BY)) -ne 128 ]; then
        echo "[ERROR] ${shape} has $((SHAPE_BX * SHAPE_BY)) threads; this sweep requires 128." >&2
        return 2
    fi
}

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

for value in "${N}" "${REPEATS}" "${BENCHMARK_STEPS}"; do
    if ! is_positive_integer "${value}"; then
        echo "[ERROR] N, REPEATS and BENCHMARK_STEPS must be positive integers."
        exit 2
    fi
done

if ! is_nonnegative_integer "${WARMUP_STEPS}"; then
    echo "[ERROR] WARMUP_STEPS must be a non-negative integer."
    exit 2
fi

if [[ "${SWEEP_PHASE}" != "x" && "${SWEEP_PHASE}" != "y" && \
      "${SWEEP_PHASE}" != "both" ]]; then
    echo "[ERROR] SWEEP_PHASE must be x, y, or both."
    exit 2
fi

if [[ "${RUN_COMBINED_WINNER}" != "0" && "${RUN_COMBINED_WINNER}" != "1" ]]; then
    echo "[ERROR] RUN_COMBINED_WINNER must be 0 or 1."
    exit 2
fi

parse_shape "${FIXED_X_SHAPE}"
FIXED_X_BX="${SHAPE_BX}"
FIXED_X_BY="${SHAPE_BY}"
parse_shape "${FIXED_Y_SHAPE}"
FIXED_Y_BX="${SHAPE_BX}"
FIXED_Y_BY="${SHAPE_BY}"

for lb in ${X_LAUNCH_BOUNDS} ${Y_LAUNCH_BOUNDS} "${FIXED_X_LB}" "${FIXED_Y_LB}"; do
    if ! is_nonnegative_integer "${lb}"; then
        echo "[ERROR] Launch-bound values must be non-negative integers: ${lb}"
        exit 2
    fi
done

TUNING_REL="tuning/${SLURM_JOB_ID}_${CASE}_${SOLVER}_n${N}"
TUNING_ROOT="${WORKDIR}/${TUNING_REL}"
mkdir -p logs "${TUNING_ROOT}/build_logs" "${TUNING_ROOT}/run_logs"

RAW_CSV="${TUNING_ROOT}/launch_sweep_raw.csv"
SUMMARY_CSV="${TUNING_ROOT}/launch_sweep_summary.csv"

echo "phase,tag,x_bx,x_by,x_lb,y_bx,y_by,y_lb,repeat,measured_steps,wall_ms,ms_per_step,x_ms_per_step,y_ms_per_step,state_hash,correctness,binary" > "${RAW_CSV}"
echo "phase,tag,x_bx,x_by,x_lb,y_bx,y_by,y_lb,repeats,measured_steps,median_wall_ms,median_ms_per_step,median_x_ms_per_step,median_y_ms_per_step,state_hash,correctness,binary" > "${SUMMARY_CSV}"

GIT_BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
REFERENCE_HASH=""

echo "===== LAUNCH SWEEP CONFIGURATION ====="
echo "job=${SLURM_JOB_ID} host=$(hostname)"
echo "git_branch=${GIT_BRANCH} git_commit=${GIT_COMMIT}"
echo "case=${CASE} solver=${SOLVER} n=${N} phase=${SWEEP_PHASE}"
echo "repeats=${REPEATS} warmup_steps=${WARMUP_STEPS} benchmark_steps=${BENCHMARK_STEPS}"
echo "run_combined_winner=${RUN_COMBINED_WINNER}"
echo "x_shapes=${X_BLOCK_SHAPES} x_lbs=${X_LAUNCH_BOUNDS}"
echo "y_shapes=${Y_BLOCK_SHAPES} y_lbs=${Y_LAUNCH_BOUNDS}"
echo "fixed_x=${FIXED_X_SHAPE}/LB${FIXED_X_LB} fixed_y=${FIXED_Y_SHAPE}/LB${FIXED_Y_LB}"
echo "output=${TUNING_ROOT}"
nvidia-smi --query-gpu=name,driver_version,clocks.current.sm,clocks.current.memory --format=csv || true

run_configuration() {
    local phase="$1"
    local x_bx="$2"
    local x_by="$3"
    local x_lb="$4"
    local y_bx="$5"
    local y_by="$6"
    local y_lb="$7"

    local tag="${phase}_x${x_bx}x${x_by}_lb${x_lb}_y${y_bx}x${y_by}_lb${y_lb}"
    local build_dir="${TUNING_REL}/build/${tag}"
    local bin_dir="${TUNING_REL}/bin/${tag}"
    local bin="${WORKDIR}/${bin_dir}/main_gpu"
    local build_log="${TUNING_ROOT}/build_logs/${tag}.log"

    local flags="${TUNE_NVCC_FLAGS}"
    flags+=" -DMHD_ADVANCE_X_BLOCK_X=${x_bx}"
    flags+=" -DMHD_ADVANCE_X_BLOCK_Y=${x_by}"
    flags+=" -DMHD_ADVANCE_Y_BLOCK_X=${y_bx}"
    flags+=" -DMHD_ADVANCE_Y_BLOCK_Y=${y_by}"
    flags+=" -DMHD_ADVANCE_X_MIN_BLOCKS_PER_SM=${x_lb}"
    flags+=" -DMHD_ADVANCE_Y_MIN_BLOCKS_PER_SM=${y_lb}"

    echo ""
    echo "===== BUILD ${tag} ====="
    set +e
    make gpu \
        BUILD_DIR="${build_dir}" \
        BIN_DIR="${bin_dir}" \
        NVCC_EXTRA_FLAGS="${flags}" \
        2>&1 | tee "${build_log}"
    local build_status=${PIPESTATUS[0]}
    set -e
    if [ "${build_status}" -ne 0 ] || [ ! -x "${bin}" ]; then
        echo "[ERROR] Build failed for ${tag}; see ${build_log}."
        exit 1
    fi

    local wall_values=()
    local step_values=()
    local x_values=()
    local y_values=()
    local config_hash=""
    local correctness="PASS"
    local measured_steps=""

    for ((repeat = 1; repeat <= REPEATS; ++repeat)); do
        local run_log="${TUNING_ROOT}/run_logs/${tag}_r${repeat}.log"
        echo "===== RUN ${tag} repeat ${repeat}/${REPEATS} ====="

        set +e
        "${bin}" "${N}" \
            --case "${CASE}" \
            --solver "${SOLVER}" \
            --no-out \
            --warmup-steps "${WARMUP_STEPS}" \
            --benchmark-steps "${BENCHMARK_STEPS}" \
            >"${run_log}" 2>&1
        local run_status=$?
        set -e
        if [ "${run_status}" -ne 0 ]; then
            echo "[ERROR] Run failed for ${tag}, repeat ${repeat}; see ${run_log}."
            tail -n 50 "${run_log}" || true
            exit "${run_status}"
        fi

        local tuning_line
        tuning_line="$(grep '^TUNING_CSV,' "${run_log}" | tail -n 1 || true)"
        if [ -z "${tuning_line}" ]; then
            echo "[ERROR] Missing TUNING_CSV output in ${run_log}."
            exit 1
        fi

        local marker wall_ms ms_per_step x_ms y_ms state_hash
        IFS=',' read -r marker measured_steps wall_ms ms_per_step x_ms y_ms state_hash <<< "${tuning_line}"
        if [ "${marker}" != "TUNING_CSV" ]; then
            echo "[ERROR] Malformed tuning output: ${tuning_line}"
            exit 1
        fi

        if [ -z "${config_hash}" ]; then
            config_hash="${state_hash}"
        elif [ "${config_hash}" != "${state_hash}" ]; then
            correctness="FAIL_REPEAT_HASH"
        fi

        if [ -z "${REFERENCE_HASH}" ]; then
            REFERENCE_HASH="${state_hash}"
        elif [ "${REFERENCE_HASH}" != "${state_hash}" ]; then
            correctness="FAIL_CROSS_CONFIG_HASH"
        fi

        wall_values+=("${wall_ms}")
        step_values+=("${ms_per_step}")
        x_values+=("${x_ms}")
        y_values+=("${y_ms}")

        echo "${phase},${tag},${x_bx},${x_by},${x_lb},${y_bx},${y_by},${y_lb},${repeat},${measured_steps},${wall_ms},${ms_per_step},${x_ms},${y_ms},${state_hash},${correctness},${bin}" >> "${RAW_CSV}"
        echo "[RESULT] ${tag} r${repeat}: step=${ms_per_step} ms x=${x_ms} ms y=${y_ms} ms hash=${state_hash} ${correctness}"
    done

    local median_wall median_step median_x median_y
    median_wall="$(median "${wall_values[@]}")"
    median_step="$(median "${step_values[@]}")"
    median_x="$(median "${x_values[@]}")"
    median_y="$(median "${y_values[@]}")"

    echo "${phase},${tag},${x_bx},${x_by},${x_lb},${y_bx},${y_by},${y_lb},${REPEATS},${measured_steps},${median_wall},${median_step},${median_x},${median_y},${config_hash},${correctness},${bin}" >> "${SUMMARY_CSV}"
    echo "[MEDIAN] ${tag}: step=${median_step} ms x=${median_x} ms y=${median_y} ms ${correctness}"
}

if [[ "${SWEEP_PHASE}" == "x" || "${SWEEP_PHASE}" == "both" ]]; then
    for shape in ${X_BLOCK_SHAPES}; do
        parse_shape "${shape}"
        x_bx="${SHAPE_BX}"
        x_by="${SHAPE_BY}"
        for x_lb in ${X_LAUNCH_BOUNDS}; do
            run_configuration x \
                "${x_bx}" "${x_by}" "${x_lb}" \
                "${FIXED_Y_BX}" "${FIXED_Y_BY}" "${FIXED_Y_LB}"
        done
    done
fi

if [[ "${SWEEP_PHASE}" == "y" || "${SWEEP_PHASE}" == "both" ]]; then
    for shape in ${Y_BLOCK_SHAPES}; do
        parse_shape "${shape}"
        y_bx="${SHAPE_BX}"
        y_by="${SHAPE_BY}"
        for y_lb in ${Y_LAUNCH_BOUNDS}; do
            run_configuration y \
                "${FIXED_X_BX}" "${FIXED_X_BY}" "${FIXED_X_LB}" \
                "${y_bx}" "${y_by}" "${y_lb}"
        done
    done
fi

if [[ "${SWEEP_PHASE}" == "both" && "${RUN_COMBINED_WINNER}" == "1" ]]; then
    best_x_line="$(awk -F, '$1 == "x" && $16 == "PASS"' "${SUMMARY_CSV}" | sort -t, -k13,13n | head -n 1)"
    best_y_line="$(awk -F, '$1 == "y" && $16 == "PASS"' "${SUMMARY_CSV}" | sort -t, -k14,14n | head -n 1)"

    if [[ -n "${best_x_line}" && -n "${best_y_line}" ]]; then
        IFS=',' read -r _ _ best_x_bx best_x_by best_x_lb _ _ _ _ _ _ _ _ _ _ _ _ <<< "${best_x_line}"
        IFS=',' read -r _ _ _ _ _ best_y_bx best_y_by best_y_lb _ _ _ _ _ _ _ _ _ <<< "${best_y_line}"

        echo ""
        echo "===== COMBINED WINNER CONFIRMATION ====="
        echo "x=${best_x_bx}x${best_x_by}/LB${best_x_lb}"
        echo "y=${best_y_bx}x${best_y_by}/LB${best_y_lb}"
        run_configuration combined \
            "${best_x_bx}" "${best_x_by}" "${best_x_lb}" \
            "${best_y_bx}" "${best_y_by}" "${best_y_lb}"
    else
        echo "[WARN] No correctness-passing x/y pair was available for combined confirmation."
    fi
fi

echo ""
echo "===== X RANKING (median advance_x time) ====="
awk -F, 'NR > 1 && $1 == "x" { print $13 "," $2 "," $16 }' "${SUMMARY_CSV}" | sort -t, -k1,1n

echo ""
echo "===== Y RANKING (median advance_y time) ====="
awk -F, 'NR > 1 && $1 == "y" { print $14 "," $2 "," $16 }' "${SUMMARY_CSV}" | sort -t, -k1,1n

echo ""
echo "===== COMBINED WINNER ====="
awk -F, 'NR > 1 && $1 == "combined" { print "step_ms=" $12 ", x_ms=" $13 ", y_ms=" $14 ", " $2 ", " $16 }' "${SUMMARY_CSV}"

echo ""
echo "Raw results    : ${RAW_CSV}"
echo "Median summary : ${SUMMARY_CSV}"
echo "Reference hash : ${REFERENCE_HASH}"
echo "Completed      : $(date --iso-8601=seconds)"
