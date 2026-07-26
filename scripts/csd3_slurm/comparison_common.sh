#!/bin/bash

# Shared helpers for the final CSD3 comparison job arrays.  This file is
# sourced by the GPU, OpenMP CPU, and pure-MPI submission scripts.

comparison_init_config() {
    read -r -a CASES <<< "${CASES_STR:-orszag_tang rotor}"
    read -r -a SOLVERS <<< "${SOLVERS_STR:-hll hllc hlld force}"
    read -r -a SCALES <<< "${SCALES_STR:-1 2 4 8}"

    if [ "${#CASES[@]}" -ne 2 ] || [ "${#SOLVERS[@]}" -ne 4 ] || [ "${#SCALES[@]}" -lt 1 ]; then
        echo "[ERROR] The array mapping requires exactly 2 cases, 4 solvers, and at least 1 scale."
        echo "[ERROR] cases=${CASES[*]} solvers=${SOLVERS[*]} scales=${SCALES[*]}"
        exit 2
    fi

    local task_id="${SLURM_ARRAY_TASK_ID:-0}"
    if ! [[ "${task_id}" =~ ^[0-9]+$ ]] || [ "${task_id}" -ge 4 ]; then
        echo "[ERROR] SLURM_ARRAY_TASK_ID must be in [0, 3]; got '${task_id}'."
        exit 2
    fi

    # One Slurm array task owns one solver.  Cases and scales are executed
    # sequentially inside that allocation, so each solver is compiled once.
    SOLVER_INDEX="${task_id}"
    SOLVER_NAME="${SOLVERS[${SOLVER_INDEX}]}"
}

comparison_set_case_scale() {
    CASE_NAME="$1"
    N_SCALE="$2"

    if ! [[ "${N_SCALE}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] Scale must be a positive integer; got '${N_SCALE}'."
        exit 2
    fi

    if [ "${N_SCALE}" -le 2 ]; then
        NUM_REPEATS="${SMALL_N_REPEATS:-3}"
    else
        NUM_REPEATS="${LARGE_N_REPEATS:-1}"
    fi

    if ! [[ "${NUM_REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] Repeat count must be a positive integer; got '${NUM_REPEATS}'."
        exit 2
    fi
}

comparison_load_module() {
    local module_name="$1"

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
    if ! module load "${module_name}"; then
        echo "[ERROR] Failed to load ${module_name}."
        exit 1
    fi
}

comparison_prepare_paths() {
    SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
    SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
    SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"

    cd "${WORKDIR}"

    RUN_ID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    RESULT_DIR="${WORKDIR}/timing/final_comparison/${RUN_ID}/${BACKEND}"
    BUILD_ROOT="${WORKDIR}/build/final_comparison/${RUN_ID}/${BACKEND}"
    BIN_ROOT="${WORKDIR}/bin/final_comparison/${RUN_ID}/${BACKEND}"
    mkdir -p logs "${RESULT_DIR}/runs" "${BUILD_ROOT}" "${BIN_ROOT}"

    SUMMARY_FILE="${RESULT_DIR}/${BACKEND}_${SOLVER_NAME}.csv"
    METADATA_FILE="${RESULT_DIR}/metadata.txt"

    echo "backend,case,solver,n,repeat,repeats_requested,build_variant,threads,ranks,nx,ny,total_cells,steps,app_elapsed_s,steps_per_s,Mcell_updates_s,wall_seconds,user_seconds,sys_seconds,cpu_percent,max_rss_kb,major_page_faults,minor_page_faults,voluntary_context_switches,involuntary_context_switches,start_utc,end_utc,exit_status,hostname,git_branch,git_commit" > "${SUMMARY_FILE}"
}

comparison_write_metadata() {
    local compiler_lines="$1"
    GIT_BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
    GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    GIT_STATUS="$(git status --short 2>/dev/null || true)"

    {
        echo "===== EXPERIMENT ====="
        echo "backend=${BACKEND}"
        echo "cases=${CASES[*]}"
        echo "solver=${SOLVER_NAME}"
        echo "scales=${SCALES[*]}"
        echo "small_n_repeats=${SMALL_N_REPEATS:-3}"
        echo "large_n_repeats=${LARGE_N_REPEATS:-1}"
        echo "array_job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
        echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
        echo "job_id=${SLURM_JOB_ID}"
        echo "hostname=$(hostname)"
        echo "start_utc=$(date --utc --iso-8601=seconds)"
        echo "workdir=${WORKDIR}"
        echo "git_branch=${GIT_BRANCH}"
        echo "git_commit=${GIT_COMMIT}"
        echo "git_status_begin"
        printf '%s\n' "${GIT_STATUS}"
        echo "git_status_end"
        echo "slurm_partition=${SLURM_JOB_PARTITION:-unknown}"
        echo "slurm_nodes=${SLURM_JOB_NUM_NODES:-unknown}"
        echo "slurm_ntasks=${SLURM_NTASKS:-unknown}"
        echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-unknown}"
        echo "build_variant=${BUILD_VARIANT}"
        echo "compiler_begin"
        printf '%s\n' "${compiler_lines}"
        echo "compiler_end"
        echo "modules_begin"
        module list 2>&1 || true
        echo "modules_end"
        echo "lscpu_begin"
        lscpu || true
        echo "lscpu_end"
        echo "slurm_job_begin"
        scontrol show job "${SLURM_JOB_ID}" 2>/dev/null || true
        echo "slurm_job_end"
    } > "${METADATA_FILE}"
}

comparison_time_value() {
    local key="$1"
    local file="$2"
    awk -F= -v wanted="${key}" '$1 == wanted {print $2; exit}' "${file}"
}

comparison_app_value() {
    local key="$1"
    local file="$2"
    awk -v wanted="${key}" '
        /^\[TIMING\]/ {
            for (i = 1; i <= NF; ++i) {
                split($i, item, "=")
                if (item[1] == wanted) value = item[2]
            }
        }
        END { if (value != "") print value }
    ' "${file}"
}

comparison_run_once() {
    local repeat_index="$1"
    shift
    local -a command=("$@")

    local stem="${RESULT_DIR}/runs/${CASE_NAME}_${SOLVER_NAME}_n${N_SCALE}_repeat_${repeat_index}"
    local console_file="${stem}.log"
    local time_file="${stem}.time"
    local start_utc end_utc start_ns end_ns wall_seconds status

    echo ""
    echo "============================================================"
    echo "RUN backend=${BACKEND} case=${CASE_NAME} solver=${SOLVER_NAME} n=${N_SCALE} repeat=${repeat_index}/${NUM_REPEATS}"
    echo "============================================================"
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'

    # GNU coreutils on CSD3 accepts "ns" (not "nanoseconds") as the
    # nanosecond-resolution --iso-8601 argument.
    start_utc="$(date --utc --iso-8601=ns)"
    start_ns="$(date +%s%N)"

    set +e
    /usr/bin/time \
        -f $'real_seconds=%e\nuser_seconds=%U\nsys_seconds=%S\ncpu_percent=%P\nmax_rss_kb=%M\nmajor_page_faults=%F\nminor_page_faults=%R\nvoluntary_context_switches=%w\ninvoluntary_context_switches=%c\ntime_exit_status=%x' \
        -o "${time_file}" \
        "${command[@]}" 2>&1 | tee "${console_file}"
    status="${PIPESTATUS[0]}"
    set -e

    end_ns="$(date +%s%N)"
    end_utc="$(date --utc --iso-8601=ns)"
    wall_seconds="$(awk -v start="${start_ns}" -v end="${end_ns}" 'BEGIN {printf "%.9f", (end-start)/1.0e9}')"

    local nx ny cells steps app_elapsed steps_per_s mcell
    nx="$(comparison_app_value nx "${console_file}")"
    ny="$(comparison_app_value ny "${console_file}")"
    steps="$(comparison_app_value steps "${console_file}")"
    app_elapsed="$(comparison_app_value elapsed_s "${console_file}")"
    steps_per_s="$(comparison_app_value steps_per_s "${console_file}")"
    mcell="$(comparison_app_value Mcell_updates_s "${console_file}")"

    # The normal GPU driver predates the common [TIMING] record, so parse its
    # equivalent human-readable fields when necessary.
    if [ -z "${nx}" ]; then
        nx="$(awk -F: '/^\[GPU\] nx:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "${console_file}")"
    fi
    if [ -z "${ny}" ]; then
        ny="$(awk -F: '/^\[GPU\] ny:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "${console_file}")"
    fi
    if [ -z "${steps}" ]; then
        steps="$(awk -F= '/^\[GPU\] Total steps=/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "${console_file}")"
    fi
    if [ -z "${app_elapsed}" ]; then
        app_elapsed="$(awk '/^[[:space:]]*Elapsed[[:space:]]*:/ {print $3; exit}' "${console_file}")"
    fi

    nx="${nx:-unknown}"
    ny="${ny:-unknown}"
    steps="${steps:-unknown}"
    app_elapsed="${app_elapsed:-unknown}"
    cells="unknown"
    if [[ "${nx}" =~ ^[0-9]+$ ]] && [[ "${ny}" =~ ^[0-9]+$ ]]; then
        cells=$((nx * ny))
    fi
    if [ -z "${steps_per_s}" ] && [[ "${steps}" =~ ^[0-9]+$ ]] && [[ "${app_elapsed}" =~ ^[0-9.eE+-]+$ ]]; then
        steps_per_s="$(awk -v steps="${steps}" -v elapsed="${app_elapsed}" 'BEGIN {if (elapsed > 0) printf "%.9f", steps/elapsed}')"
    fi
    if [ -z "${mcell}" ] && [[ "${cells}" =~ ^[0-9]+$ ]] && [[ "${steps}" =~ ^[0-9]+$ ]] && [[ "${app_elapsed}" =~ ^[0-9.eE+-]+$ ]]; then
        mcell="$(awk -v cells="${cells}" -v steps="${steps}" -v elapsed="${app_elapsed}" 'BEGIN {if (elapsed > 0) printf "%.9f", cells*steps/elapsed/1.0e6}')"
    fi

    local user_seconds sys_seconds cpu_percent rss major_faults minor_faults voluntary_cs involuntary_cs
    user_seconds="$(comparison_time_value user_seconds "${time_file}")"
    sys_seconds="$(comparison_time_value sys_seconds "${time_file}")"
    cpu_percent="$(comparison_time_value cpu_percent "${time_file}")"
    rss="$(comparison_time_value max_rss_kb "${time_file}")"
    major_faults="$(comparison_time_value major_page_faults "${time_file}")"
    minor_faults="$(comparison_time_value minor_page_faults "${time_file}")"
    voluntary_cs="$(comparison_time_value voluntary_context_switches "${time_file}")"
    involuntary_cs="$(comparison_time_value involuntary_context_switches "${time_file}")"

    echo "[RESULT] status=${status} app_elapsed_s=${app_elapsed} wall_seconds=${wall_seconds} steps=${steps} Mcell_updates_s=${mcell:-unknown}"

    # Append immediately after this run.  Each array task owns a distinct
    # RUN_ID/result directory and solver CSV, so no other task writes this
    # file and no lock or end-of-job accumulation is needed.
    printf '%s\n' \
        "${BACKEND},${CASE_NAME},${SOLVER_NAME},${N_SCALE},${repeat_index},${NUM_REPEATS},${BUILD_VARIANT},${THREADS_REPORTED:-0},${RANKS_REPORTED:-0},${nx},${ny},${cells},${steps},${app_elapsed},${steps_per_s:-unknown},${mcell:-unknown},${wall_seconds},${user_seconds:-unknown},${sys_seconds:-unknown},${cpu_percent:-unknown},${rss:-unknown},${major_faults:-unknown},${minor_faults:-unknown},${voluntary_cs:-unknown},${involuntary_cs:-unknown},${start_utc},${end_utc},${status},$(hostname),${GIT_BRANCH},${GIT_COMMIT}" \
        >> "${SUMMARY_FILE}"
    echo "[CSV] Appended completed run to ${SUMMARY_FILE}"

    # A failed/unstable solver is data too.  Record it and continue so that one
    # configuration does not discard the rest of the requested repetitions.
    if [ "${status}" -ne 0 ]; then
        echo "[WARN] Run failed; details are retained in ${console_file}."
        FAILED_RUNS=$((FAILED_RUNS + 1))
    fi
}

comparison_finish() {
    {
        echo "end_utc=$(date --utc --iso-8601=seconds)"
        echo "failed_runs=${FAILED_RUNS}"
    } >> "${METADATA_FILE}"

    echo ""
    echo "===== SUMMARY ====="
    cat "${SUMMARY_FILE}"
    echo "Metadata : ${METADATA_FILE}"
    echo "Results  : ${RESULT_DIR}"
    echo "Failures : ${FAILED_RUNS}"

    if [ "${FAILED_RUNS}" -ne 0 ]; then
        exit 1
    fi
}
