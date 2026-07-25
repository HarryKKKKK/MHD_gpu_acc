#!/bin/bash -l
#SBATCH -J mhd3d_cmp
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Compare the true 3D OpenMP, pure-MPI and CUDA implementations on the same
# CSD3 Ampere node. The moderate blast and weakly compressible IMTG cases are
# each run exactly once per backend. Volume output and rendering are disabled
# so the reported elapsed times measure the solver rather than filesystem I/O.
#
# Resource comparison:
#   OpenMP : OMP_THREADS CPU cores (default 8)
#   MPI    : MPI_RANKS ranks x 1 CPU core (default 8)
#   GPU    : one Ampere GPU
#
# Default pre-run (64^3):
#
#   mkdir -p logs
#   sbatch scripts/csd3_slurm/slurm_compare_3d_backends.sh
#
# Final 128^3 comparison:
#
#   sbatch --export=ALL,RESOLUTION=128 \
#     scripts/csd3_slurm/slurm_compare_3d_backends.sh
#
# Short smoke test:
#
#   sbatch --export=ALL,RESOLUTION=32,BLAST_T_END=0.02,IMTG_T_END=0.25 \
#     scripts/csd3_slurm/slurm_compare_3d_backends.sh
#
# Common overrides:
#   RESOLUTION=64
#   OMP_THREADS=8
#   MPI_RANKS=8                 RESOLUTION must be divisible by MPI_RANKS
#   SOLVER=hlld
#   CFL=0.20
#   SNAPSHOTS=5                 timestep target intervals; no files are written
#   BLAST_T_END=0.10
#   IMTG_T_END=5.809475019311126
#   MAKE_JOBS=8
#   RESULT_DIR=timing/compare3d_JOBID
#
# Results:
#   backend_times.csv           one measured elapsed time per backend/case
#   gpu_speedup_summary.csv     GPU speedups, time ratios, and saved percentages
#   blast_{omp,mpi,gpu}.log
#   imtg_{omp,mpi,gpu}.log
#
# Definitions:
#   GPU speedup vs CPU backend = T_CPU / T_GPU
#   GPU time percentage        = 100 * T_GPU / T_CPU
#   GPU time saved percentage  = 100 * (1 - T_GPU / T_CPU)
#   GPU share of all 3 times   = 100*T_GPU/(T_OMP+T_MPI+T_GPU)

set -euo pipefail

JOB_ID="${SLURM_JOB_ID:-manual}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORKDIR="${WORKDIR:-${SUBMIT_DIR}}"
cd "${WORKDIR}"

if [[ ! -f Makefile ]]; then
    echo "[ERROR] Submit from the MHD repository root or set WORKDIR."
    exit 1
fi
mkdir -p logs timing

echo "===== MODULE SETUP ====="
if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash \
                /usr/local/Modules/init/bash; do
        if [[ -f "${init}" ]]; then source "${init}"; break; fi
    done
fi
if ! command -v module >/dev/null 2>&1; then
    echo "[ERROR] Environment Modules is unavailable."
    exit 1
fi
module purge
module load rhel8/default-amp
module list 2>&1 || true

for tool in g++ mpicxx mpirun nvcc nvidia-smi make awk; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "[ERROR] ${tool} is unavailable after loading rhel8/default-amp."
        exit 1
    fi
done

RESOLUTION="${RESOLUTION:-64}"
OMP_THREADS="${OMP_THREADS:-8}"
MPI_RANKS="${MPI_RANKS:-8}"
SOLVER="${SOLVER:-hlld}"
CFL="${CFL:-0.20}"
SNAPSHOTS="${SNAPSHOTS:-5}"
BLAST_T_END="${BLAST_T_END:-0.10}"
IMTG_T_END="${IMTG_T_END:-5.809475019311126}"
MAKE_JOBS="${MAKE_JOBS:-8}"
RESULT_DIR="${RESULT_DIR:-timing/compare3d_${JOB_ID}}"
BUILD_ROOT="${BUILD_ROOT:-build/csd3_compare3d_${JOB_ID}}"
BIN_ROOT="${BIN_ROOT:-bin/csd3_compare3d_${JOB_ID}}"

for integer_value in RESOLUTION OMP_THREADS MPI_RANKS SNAPSHOTS; do
    value="${!integer_value}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] ${integer_value} must be a positive integer."
        exit 2
    fi
done
if (( RESOLUTION < 8 )); then
    echo "[ERROR] RESOLUTION must be at least 8."
    exit 2
fi
if (( RESOLUTION % MPI_RANKS != 0 )); then
    echo "[ERROR] RESOLUTION=${RESOLUTION} must be divisible by MPI_RANKS=${MPI_RANKS}."
    exit 2
fi
if (( RESOLUTION / MPI_RANKS < 2 )); then
    echo "[ERROR] Each MPI rank needs at least two active z planes."
    exit 2
fi
ALLOCATED_CPUS=$(( ${SLURM_NTASKS:-8} * ${SLURM_CPUS_PER_TASK:-1} ))
if (( OMP_THREADS > ALLOCATED_CPUS || MPI_RANKS > ALLOCATED_CPUS )); then
    echo "[ERROR] Requested OMP_THREADS/MPI_RANKS exceeds ${ALLOCATED_CPUS} allocated CPUs."
    exit 2
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader |
    head -1 | xargs)"
COMPUTE_CAP="$(
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
    head -1 | tr -d '[:space:]' || true
)"
if [[ "${COMPUTE_CAP}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    CUDA_SM="${COMPUTE_CAP/./}"
elif [[ "${GPU_NAME}" == *A100* || "${GPU_NAME}" == *A30* ]]; then
    CUDA_SM=80
elif [[ "${GPU_NAME}" == *A10* || "${GPU_NAME}" == *A40* ]]; then
    CUDA_SM=86
else
    CUDA_SM="${CUDA_SM:-80}"
fi
CUDA_ARCH_FLAG="-arch=sm_${CUDA_SM}"

mkdir -p "${RESULT_DIR}"
TIMES_CSV="${RESULT_DIR}/backend_times.csv"
SUMMARY_CSV="${RESULT_DIR}/gpu_speedup_summary.csv"
echo "case,backend,resolution,workers,elapsed_s" > "${TIMES_CSV}"
echo "case,omp_s,mpi_s,gpu_s,gpu_speedup_vs_omp,gpu_speedup_vs_mpi,gpu_time_pct_of_omp,gpu_time_pct_of_mpi,gpu_time_saved_pct_vs_omp,gpu_time_saved_pct_vs_mpi,gpu_share_of_all_three_pct" > "${SUMMARY_CSV}"

echo "===== 3D BACKEND COMPARISON ====="
echo "Job ID       : ${JOB_ID}"
echo "Host         : $(hostname)"
echo "Start        : $(date --iso-8601=seconds)"
echo "GPU          : ${GPU_NAME}"
echo "Resolution   : ${RESOLUTION}^3"
echo "OMP threads  : ${OMP_THREADS}"
echo "MPI ranks    : ${MPI_RANKS}"
echo "Solver / CFL : ${SOLVER} / ${CFL}"
echo "Snapshots    : ${SNAPSHOTS} intervals (timing only)"
echo "Blast t_end  : ${BLAST_T_END}"
echo "IMTG t_end   : ${IMTG_T_END}"
echo "Results      : ${RESULT_DIR}"

echo "===== BUILD ALL TRUE-3D BACKENDS ====="
make -j "${MAKE_JOBS}" cpu_3d mpi_3d gpu_3d \
    BUILD_DIR="${BUILD_ROOT}" \
    BIN_DIR="${BIN_ROOT}" \
    CUDA_ARCH="${CUDA_ARCH_FLAG}"

OMP_BIN="${BIN_ROOT}/main_cpu_3d"
MPI_BIN="${BIN_ROOT}/main_mpi_3d"
GPU_BIN="${BIN_ROOT}/main_gpu_3d"
for binary in "${OMP_BIN}" "${MPI_BIN}" "${GPU_BIN}"; do
    if [[ ! -x "${binary}" ]]; then
        echo "[ERROR] Expected executable was not produced: ${binary}"
        exit 1
    fi
done

extract_elapsed() {
    local marker="$1"
    local logfile="$2"
    awk -v marker="${marker}" '
        $1 == marker {
            for(i=1;i<=NF;i++) {
                split($i,pair,"=")
                if(pair[1]=="elapsed_s") value=pair[2]
            }
        }
        END {
            if(value=="" || value+0<=0) exit 1
            print value
        }
    ' "${logfile}"
}

run_case() {
    local case_name="$1"
    local t_end="$2"
    local common_args=(
        --case "${case_name}"
        --resolution "${RESOLUTION}"
        --t-end "${t_end}"
        --snapshots "${SNAPSHOTS}"
        --solver "${SOLVER}"
        --cfl "${CFL}"
        --no-out
    )
    local omp_log="${RESULT_DIR}/${case_name}_omp.log"
    local mpi_log="${RESULT_DIR}/${case_name}_mpi.log"
    local gpu_log="${RESULT_DIR}/${case_name}_gpu.log"

    echo
    echo "===== CASE ${case_name}: OPENMP (${OMP_THREADS} threads) ====="
    OMP_NUM_THREADS="${OMP_THREADS}" OMP_PROC_BIND=close OMP_PLACES=cores \
        srun --exclusive --ntasks=1 --cpus-per-task="${OMP_THREADS}" \
        "${OMP_BIN}" "${common_args[@]}" 2>&1 | tee "${omp_log}"

    echo
    echo "===== CASE ${case_name}: MPI (${MPI_RANKS} ranks) ====="
    OMP_NUM_THREADS=1 \
        mpirun -np "${MPI_RANKS}" --bind-to core \
        "${MPI_BIN}" "${common_args[@]}" 2>&1 | tee "${mpi_log}"

    echo
    echo "===== CASE ${case_name}: GPU (${GPU_NAME}) ====="
    srun --exclusive --ntasks=1 --cpus-per-task=1 --gres=gpu:1 \
        "${GPU_BIN}" "${common_args[@]}" 2>&1 | tee "${gpu_log}"

    local omp_s mpi_s gpu_s
    omp_s="$(extract_elapsed "[CPU3D]" "${omp_log}")"
    mpi_s="$(extract_elapsed "[MPI3D]" "${mpi_log}")"
    gpu_s="$(extract_elapsed "[GPU3D]" "${gpu_log}")"

    echo "${case_name},openmp,${RESOLUTION},${OMP_THREADS},${omp_s}" >> "${TIMES_CSV}"
    echo "${case_name},mpi,${RESOLUTION},${MPI_RANKS},${mpi_s}" >> "${TIMES_CSV}"
    echo "${case_name},gpu,${RESOLUTION},1,${gpu_s}" >> "${TIMES_CSV}"

    awk -v case_name="${case_name}" \
        -v omp="${omp_s}" -v mpi="${mpi_s}" -v gpu="${gpu_s}" '
        BEGIN {
            printf "%s,%.9g,%.9g,%.9g,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                case_name,omp,mpi,gpu,
                omp/gpu,mpi/gpu,
                100.0*gpu/omp,100.0*gpu/mpi,
                100.0*(1.0-gpu/omp),100.0*(1.0-gpu/mpi),
                100.0*gpu/(omp+mpi+gpu)
        }
    ' >> "${SUMMARY_CSV}"

    echo
    echo "----- ${case_name} GPU SPEEDUP -----"
    awk -v omp="${omp_s}" -v mpi="${mpi_s}" -v gpu="${gpu_s}" '
        BEGIN {
            printf "OpenMP: %.6f s\nMPI   : %.6f s\nGPU   : %.6f s\n",omp,mpi,gpu
            printf "GPU speedup vs OpenMP : %.3fx\n",omp/gpu
            printf "GPU speedup vs MPI    : %.3fx\n",mpi/gpu
            printf "GPU time / OpenMP time: %.2f%% (saved %.2f%%)\n",
                   100.0*gpu/omp,100.0*(1.0-gpu/omp)
            printf "GPU time / MPI time   : %.2f%% (saved %.2f%%)\n",
                   100.0*gpu/mpi,100.0*(1.0-gpu/mpi)
        }
    '
}

run_case blast "${BLAST_T_END}"
run_case imtg "${IMTG_T_END}"

echo
echo "===== FINAL GPU SPEEDUP SUMMARY ====="
if command -v column >/dev/null 2>&1; then
    column -s, -t "${SUMMARY_CSV}"
else
    cat "${SUMMARY_CSV}"
fi
echo
echo "Completed: $(date --iso-8601=seconds)"
echo "Timing CSV : ${WORKDIR}/${TIMES_CSV}"
echo "Summary CSV: ${WORKDIR}/${SUMMARY_CSV}"
