CXX    := g++
NVCC   := nvcc
MPICXX := mpicxx

# =========================
# Common flags
# =========================
# -ffp-contract=off / --fmad=false: forbid fusing a*b+c into a single
# rounding (FMA) on either side. nvcc contracts FMAs by default; g++ may
# too under some standard modes. Left on, host and device round the exact
# same riemann.hpp formula differently, and that
# 1-ULP difference is enough for CPU and GPU runs to diverge once the
# solver reaches a chaotic regime. Forcing strict per-operation rounding
# removes this whole class of CPU/GPU non-reproducibility.
CXXFLAGS_BASE     := -std=c++17 -O3 -Wall -Wextra -pedantic -Ihead -ffp-contract=off
NVCCFLAGS_BASE    := -std=c++17 -O3 -Ihead -Xcompiler="-Wall -Wextra" --fmad=false
# -lineinfo for source-level correlation in `ncu`; empty by default so it
# never affects normal builds. Override on the command line, e.g.:
#   make gpu NVCC_EXTRA_FLAGS=-lineinfo
NVCC_EXTRA_FLAGS  ?=
# MPI wrappers do not necessarily use GCC.  GNU/OpenMPI builds keep the
# default below; Intel classic builds can override this with -no-fma.
MPI_FP_FLAGS      ?= -ffp-contract=off
MPICXXFLAGS_BASE  := -std=c++17 -O3 -Wall -Wextra -pedantic -Ihead -DOMPI_SKIP_MPICXX $(MPI_FP_FLAGS)

# CUDA_ARCH := -arch=sm_90
CUDA_ARCH := -arch=sm_80
NVCCFLAGS_BASE += $(CUDA_ARCH)

# =========================
# OpenMP flags
# =========================
OMPFLAGS := -fopenmp

# CPU OpenMP build
CXXFLAGS := $(CXXFLAGS_BASE) $(OMPFLAGS)

# GPU build
NVCCFLAGS := $(NVCCFLAGS_BASE) $(NVCC_EXTRA_FLAGS)

# Pure MPI build: deliberately no OpenMP
MPICXXFLAGS := $(MPICXXFLAGS_BASE)

# Optional hybrid MPI + OpenMP build
MPICXXFLAGS_OMP := $(MPICXXFLAGS_BASE) $(OMPFLAGS)

# =========================
# Directories / targets
# =========================
BUILD_DIR := build
CPU_BUILD_DIR := $(BUILD_DIR)/cpu
GPU_BUILD_DIR := $(BUILD_DIR)/gpu
MPI_BUILD_DIR := $(BUILD_DIR)/mpi
MPI_3D_BUILD_DIR := $(BUILD_DIR)/mpi_3d
MPI_OMP_BUILD_DIR := $(BUILD_DIR)/mpi_omp
BIN_DIR := bin

CPU_TARGET := $(BIN_DIR)/main_cpu
CPU_SERIAL_TARGET := $(BIN_DIR)/main_cpu_serial
CPU_3D_TARGET := $(BIN_DIR)/main_cpu_3d
GPU_TARGET := $(BIN_DIR)/main_gpu
GPU_3D_TARGET := $(BIN_DIR)/main_gpu_3d
MPI_TARGET := $(BIN_DIR)/main_mpi
MPI_3D_TARGET := $(BIN_DIR)/main_mpi_3d
MPI_OMP_TARGET := $(BIN_DIR)/main_mpi_omp

CPU_MAIN := scripts/cpu/main_cpu.cpp
CPU_3D_MAIN := scripts/cpu/main_cpu_3d.cpp
GPU_MAIN := scripts/gpu/main_gpu.cu
GPU_3D_MAIN := scripts/gpu/main_gpu_3d.cu
MPI_MAIN := scripts/cpu/main_mpi.cpp
MPI_3D_MAIN := scripts/cpu/main_mpi_3d.cpp

CPU_OBJS := \
	$(CPU_BUILD_DIR)/main_cpu.o \
	$(CPU_BUILD_DIR)/test_cases.o \
	$(CPU_BUILD_DIR)/init.o \
	$(CPU_BUILD_DIR)/solver_cpu.o

CPU_SERIAL_OBJS := \
	$(CPU_BUILD_DIR)/main_cpu_serial.o \
	$(CPU_BUILD_DIR)/test_cases_serial.o \
	$(CPU_BUILD_DIR)/init_serial.o \
	$(CPU_BUILD_DIR)/solver_cpu_serial.o

CPU_3D_OBJS := \
	$(CPU_BUILD_DIR)/main_cpu_3d.o \
	$(CPU_BUILD_DIR)/solver3d_cpu.o

CPU_3D_TEST_TARGET := $(BIN_DIR)/test_solver3d

GPU_OBJS := \
	$(GPU_BUILD_DIR)/main_gpu.o \
	$(GPU_BUILD_DIR)/test_cases.o \
	$(GPU_BUILD_DIR)/init.o \
	$(GPU_BUILD_DIR)/solver_gpu.o \
	$(GPU_BUILD_DIR)/boundary_gpu.o

GPU_3D_OBJS := \
	$(GPU_BUILD_DIR)/main_gpu_3d.o \
	$(GPU_BUILD_DIR)/solver3d_gpu.o \
	$(GPU_BUILD_DIR)/boundary3d_gpu.o

GPU_3D_TEST_TARGET := $(BIN_DIR)/test_solver3d_gpu

# Pure MPI: do not link solver_cpu.o
MPI_OBJS := \
	$(MPI_BUILD_DIR)/main_mpi.o \
	$(MPI_BUILD_DIR)/test_cases.o \
	$(MPI_BUILD_DIR)/init.o \
	$(MPI_BUILD_DIR)/solver_mpi.o

MPI_3D_OBJS := \
	$(MPI_3D_BUILD_DIR)/main_mpi_3d.o \
	$(MPI_3D_BUILD_DIR)/solver3d_cpu.o

# Optional hybrid target: still only links solver_mpi.o.
# Use this only if solver_mpi.cpp itself contains OpenMP pragmas later.
MPI_OMP_OBJS := \
	$(MPI_OMP_BUILD_DIR)/main_mpi.o \
	$(MPI_OMP_BUILD_DIR)/test_cases.o \
	$(MPI_OMP_BUILD_DIR)/init.o \
	$(MPI_OMP_BUILD_DIR)/solver_mpi.o

# Safer default: do not build GPU unless explicitly requested
.PHONY: all
all: cpu cpu_serial mpi

# =========================
# CPU with OpenMP
# =========================
.PHONY: cpu
cpu: $(CPU_TARGET)

$(CPU_TARGET): $(CPU_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(CPU_OBJS) -o $@ -lstdc++fs

$(CPU_BUILD_DIR)/main_cpu.o: $(CPU_MAIN)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CPU_BUILD_DIR)/test_cases.o: src/test_cases.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CPU_BUILD_DIR)/init.o: src/init.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CPU_BUILD_DIR)/solver_cpu.o: src/cpu/solver_cpu.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# =========================
# CPU serial baseline
# =========================
.PHONY: cpu_serial
cpu_serial: $(CPU_SERIAL_TARGET)

$(CPU_SERIAL_TARGET): $(CPU_SERIAL_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS_BASE) $(CPU_SERIAL_OBJS) -o $@ -lstdc++fs

$(CPU_BUILD_DIR)/main_cpu_serial.o: $(CPU_MAIN)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS_BASE) -c $< -o $@

$(CPU_BUILD_DIR)/test_cases_serial.o: src/test_cases.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS_BASE) -c $< -o $@

$(CPU_BUILD_DIR)/init_serial.o: src/init.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS_BASE) -c $< -o $@

$(CPU_BUILD_DIR)/solver_cpu_serial.o: src/cpu/solver_cpu.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS_BASE) -c $< -o $@

# =========================
# 3D CPU baseline
# =========================
.PHONY: cpu_3d
cpu_3d: $(CPU_3D_TARGET)

$(CPU_3D_TARGET): $(CPU_3D_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(CPU_3D_OBJS) -o $@ -lstdc++fs

$(CPU_BUILD_DIR)/main_cpu_3d.o: $(CPU_3D_MAIN) head/blast3d_case.hpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CPU_BUILD_DIR)/solver3d_cpu.o: src/cpu/solver3d_cpu.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: test_3d
test_3d: $(CPU_3D_TEST_TARGET)
	$(CPU_3D_TEST_TARGET)

$(CPU_3D_TEST_TARGET): validation/test_solver3d.cpp $(CPU_BUILD_DIR)/solver3d_cpu.o
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lstdc++fs

# =========================
# Pure MPI
# =========================
.PHONY: mpi
mpi: $(MPI_TARGET)

$(MPI_TARGET): $(MPI_OBJS)
	@mkdir -p $(BIN_DIR)
	$(MPICXX) $(MPICXXFLAGS) $(MPI_OBJS) -o $@ -lstdc++fs

$(MPI_BUILD_DIR)/main_mpi.o: $(MPI_MAIN)
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS) -c $< -o $@

$(MPI_BUILD_DIR)/test_cases.o: src/test_cases.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS) -c $< -o $@

$(MPI_BUILD_DIR)/init.o: src/init.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS) -c $< -o $@

$(MPI_BUILD_DIR)/solver_mpi.o: src/cpu/solver_mpi.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS) -c $< -o $@

# =========================
# Pure MPI 3D z-slab decomposition
# =========================
.PHONY: mpi_3d
mpi_3d: $(MPI_3D_TARGET)

$(MPI_3D_TARGET): $(MPI_3D_OBJS)
	@mkdir -p $(BIN_DIR)
	$(MPICXX) $(MPICXXFLAGS) $(MPI_3D_OBJS) -o $@ -lstdc++fs

$(MPI_3D_BUILD_DIR)/main_mpi_3d.o: $(MPI_3D_MAIN) head/blast3d_case.hpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS) -c $< -o $@

$(MPI_3D_BUILD_DIR)/solver3d_cpu.o: src/cpu/solver3d_cpu.cpp head/cpu/solver3d_cpu.hpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS) -c $< -o $@

# =========================
# Optional MPI + OpenMP build
# =========================
.PHONY: mpi_omp
mpi_omp: $(MPI_OMP_TARGET)

$(MPI_OMP_TARGET): $(MPI_OMP_OBJS)
	@mkdir -p $(BIN_DIR)
	$(MPICXX) $(MPICXXFLAGS_OMP) $(MPI_OMP_OBJS) -o $@ -lstdc++fs

$(MPI_OMP_BUILD_DIR)/main_mpi.o: $(MPI_MAIN)
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS_OMP) -c $< -o $@

$(MPI_OMP_BUILD_DIR)/test_cases.o: src/test_cases.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS_OMP) -c $< -o $@

$(MPI_OMP_BUILD_DIR)/init.o: src/init.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS_OMP) -c $< -o $@

$(MPI_OMP_BUILD_DIR)/solver_mpi.o: src/cpu/solver_mpi.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS_OMP) -c $< -o $@

# =========================
# GPU
# =========================
.PHONY: gpu
gpu: $(GPU_TARGET)

$(GPU_TARGET): $(GPU_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(GPU_OBJS) -o $@ -lstdc++fs

$(GPU_BUILD_DIR)/main_gpu.o: $(GPU_MAIN)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(GPU_BUILD_DIR)/test_cases.o: src/test_cases.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -x c++ -c $< -o $@

$(GPU_BUILD_DIR)/init.o: src/init.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -x c++ -c $< -o $@

$(GPU_BUILD_DIR)/solver_gpu.o: src/gpu/solver_gpu.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(GPU_BUILD_DIR)/boundary_gpu.o: src/gpu/boundary_gpu.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# =========================
# 3D GPU baseline
# =========================
.PHONY: gpu_3d
gpu_3d: $(GPU_3D_TARGET)

$(GPU_3D_TARGET): $(GPU_3D_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(GPU_3D_OBJS) -o $@ -lstdc++fs

$(GPU_BUILD_DIR)/main_gpu_3d.o: $(GPU_3D_MAIN) head/blast3d_case.hpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(GPU_BUILD_DIR)/solver3d_gpu.o: src/gpu/solver3d_gpu.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(GPU_BUILD_DIR)/boundary3d_gpu.o: src/gpu/boundary3d_gpu.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(GPU_BUILD_DIR)/solver3d_cpu_reference.o: src/cpu/solver3d_cpu.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS_BASE) -c $< -o $@

.PHONY: test_gpu_3d
test_gpu_3d: $(GPU_3D_TEST_TARGET)
	$(GPU_3D_TEST_TARGET)

$(GPU_3D_TEST_TARGET): validation/test_solver3d_gpu.cu \
		$(GPU_BUILD_DIR)/solver3d_gpu.o \
		$(GPU_BUILD_DIR)/boundary3d_gpu.o \
		$(GPU_BUILD_DIR)/solver3d_cpu_reference.o
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lstdc++fs

# =========================
# Run helpers
# =========================
.PHONY: run_cpu
run_cpu: $(CPU_TARGET)
	$(CPU_TARGET)

.PHONY: run_cpu_serial
run_cpu_serial: $(CPU_SERIAL_TARGET)
	$(CPU_SERIAL_TARGET)

.PHONY: run_cpu_omp
run_cpu_omp: $(CPU_TARGET)
	OMP_NUM_THREADS=8 OMP_PROC_BIND=true OMP_PLACES=cores $(CPU_TARGET)

.PHONY: run_gpu
run_gpu: $(GPU_TARGET)
	$(GPU_TARGET)

.PHONY: run_mpi
run_mpi: $(MPI_TARGET)
	OMP_NUM_THREADS=1 mpirun -np 4 $(MPI_TARGET) 1 --timing-only

.PHONY: run_mpi_output
run_mpi_output: $(MPI_TARGET)
	OMP_NUM_THREADS=1 mpirun -np 4 $(MPI_TARGET) 1 --output

.PHONY: run_mpi_omp
run_mpi_omp: $(MPI_OMP_TARGET)
	OMP_NUM_THREADS=4 OMP_PROC_BIND=true OMP_PLACES=cores mpirun -np 4 $(MPI_OMP_TARGET) 1 --timing-only

# =========================
# Clean
# =========================
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
