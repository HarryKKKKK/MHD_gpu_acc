#pragma once

#include "gpu/grid_gpu.cuh"
#include "riemann.hpp"
#include "test_cases.hpp"   // BoundaryConfig

#include <cstddef>

#ifndef MHD_ADVANCE_X_BLOCK_X
#define MHD_ADVANCE_X_BLOCK_X 16
#endif

#ifndef MHD_ADVANCE_X_BLOCK_Y
#define MHD_ADVANCE_X_BLOCK_Y 8
#endif

#ifndef MHD_ADVANCE_Y_BLOCK_X
#define MHD_ADVANCE_Y_BLOCK_X 16
#endif

#ifndef MHD_ADVANCE_Y_BLOCK_Y
#define MHD_ADVANCE_Y_BLOCK_Y 8
#endif

#ifndef MHD_ADVANCE_X_MIN_BLOCKS_PER_SM
#define MHD_ADVANCE_X_MIN_BLOCKS_PER_SM 3
#endif

#ifndef MHD_ADVANCE_Y_MIN_BLOCKS_PER_SM
#define MHD_ADVANCE_Y_MIN_BLOCKS_PER_SM 0
#endif

struct GpuLaunchConfig {
    int x_block_x;
    int x_block_y;
    int x_min_blocks_per_sm;
    int y_block_x;
    int y_block_y;
    int y_min_blocks_per_sm;
};

struct GpuAdvanceTimings {
    double      x_ms = 0.0;
    double      y_ms = 0.0;
    std::size_t samples = 0;
};

struct GpuWorkspace {
    int nx = 0;
    int ny = 0;

    double* speed_d     = nullptr;
    double* max_speed_d = nullptr;

    void*       reduce_tmp       = nullptr;
    std::size_t reduce_tmp_bytes = 0;
};

void set_gpu_physics_gamma(double gamma);
void set_gpu_physics_ch(double ch);

void init_gpu_workspace(GpuWorkspace& ws, const Grid2DGPU& grid);
void free_gpu_workspace(GpuWorkspace& ws);

double compute_dt_gpu(const Grid2DGPU& grid, GpuWorkspace& ws, double cfl);

GpuLaunchConfig get_gpu_launch_config();

void advance_gpu(
    const Grid2DGPU& Uold,
    Grid2DGPU&       Utmp,
    Grid2DGPU&       Unew,
    GpuWorkspace&    ws,
    double           dt,
    RiemannSolver    solver,
    const BoundaryConfig& bc,
    GpuAdvanceTimings* timings = nullptr
);
