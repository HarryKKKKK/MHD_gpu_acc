#pragma once

#include "gpu/grid_gpu.cuh"
#include "riemann.hpp"
#include "test_cases.hpp"   // BoundaryConfig

#include <cstddef>

struct GpuWorkspace {
    int nx = 0;
    int ny = 0;

    // Workspace for compute_dt_gpu (per-block max wave speeds + CUB reduction).
    double* speed_d     = nullptr;
    double* max_speed_d = nullptr;

    void*       reduce_tmp       = nullptr;
    std::size_t reduce_tmp_bytes = 0;
};

// ============================================================
// Device-side physics constants.
// Must be called once before the time loop:
//   set_gpu_physics_gamma(cfg.gamma);
// Must be called inside compute_dt_gpu (automatically), but can
// also be called manually to prime ch_glm before the first step.
// ============================================================
void set_gpu_physics_gamma(double gamma);
void set_gpu_physics_ch(double ch);

// ============================================================
// Workspace lifecycle
// ============================================================
void init_gpu_workspace(GpuWorkspace& ws, const Grid2DGPU& grid);
void free_gpu_workspace(GpuWorkspace& ws);

// ============================================================
// Time step (also sets device-side ch_glm = max signal speed)
// ============================================================
double compute_dt_gpu(const Grid2DGPU& grid, GpuWorkspace& ws, double cfl);

// ============================================================
// Advance — second-order MUSCL-Hancock (x-then-y splitting)
// ============================================================
void advance_second_order_gpu(
    const Grid2DGPU& Uold,
    Grid2DGPU&       Utmp,
    Grid2DGPU&       Unew,
    GpuWorkspace&    ws,
    double           dt,
    RiemannSolver    solver,
    const BoundaryConfig& bc
);
