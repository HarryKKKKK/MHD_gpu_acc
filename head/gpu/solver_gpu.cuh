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
// Time step (also sets device-side ch_glm = max signal speed).
//
// out_max_speed, if non-null, receives the max signal speed computed
// this step *before* ch_glm is updated (diagnostic use only — passing
// nullptr, the default, reproduces the exact prior signature/behaviour).
// Note this is set even when max_speed <= 0.0 (the case where the
// function falls back to returning a huge dt without touching ch_glm at
// all), so a diagnostic reader can detect that otherwise-silent path.
// ============================================================
double compute_dt_gpu(const Grid2DGPU& grid, GpuWorkspace& ws, double cfl,
                       double* out_max_speed = nullptr);

// ============================================================
// Diagnostic-only counter: how many times safe_prim/safe_cons rejected a
// reconstructed candidate (is_physical() failed) since the last reset.
// The device-side atomicAdd that feeds this always runs (cheap: only the
// rejected-candidate branch pays for it), so it never changes which
// value safe_prim/safe_cons returns. Meant to be reset once and read
// once per step, only when the driver's --diag-interval is active.
// ============================================================
void reset_floor_trigger_count_gpu();
unsigned long long read_floor_trigger_count_gpu();

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
