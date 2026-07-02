#pragma once

#include "gpu/grid_gpu.cuh"
#include "riemann.hpp"
#include "test_cases.hpp"   // BoundaryConfig

#include <cstddef>

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

double compute_dt_gpu(const Grid2DGPU& grid, GpuWorkspace& ws, double cfl,
                       double* out_max_speed = nullptr);

void reset_floor_trigger_count_gpu();
unsigned long long read_floor_trigger_count_gpu();

void advance_second_order_gpu(
    const Grid2DGPU& Uold,
    Grid2DGPU&       Utmp,
    Grid2DGPU&       Unew,
    GpuWorkspace&    ws,
    double           dt,
    RiemannSolver    solver,
    const BoundaryConfig& bc
);
