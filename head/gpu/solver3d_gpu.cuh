#pragma once

#include <cstddef>

#include "gpu/boundary3d_gpu.cuh"
#include "gpu/grid3d_gpu.cuh"
#include "riemann.hpp"

struct GpuWorkspace3D {
    int nx = 0, ny = 0, nz = 0;
    double* speeds = nullptr;
    double* max_speed = nullptr;
    void* reduce_tmp = nullptr;
    std::size_t reduce_tmp_bytes = 0;
};

void set_gpu3d_physics_gamma(double gamma);
void set_gpu3d_physics_ch(double ch);
void init_gpu_workspace_3d(GpuWorkspace3D& ws, const Grid3DGPU& grid);
void free_gpu_workspace_3d(GpuWorkspace3D& ws);
double compute_dt_gpu_3d(const Grid3DGPU& grid, GpuWorkspace3D& ws, double cfl);

void advance_gpu_3d(const Grid3DGPU& old,
                    Grid3DGPU& after_x,
                    Grid3DGPU& after_y,
                    Grid3DGPU& next,
                    GpuWorkspace3D& ws,
                    double dt,
                    RiemannSolver solver,
                    const BoundaryConfig3D& bc);
