#pragma once

#include <cstddef>

#include "cpu/boundary3d_cpu.hpp"
#include "gpu/grid3d_gpu.cuh"
#include "riemann.hpp"

struct GpuWorkspace3D {
    int nx=0,ny=0,nz=0;
    double* block_max=nullptr;
    double* max_speed=nullptr;
    void* reduce_tmp=nullptr;
    std::size_t reduce_tmp_bytes=0;
};

void set_gpu3d_physics_gamma(double gamma);
void init_gpu_workspace(GpuWorkspace3D&,const Grid3DGPU&);
void free_gpu_workspace(GpuWorkspace3D&);
double compute_dt_gpu(const Grid3DGPU&,GpuWorkspace3D&,double cfl);

void advance_gpu(const Grid3DGPU& old,Grid3DGPU& ux,Grid3DGPU& uy,
                 Grid3DGPU& out,GpuWorkspace3D&,double dt,
                 RiemannSolver,const BoundaryConfig3D&);
