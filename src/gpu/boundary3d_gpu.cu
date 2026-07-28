#include "gpu/boundary3d_gpu.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {

void check_launch(const char* name) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(name) + ": " +
                                 cudaGetErrorString(err));
}

__device__ int low_source(int begin, int end, int layer, BoundaryType type) {
    return type == BoundaryType::Periodic ? end - 1 - layer : begin;
}

__device__ int high_source(int begin, int end, int layer, BoundaryType type) {
    return type == BoundaryType::Periodic ? begin + layer : end - 1;
}

__global__ void boundary_x_kernel(Grid3DGPUView g,
                                  BoundaryType low, BoundaryType high) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= g.total_ny() || k >= g.total_nz()) return;
    const int ib = g.i_begin();
    const int ie = ib + g.nx;
    for (int layer = 0; layer < g.ng; ++layer) {
        g.cells[g.flat_index(ib - 1 - layer, j, k)] =
            g.cells[g.flat_index(low_source(ib, ie, layer, low), j, k)];
        g.cells[g.flat_index(ie + layer, j, k)] =
            g.cells[g.flat_index(high_source(ib, ie, layer, high), j, k)];
    }
}

__global__ void boundary_y_kernel(Grid3DGPUView g,
                                  BoundaryType low, BoundaryType high) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= g.total_nx() || k >= g.total_nz()) return;
    const int jb = g.j_begin();
    const int je = jb + g.ny;
    for (int layer = 0; layer < g.ng; ++layer) {
        g.cells[g.flat_index(i, jb - 1 - layer, k)] =
            g.cells[g.flat_index(i, low_source(jb, je, layer, low), k)];
        g.cells[g.flat_index(i, je + layer, k)] =
            g.cells[g.flat_index(i, high_source(jb, je, layer, high), k)];
    }
}

__global__ void boundary_z_kernel(Grid3DGPUView g,
                                  BoundaryType low, BoundaryType high) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= g.total_nx() || j >= g.total_ny()) return;
    const int kb = g.k_begin();
    const int ke = kb + g.nz;
    for (int layer = 0; layer < g.ng; ++layer) {
        g.cells[g.flat_index(i, j, kb - 1 - layer)] =
            g.cells[g.flat_index(i, j, low_source(kb, ke, layer, low))];
        g.cells[g.flat_index(i, j, ke + layer)] =
            g.cells[g.flat_index(i, j, high_source(kb, ke, layer, high))];
    }
}

} // namespace

void apply_boundary_gpu_3d(Grid3DGPU& grid, const BoundaryConfig3D& bc) {
    const dim3 threads(16, 16);
    dim3 blocks(
        (grid.total_ny() + threads.x - 1) / threads.x,
        (grid.total_nz() + threads.y - 1) / threads.y);
    boundary_x_kernel<<<blocks, threads>>>(make_view(grid), bc.left, bc.right);
    check_launch("launch 3D x-boundary kernel");

    blocks = dim3(
        (grid.total_nx() + threads.x - 1) / threads.x,
        (grid.total_nz() + threads.y - 1) / threads.y);
    boundary_y_kernel<<<blocks, threads>>>(make_view(grid), bc.bottom, bc.top);
    check_launch("launch 3D y-boundary kernel");

    blocks = dim3(
        (grid.total_nx() + threads.x - 1) / threads.x,
        (grid.total_ny() + threads.y - 1) / threads.y);
    boundary_z_kernel<<<blocks, threads>>>(make_view(grid), bc.front, bc.back);
    check_launch("launch 3D z-boundary kernel");
}
