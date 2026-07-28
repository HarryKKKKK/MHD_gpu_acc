#include "gpu/solver3d_gpu.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "physics.hpp"

namespace {

constexpr int kDtThreads = 256;
constexpr double kRhoFloor = 1.0e-12;
constexpr double kPFloor = 1.0e-12;
enum Axis { X = 0, Y = 1, Z = 2 };

void cuda_check(cudaError_t err, const char* operation) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(err));
}

__device__ double minmod_scalar(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return a > 0.0 ? fmin(a, b) : fmax(a, b);
}

__device__ Primitive minmod_primitive(const Primitive& a, const Primitive& b) {
    return Primitive(
        minmod_scalar(a.rho, b.rho),
        minmod_scalar(a.u, b.u),
        minmod_scalar(a.v, b.v),
        minmod_scalar(a.w, b.w),
        minmod_scalar(a.Bx, b.Bx),
        minmod_scalar(a.By, b.By),
        minmod_scalar(a.Bz, b.Bz),
        minmod_scalar(a.p, b.p),
        minmod_scalar(a.psi, b.psi));
}

__device__ bool physical(const Primitive& v) {
    return v.rho > kRhoFloor && v.p > kPFloor &&
           isfinite(v.rho) && isfinite(v.p) &&
           isfinite(v.u) && isfinite(v.v) && isfinite(v.w) &&
           isfinite(v.Bx) && isfinite(v.By) && isfinite(v.Bz) &&
           isfinite(v.psi);
}

__device__ Conserved enforce_physical(const Conserved& candidate,
                                      const Conserved& fallback) {
    return physical(phys::cons_to_prim(candidate)) ? candidate : fallback;
}

__device__ void shift(int axis, int offset, int& i, int& j, int& k) {
    if (axis == X) i += offset;
    else if (axis == Y) j += offset;
    else k += offset;
}

__device__ Conserved axis_flux(const Conserved& u, int axis, double ch) {
    if (axis == X) return phys::flux_x(u, ch);
    if (axis == Y) return phys::flux_y(u, ch);
    return phys::flux_z(u, ch);
}

__device__ Conserved swap_xz(const Conserved& u) {
    return Conserved(u.rho, u.rhow, u.rhov, u.rhou,
                     u.Bz, u.By, u.Bx, u.E, u.psi);
}

__device__ Conserved riemann_axis(const Conserved& left,
                                  const Conserved& right,
                                  int axis,
                                  RiemannSolver solver) {
    if (axis == X)
        return riemann_flux(left, right, Direction::X, solver);
    if (axis == Y)
        return riemann_flux(left, right, Direction::Y, solver);
    return swap_xz(riemann_flux(
        swap_xz(left), swap_xz(right), Direction::X, solver));
}

__device__ void reconstruct(ConstGrid3DGPUView q,
                            int i, int j, int k,
                            int axis, double dt_over_d,
                            Conserved& left, Conserved& right) {
    int im = i, jm = j, km = k;
    int ip = i, jp = j, kp = k;
    shift(axis, -1, im, jm, km);
    shift(axis, 1, ip, jp, kp);

    const Conserved um = q.cells[q.flat_index(im, jm, km)];
    const Conserved uc = q.cells[q.flat_index(i, j, k)];
    const Conserved up = q.cells[q.flat_index(ip, jp, kp)];
    const Primitive wm = phys::cons_to_prim(um);
    const Primitive wc = phys::cons_to_prim(uc);
    const Primitive wp = phys::cons_to_prim(up);
    const Primitive slope = minmod_primitive(wc - wm, wp - wc);
    const Primitive wl_candidate = wc - 0.5 * slope;
    const Primitive wr_candidate = wc + 0.5 * slope;
    const Primitive wl = physical(wl_candidate) ? wl_candidate : wc;
    const Primitive wr = physical(wr_candidate) ? wr_candidate : wc;
    const Conserved ul = phys::prim_to_cons(wl);
    const Conserved ur = phys::prim_to_cons(wr);
    const Conserved half = 0.5 * dt_over_d *
        (axis_flux(ur, axis, 0.0) - axis_flux(ul, axis, 0.0));
    left = enforce_physical(ul - half, ul);
    right = enforce_physical(ur - half, ur);
}

// Baseline design: one thread updates one cell and reads all stencil values
// directly from global memory. No shared-memory tile, face cache, fusion
// across directions, launch-bounds tuning, or solver specialization.
__global__ void advance_axis_kernel(ConstGrid3DGPUView in,
                                    Grid3DGPUView out,
                                    double dt_over_d,
                                    int axis,
                                    RiemannSolver solver) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    const int lj = blockIdx.y * blockDim.y + threadIdx.y;
    const int lk = blockIdx.z * blockDim.z + threadIdx.z;
    if (li >= in.nx || lj >= in.ny || lk >= in.nz) return;
    const int i = in.i_begin() + li;
    const int j = in.j_begin() + lj;
    const int k = in.k_begin() + lk;

    int im = i, jm = j, km = k;
    int ip = i, jp = j, kp = k;
    shift(axis, -1, im, jm, km);
    shift(axis, 1, ip, jp, kp);

    Conserved lm, rm, lc, rc, lp, rp;
    reconstruct(in, im, jm, km, axis, dt_over_d, lm, rm);
    reconstruct(in, i, j, k, axis, dt_over_d, lc, rc);
    reconstruct(in, ip, jp, kp, axis, dt_over_d, lp, rp);

    const Conserved fminus = riemann_axis(rm, lc, axis, solver);
    const Conserved fplus = riemann_axis(rc, lp, axis, solver);
    const Conserved old = in.cells[in.flat_index(i, j, k)];
    const Conserved candidate = old - dt_over_d * (fplus - fminus);
    out.cells[out.flat_index(i, j, k)] =
        enforce_physical(candidate, old);
}

__global__ void signal_speed_kernel(ConstGrid3DGPUView q, double* speeds) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = q.nx * q.ny * q.nz;
    if (p >= n) return;
    const int li = p % q.nx;
    const int t = p / q.nx;
    const int lj = t % q.ny;
    const int lk = t / q.ny;
    const Conserved u = q.cells[q.flat_index(
        q.i_begin() + li, q.j_begin() + lj, q.k_begin() + lk)];
    const Primitive v = phys::cons_to_prim(u);
    double speed = 0.0;
    if (isfinite(v.rho) && isfinite(v.p) && v.rho > 0.0 && v.p > 0.0) {
        const double sx = phys::max_signal_speed_x(v, 0.0);
        const double sy = phys::max_signal_speed_y(v, 0.0);
        const double sz = phys::max_signal_speed_z(v, 0.0);
        if (isfinite(sx) && isfinite(sy) && isfinite(sz))
            speed = fmax(sx, fmax(sy, sz));
    }
    speeds[p] = speed;
}

__global__ void damp_psi_kernel(Grid3DGPUView q, double factor) {
    const std::size_t p =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t n = static_cast<std::size_t>(q.total_nx()) *
                          q.total_ny() * q.total_nz();
    if (p < n) q.cells[p].psi *= factor;
}

} // namespace

void set_gpu3d_physics_gamma(double gamma) {
    cuda_check(cudaMemcpyToSymbol(phys::d_gamma, &gamma, sizeof(double)),
               "set 3D gamma");
    phys::gamma = gamma;
}

void set_gpu3d_physics_ch(double ch) {
    cuda_check(cudaMemcpyToSymbol(phys::d_ch_glm, &ch, sizeof(double)),
               "set 3D GLM speed");
    phys::ch_glm = ch;
}

void init_gpu_workspace_3d(GpuWorkspace3D& ws, const Grid3DGPU& grid) {
    free_gpu_workspace_3d(ws);
    ws.nx = grid.nx(); ws.ny = grid.ny(); ws.nz = grid.nz();
    const std::size_t cells =
        static_cast<std::size_t>(ws.nx) * ws.ny * ws.nz;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&ws.speeds),
                          cells * sizeof(double)), "allocate 3D speeds");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&ws.max_speed),
                          sizeof(double)), "allocate 3D maximum speed");
    cub::DeviceReduce::Max(nullptr, ws.reduce_tmp_bytes,
                           ws.speeds, ws.max_speed, cells);
    cuda_check(cudaMalloc(&ws.reduce_tmp, ws.reduce_tmp_bytes),
               "allocate 3D reduction workspace");
}

void free_gpu_workspace_3d(GpuWorkspace3D& ws) {
    if (ws.speeds) cuda_check(cudaFree(ws.speeds), "free 3D speeds");
    if (ws.max_speed) cuda_check(cudaFree(ws.max_speed), "free 3D maximum speed");
    if (ws.reduce_tmp) cuda_check(cudaFree(ws.reduce_tmp), "free 3D reduction workspace");
    ws = GpuWorkspace3D{};
}

double compute_dt_gpu_3d(const Grid3DGPU& grid,
                         GpuWorkspace3D& ws,
                         double cfl) {
    if (ws.nx != grid.nx() || ws.ny != grid.ny() || ws.nz != grid.nz())
        throw std::runtime_error("3D GPU workspace does not match grid.");
    const int cells = grid.nx() * grid.ny() * grid.nz();
    const int blocks = (cells + kDtThreads - 1) / kDtThreads;
    signal_speed_kernel<<<blocks, kDtThreads>>>(make_view(grid), ws.speeds);
    cuda_check(cudaGetLastError(), "launch 3D signal-speed kernel");
    cub::DeviceReduce::Max(ws.reduce_tmp, ws.reduce_tmp_bytes,
                           ws.speeds, ws.max_speed, cells);
    cuda_check(cudaGetLastError(), "reduce 3D maximum speed");
    double maximum = 0.0;
    cuda_check(cudaMemcpy(&maximum, ws.max_speed, sizeof(double),
                          cudaMemcpyDeviceToHost), "copy 3D maximum speed");
    if (!std::isfinite(maximum) || maximum <= 0.0)
        throw std::runtime_error("3D GPU maximum signal speed is invalid.");
    set_gpu3d_physics_ch(maximum);
    return cfl * std::min(grid.dx(), std::min(grid.dy(), grid.dz())) / maximum;
}

void advance_gpu_3d(const Grid3DGPU& old,
                    Grid3DGPU& after_x,
                    Grid3DGPU& after_y,
                    Grid3DGPU& next,
                    GpuWorkspace3D& ws,
                    double dt,
                    RiemannSolver solver,
                    const BoundaryConfig3D& bc) {
    if (ws.nx != old.nx() || ws.ny != old.ny() || ws.nz != old.nz())
        throw std::runtime_error("3D GPU workspace does not match grid.");
    const dim3 threads(8, 4, 4);
    const dim3 blocks((old.nx() + threads.x - 1) / threads.x,
                      (old.ny() + threads.y - 1) / threads.y,
                      (old.nz() + threads.z - 1) / threads.z);

    advance_axis_kernel<<<blocks, threads>>>(
        make_view(old), make_view(after_x), dt / old.dx(), X, solver);
    cuda_check(cudaGetLastError(), "launch baseline x sweep");
    apply_boundary_gpu_3d(after_x, bc);

    advance_axis_kernel<<<blocks, threads>>>(
        make_view(static_cast<const Grid3DGPU&>(after_x)),
        make_view(after_y), dt / old.dy(), Y, solver);
    cuda_check(cudaGetLastError(), "launch baseline y sweep");
    apply_boundary_gpu_3d(after_y, bc);

    advance_axis_kernel<<<blocks, threads>>>(
        make_view(static_cast<const Grid3DGPU&>(after_y)),
        make_view(next), dt / old.dz(), Z, solver);
    cuda_check(cudaGetLastError(), "launch baseline z sweep");
    apply_boundary_gpu_3d(next, bc);

    const double factor =
        (phys::ch_glm > 0.0 && phys::cr_glm > 0.0)
            ? std::exp(-dt * phys::ch_glm / phys::cr_glm) : 1.0;
    const std::size_t cells = next.num_cells();
    const int blocks_1d = static_cast<int>((cells + kDtThreads - 1) / kDtThreads);
    damp_psi_kernel<<<blocks_1d, kDtThreads>>>(make_view(next), factor);
    cuda_check(cudaGetLastError(), "launch 3D psi damping");
}
