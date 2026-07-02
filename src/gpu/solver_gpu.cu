#include "gpu/solver_gpu.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "gpu/boundary_gpu.cuh"
#include "physics.hpp"
#include "riemann.hpp"
#include "types.hpp"

namespace {

constexpr double kRhoFloor   = 1.0e-12;
constexpr double kPFloor     = 1.0e-12;
constexpr double kCrGlm      = 0.18;
constexpr int    kDtBlockSize = 256;

constexpr int PAD_X = 1;
constexpr int PAD_Y = 1;

inline void check_cuda(cudaError_t err, const char* call, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line
                  << " in " << call << " : " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(call) check_cuda((call), #call, __FILE__, __LINE__)

__device__ inline int clamp_i(int x, int lo, int hi) {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

__device__ inline double minmod_s(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (a > 0.0) ? fmin(a, b) : fmax(a, b);
}

__device__ inline Primitive minmod9(const Primitive& a, const Primitive& b) {
    return Primitive(
        minmod_s(a.rho, b.rho),
        minmod_s(a.u,   b.u),
        minmod_s(a.v,   b.v),
        minmod_s(a.w,   b.w),
        minmod_s(a.Bx,  b.Bx),
        minmod_s(a.By,  b.By),
        minmod_s(a.Bz,  b.Bz),
        minmod_s(a.p,   b.p),
        minmod_s(a.psi, b.psi)
    );
}

__device__ inline bool is_physical(const Primitive& V) {
    return V.rho > kRhoFloor && V.p > kPFloor
        && isfinite(V.rho)  && isfinite(V.p)
        && isfinite(V.u)    && isfinite(V.v)    && isfinite(V.w)
        && isfinite(V.Bx)   && isfinite(V.By)   && isfinite(V.Bz)
        && isfinite(V.psi);
}

__device__ unsigned long long d_floor_trigger_count = 0ULL;

__device__ inline Primitive safe_prim(const Primitive& cand, const Primitive& fb) {
    if (is_physical(cand)) return cand;
    atomicAdd(&d_floor_trigger_count, 1ULL);
    return fb;
}

__device__ inline Conserved safe_cons(const Conserved& cand, const Conserved& fb) {
    if (is_physical(phys::cons_to_prim(cand))) return cand;
    atomicAdd(&d_floor_trigger_count, 1ULL);
    return fb;
}

__device__ inline Conserved gload(const ConstGrid2DGPUView& U, int i, int j) {
    const int idx = U.flat_index(i, j);
    return Conserved(
        U.rho[idx], U.rhou[idx], U.rhov[idx], U.rhow[idx],
        U.Bx[idx],  U.By[idx],   U.Bz[idx],
        U.E[idx],   U.psi[idx]
    );
}

__device__ inline void gstore(Grid2DGPUView& U, int i, int j, const Conserved& C) {
    const int idx = U.flat_index(i, j);
    U.rho[idx]  = C.rho;
    U.rhou[idx] = C.rhou;
    U.rhov[idx] = C.rhov;
    U.rhow[idx] = C.rhow;
    U.Bx[idx]   = C.Bx;
    U.By[idx]   = C.By;
    U.Bz[idx]   = C.Bz;
    U.E[idx]    = C.E;
    U.psi[idx]  = C.psi;
}

struct Tile9 {
    double *rho, *rhou, *rhov, *rhow, *Bx, *By, *Bz, *E, *psi;

    __device__ Conserved load(int idx) const {
        return Conserved(rho[idx], rhou[idx], rhov[idx], rhow[idx],
                         Bx[idx],  By[idx],   Bz[idx],
                         E[idx],   psi[idx]);
    }
    __device__ void store(int idx, const Conserved& C) {
        rho[idx]  = C.rho;
        rhou[idx] = C.rhou;
        rhov[idx] = C.rhov;
        rhow[idx] = C.rhow;
        Bx[idx]   = C.Bx;
        By[idx]   = C.By;
        Bz[idx]   = C.Bz;
        E[idx]    = C.E;
        psi[idx]  = C.psi;
    }
};

__device__ inline Tile9 carve(double*& p, int n_doubles) {
    Tile9 t;
    t.rho  = p; p += n_doubles;
    t.rhou = p; p += n_doubles;
    t.rhov = p; p += n_doubles;
    t.rhow = p; p += n_doubles;
    t.Bx   = p; p += n_doubles;
    t.By   = p; p += n_doubles;
    t.Bz   = p; p += n_doubles;
    t.E    = p; p += n_doubles;
    t.psi  = p; p += n_doubles;
    return t;
}

__device__ inline void reconstruct_muscl_hancock(
    const Conserved& Um, const Conserved& Uc, const Conserved& Up,
    double dt_over_d, Direction dir,
    Conserved& UL_star, Conserved& UR_star
) {
    const Primitive Wm = phys::cons_to_prim(Um);
    const Primitive Wc = phys::cons_to_prim(Uc);
    const Primitive Wp = phys::cons_to_prim(Up);

    const Primitive slope = minmod9(Wc - Wm, Wp - Wc);

    const Primitive WL = safe_prim(Wc - 0.5 * slope, Wc);
    const Primitive WR = safe_prim(Wc + 0.5 * slope, Wc);

    const Conserved UL = phys::prim_to_cons(WL);
    const Conserved UR = phys::prim_to_cons(WR);

    // ch=0 here intentionally (matches predictor step in solver_cpu.cpp)
    const Conserved FL = (dir == Direction::X) ? phys::flux_x(UL, 0.0)
                                               : phys::flux_y(UL, 0.0);
    const Conserved FR = (dir == Direction::X) ? phys::flux_x(UR, 0.0)
                                               : phys::flux_y(UR, 0.0);
    const Conserved half = 0.5 * dt_over_d * (FR - FL);

    UL_star = safe_cons(UL - half, UL);
    UR_star = safe_cons(UR - half, UR);
}

template <int BLOCK_SIZE>
__global__ void compute_block_max_speed_kernel(
    ConstGrid2DGPUView grid,
    double* __restrict__ speed_block_max
) {
    constexpr int WARPS = BLOCK_SIZE / 32;
    __shared__ double warp_max[WARPS];

    const int local_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int n = grid.nx * grid.ny;
    double local_speed = 0.0;

    if (local_idx < n) {
        const int lj = local_idx / grid.nx;
        const int li = local_idx - lj * grid.nx;
        const int i  = grid.i_begin() + li;
        const int j  = grid.j_begin() + lj;

        const Conserved U = gload(grid, i, j);
        const Primitive V = phys::cons_to_prim(U);

        if (is_physical(V)) {
            const double sx = phys::max_signal_speed_x(V, 0.0);
            const double sy = phys::max_signal_speed_y(V, 0.0);
            if (isfinite(sx) && isfinite(sy))
                local_speed = fmax(sx, sy);
        }
    }

    const unsigned mask = 0xFFFFFFFFu;
    local_speed = fmax(local_speed, __shfl_down_sync(mask, local_speed, 16));
    local_speed = fmax(local_speed, __shfl_down_sync(mask, local_speed,  8));
    local_speed = fmax(local_speed, __shfl_down_sync(mask, local_speed,  4));
    local_speed = fmax(local_speed, __shfl_down_sync(mask, local_speed,  2));
    local_speed = fmax(local_speed, __shfl_down_sync(mask, local_speed,  1));

    const int lane   = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;
    if (lane == 0) warp_max[warpId] = local_speed;
    __syncthreads();

    if (warpId == 0) {
        double val = (lane < WARPS) ? warp_max[lane] : 0.0;
        val = fmax(val, __shfl_down_sync(mask, val, 16));
        val = fmax(val, __shfl_down_sync(mask, val,  8));
        val = fmax(val, __shfl_down_sync(mask, val,  4));
        val = fmax(val, __shfl_down_sync(mask, val,  2));
        val = fmax(val, __shfl_down_sync(mask, val,  1));
        if (lane == 0) speed_block_max[blockIdx.x] = val;
    }
}

__global__ void apply_psi_damping_kernel(Grid2DGPUView grid, double factor) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + grid.i_begin();
    const int j = blockIdx.y * blockDim.y + threadIdx.y + grid.j_begin();
    if (i >= grid.i_end() || j >= grid.j_end()) return;
    const int idx = grid.flat_index(i, j);
    grid.psi[idx] *= factor;
}

__global__ void advance_x_kernel(
    ConstGrid2DGPUView Uin,
    Grid2DGPUView      Uout,
    double             dt,
    RiemannSolver      solver
) {
    const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_j = blockIdx.y * blockDim.y + threadIdx.y;

    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_threads = blockDim.x * blockDim.y;
    const int bx = blockDim.x;
    const int by = blockDim.y;
    const int block_i_start = blockIdx.x * bx;

    const int sw = bx + 4 + PAD_X;
    const int rw = bx + 2 + PAD_X;
    const int fw = bx + 1;
    const int sn = sw * by;
    const int rn = rw * by;
    const int fn = fw * by;

    extern __shared__ double smem[];
    double* ptr = smem;

    Tile9 S  = carve(ptr, sn);
    Tile9 L  = carve(ptr, rn);
    Tile9 R  = carve(ptr, rn);
    Tile9 F  = carve(ptr, fn);

    for (int lin = tid; lin < sn; lin += block_threads) {
        const int sj = lin / sw;
        const int si = lin - sj * sw;

        const int lj  = blockIdx.y * by + sj;
        const int li_raw = block_i_start - 2 + si;

        if (lj < Uin.ny && si < bx + 4) {
            const int li_clamp = clamp_i(li_raw, -Uin.ng, Uin.nx + Uin.ng - 1);
            const int gi = Uin.i_begin() + li_clamp;
            const int gj = Uin.j_begin() + lj;
            const Conserved U = gload(Uin, gi, gj);
            S.store(lin, U);
        }
    }
    __syncthreads();

    const double dt_dx = dt / Uin.dx;

    for (int lin = tid; lin < rn; lin += block_threads) {
        const int sj = lin / rw;
        const int sr = lin - sj * rw;

        const int lj      = blockIdx.y * by + sj;
        const int li_recon = block_i_start - 1 + sr;

        if (lj < Uin.ny && sr < bx + 2
            && li_recon >= -1 && li_recon <= Uin.nx)
        {
            const int sc = sj * sw + (sr + 1);
            const Conserved Um = S.load(sc - 1);
            const Conserved Uc = S.load(sc);
            const Conserved Up = S.load(sc + 1);

            Conserved UL, UR;
            reconstruct_muscl_hancock(Um, Uc, Up, dt_dx, Direction::X, UL, UR);
            L.store(lin, UL);
            R.store(lin, UR);
        }
    }
    __syncthreads();

    for (int lin = tid; lin < fn; lin += block_threads) {
        const int sj = lin / fw;
        const int sf = lin - sj * fw;

        const int lj     = blockIdx.y * by + sj;
        const int li_face = block_i_start + sf;

        if (lj < Uin.ny && li_face >= 0 && li_face <= Uin.nx) {
            const int left_idx  = sj * rw + sf;
            const int right_idx = sj * rw + sf + 1;

            const Conserved UL = R.load(left_idx);
            const Conserved UR = L.load(right_idx);

            F.store(lin, riemann_flux(UL, UR, Direction::X, solver));
        }
    }
    __syncthreads();

    if (local_i >= Uin.nx || local_j >= Uin.ny) return;

    const int sj = threadIdx.y;
    const int fm = sj * fw + threadIdx.x;
    const int fp = sj * fw + threadIdx.x + 1;

    const int sc = sj * sw + threadIdx.x + 2;
    const Conserved Uc     = S.load(sc);
    const Conserved Unew_c = Uc - dt_dx * (F.load(fp) - F.load(fm));

    gstore(Uout,
           Uin.i_begin() + local_i,
           Uin.j_begin() + local_j,
           safe_cons(Unew_c, Uc));
}

__global__ void advance_y_kernel(
    ConstGrid2DGPUView Uin,
    Grid2DGPUView      Uout,
    double             dt,
    RiemannSolver      solver
) {
    const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_j = blockIdx.y * blockDim.y + threadIdx.y;

    const int tid           = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_threads  = blockDim.x * blockDim.y;
    const int bx = blockDim.x;
    const int by = blockDim.y;
    const int block_j_start = blockIdx.y * by;

    const int sw  = bx + PAD_Y;
    const int rw  = bx + PAD_Y;
    const int fw  = bx + PAD_Y;
    const int sn  = sw * (by + 4);
    const int rn  = rw * (by + 2);
    const int fn_ = fw * (by + 1);

    extern __shared__ double smem[];
    double* ptr = smem;

    Tile9 S = carve(ptr, sn);
    Tile9 L = carve(ptr, rn);
    Tile9 R = carve(ptr, rn);
    Tile9 F = carve(ptr, fn_);

    for (int lin = tid; lin < sn; lin += block_threads) {
        const int sj = lin / sw;
        const int si = lin - sj * sw;

        const int li      = blockIdx.x * bx + si;
        const int lj_raw  = block_j_start - 2 + sj;

        if (li < Uin.nx && si < bx) {
            const int lj_clamp = clamp_i(lj_raw, -Uin.ng, Uin.ny + Uin.ng - 1);
            const int gi = Uin.i_begin() + li;
            const int gj = Uin.j_begin() + lj_clamp;
            const Conserved U = gload(Uin, gi, gj);
            S.store(lin, U);
        }
    }
    __syncthreads();

    const double dt_dy = dt / Uin.dy;

    for (int lin = tid; lin < rn; lin += block_threads) {
        const int sr = lin / rw;
        const int si = lin - sr * rw;

        const int li      = blockIdx.x * bx + si;
        const int lj_recon = block_j_start - 1 + sr;

        if (li < Uin.nx && si < bx
            && lj_recon >= -1 && lj_recon <= Uin.ny)
        {
            const int sc = (sr + 1) * sw + si;
            const Conserved Um = S.load(sc - sw);
            const Conserved Uc = S.load(sc);
            const Conserved Up = S.load(sc + sw);

            Conserved UL, UR;
            reconstruct_muscl_hancock(Um, Uc, Up, dt_dy, Direction::Y, UL, UR);
            L.store(lin, UL);
            R.store(lin, UR);
        }
    }
    __syncthreads();

    for (int lin = tid; lin < fn_; lin += block_threads) {
        const int sf = lin / fw;
        const int si = lin - sf * fw;

        const int li     = blockIdx.x * bx + si;
        const int lj_face = block_j_start + sf;

        if (li < Uin.nx && si < bx
            && lj_face >= 0 && lj_face <= Uin.ny)
        {
            const int lower = sf       * rw + si;
            const int upper = (sf + 1) * rw + si;

            const Conserved UL = R.load(lower);
            const Conserved UR = L.load(upper);

            F.store(lin, riemann_flux(UL, UR, Direction::Y, solver));
        }
    }
    __syncthreads();

    if (local_i >= Uin.nx || local_j >= Uin.ny) return;

    const int si  = threadIdx.x;
    const int fm  = threadIdx.y       * fw + si;
    const int fp  = (threadIdx.y + 1) * fw + si;

    const int sc  = (threadIdx.y + 2) * sw + si;
    const Conserved Uc     = S.load(sc);
    const Conserved Unew_c = Uc - dt_dy * (F.load(fp) - F.load(fm));

    gstore(Uout,
           Uin.i_begin() + local_i,
           Uin.j_begin() + local_j,
           safe_cons(Unew_c, Uc));
}

} // anonymous namespace

void set_gpu_physics_gamma(double g) {
    CUDA_CHECK(cudaMemcpyToSymbol(phys::d_gamma, &g, sizeof(double)));
    phys::gamma = g;
}

void set_gpu_physics_ch(double ch) {
    CUDA_CHECK(cudaMemcpyToSymbol(phys::d_ch_glm, &ch, sizeof(double)));
    phys::ch_glm = ch;
}

void reset_floor_trigger_count_gpu() {
    const unsigned long long zero = 0ULL;
    CUDA_CHECK(cudaMemcpyToSymbol(d_floor_trigger_count, &zero, sizeof(zero)));
}

unsigned long long read_floor_trigger_count_gpu() {
    unsigned long long value = 0ULL;
    CUDA_CHECK(cudaMemcpyFromSymbol(&value, d_floor_trigger_count, sizeof(value)));
    return value;
}

void init_gpu_workspace(GpuWorkspace& ws, const Grid2DGPU& grid) {
    free_gpu_workspace(ws);
    ws.nx = grid.nx();
    ws.ny = grid.ny();

    const int interior_cells = grid.nx() * grid.ny();
    const int num_blocks =
        (interior_cells + kDtBlockSize - 1) / kDtBlockSize;

    CUDA_CHECK(cudaMalloc(&ws.speed_d,
        static_cast<std::size_t>(num_blocks) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ws.max_speed_d, sizeof(double)));

    ws.reduce_tmp_bytes = 0;
    cub::DeviceReduce::Max(nullptr, ws.reduce_tmp_bytes,
                           ws.speed_d, ws.max_speed_d, num_blocks);
    CUDA_CHECK(cudaMalloc(&ws.reduce_tmp, ws.reduce_tmp_bytes));
}

void free_gpu_workspace(GpuWorkspace& ws) {
    if (ws.speed_d)    CUDA_CHECK(cudaFree(ws.speed_d));
    if (ws.max_speed_d) CUDA_CHECK(cudaFree(ws.max_speed_d));
    if (ws.reduce_tmp) CUDA_CHECK(cudaFree(ws.reduce_tmp));
    ws = GpuWorkspace{};
}

double compute_dt_gpu(const Grid2DGPU& grid, GpuWorkspace& ws, double cfl,
                       double* out_max_speed) {
    if (ws.nx != grid.nx() || ws.ny != grid.ny() || !ws.speed_d)
        throw std::runtime_error("compute_dt_gpu: workspace not initialised.");

    const int interior_cells = grid.nx() * grid.ny();
    const int num_blocks =
        (interior_cells + kDtBlockSize - 1) / kDtBlockSize;

    compute_block_max_speed_kernel<kDtBlockSize>
        <<<num_blocks, kDtBlockSize>>>(make_view(grid), ws.speed_d);
    CUDA_CHECK(cudaGetLastError());

    cub::DeviceReduce::Max(ws.reduce_tmp, ws.reduce_tmp_bytes,
                           ws.speed_d, ws.max_speed_d, num_blocks);
    CUDA_CHECK(cudaGetLastError());

    double max_speed = 0.0;
    CUDA_CHECK(cudaMemcpy(&max_speed, ws.max_speed_d,
                          sizeof(double), cudaMemcpyDeviceToHost));

    if (out_max_speed) *out_max_speed = max_speed;

    if (max_speed <= 0.0) {
        throw std::runtime_error("compute_dt_gpu: non-positive maximum wave speed.");
    }

    set_gpu_physics_ch(max_speed);

    return cfl * std::min(grid.dx(), grid.dy()) / max_speed;
}

void advance_second_order_gpu(
    const Grid2DGPU& Uold,
    Grid2DGPU&       Utmp,
    Grid2DGPU&       Unew,
    GpuWorkspace&    ws,
    double           dt,
    RiemannSolver    solver,
    const BoundaryConfig& bc
) {
    if (ws.nx != Uold.nx() || ws.ny != Uold.ny() || !ws.speed_d)
        throw std::runtime_error("advance_second_order_gpu: workspace not initialised.");

    const int bx = 16, by = 16;
    const dim3 threads(bx, by);
    const dim3 blocks(
        (Uold.nx() + bx - 1) / bx,
        (Uold.ny() + by - 1) / by
    );

    const int x_sw = bx + 4 + PAD_X;
    const int x_rw = bx + 2 + PAD_X;
    const int x_fw = bx + 1;
    const std::size_t x_smem =
        9 * static_cast<std::size_t>(
            x_sw * by + 2 * x_rw * by + x_fw * by
        ) * sizeof(double);

    const int y_sw = bx + PAD_Y;
    const int y_sh = by + 4;
    const int y_rh = by + 2;
    const int y_fh = by + 1;
    const std::size_t y_smem =
        9 * static_cast<std::size_t>(
            y_sw * y_sh + 2 * y_sw * y_rh + y_sw * y_fh
        ) * sizeof(double);

    CUDA_CHECK(cudaFuncSetAttribute(
        advance_x_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(x_smem)));
    CUDA_CHECK(cudaFuncSetAttribute(
        advance_y_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(y_smem)));

    advance_x_kernel<<<blocks, threads, x_smem>>>(
        make_view(static_cast<const Grid2DGPU&>(Uold)),
        make_view(Utmp), dt, solver);
    CUDA_CHECK(cudaGetLastError());
    apply_boundary_gpu(Utmp, bc);

    advance_y_kernel<<<blocks, threads, y_smem>>>(
        make_view(static_cast<const Grid2DGPU&>(Utmp)),
        make_view(Unew), dt, solver);
    CUDA_CHECK(cudaGetLastError());
    apply_boundary_gpu(Unew, bc);

    const double ch = [&]() {
        double h = 0.0;
        CUDA_CHECK(cudaMemcpyFromSymbol(&h, phys::d_ch_glm, sizeof(double)));
        return h;
    }();
    if (ch > 0.0) {
        const double factor = std::exp(-dt * ch / kCrGlm);
        apply_psi_damping_kernel<<<blocks, threads>>>(make_view(Unew), factor);
        CUDA_CHECK(cudaGetLastError());
    }
}
