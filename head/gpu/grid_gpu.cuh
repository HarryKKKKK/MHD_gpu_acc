#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "types.hpp"

// ============================================================
// Grid2DGPU — device-side structure-of-arrays grid.
// Stores all 9 GLM-MHD conserved fields:
//   rho, rhou, rhov, rhow, Bx, By, Bz, E, psi
// Ghost cells of width ng surround the active domain on all sides.
// ============================================================
class Grid2DGPU {
public:
    Grid2DGPU() = default;

    Grid2DGPU(int nx_, int ny_, int ng_,
              double x_min_, double x_max_,
              double y_min_, double y_max_) {
        allocate(nx_, ny_, ng_, x_min_, x_max_, y_min_, y_max_);
    }

    Grid2DGPU(const Grid2DGPU&)            = delete;
    Grid2DGPU& operator=(const Grid2DGPU&) = delete;

    Grid2DGPU(Grid2DGPU&& other) noexcept { move_from(std::move(other)); }

    Grid2DGPU& operator=(Grid2DGPU&& other) noexcept {
        if (this != &other) { release(); move_from(std::move(other)); }
        return *this;
    }

    ~Grid2DGPU() { release(); }

    void allocate(int nx_, int ny_, int ng_,
                  double x_min_, double x_max_,
                  double y_min_, double y_max_) {
        release();
        nx__ = nx_;  ny__ = ny_;  ng__ = ng_;
        x_min__ = x_min_;  x_max__ = x_max_;
        y_min__ = y_min_;  y_max__ = y_max_;
        dx__ = (x_max__ - x_min__) / static_cast<double>(nx__);
        dy__ = (y_max__ - y_min__) / static_cast<double>(ny__);
        const std::size_t n = num_cells();
        cudaMalloc(&rho_,  n * sizeof(double));
        cudaMalloc(&rhou_, n * sizeof(double));
        cudaMalloc(&rhov_, n * sizeof(double));
        cudaMalloc(&rhow_, n * sizeof(double));
        cudaMalloc(&Bx_,   n * sizeof(double));
        cudaMalloc(&By_,   n * sizeof(double));
        cudaMalloc(&Bz_,   n * sizeof(double));
        cudaMalloc(&E_,    n * sizeof(double));
        cudaMalloc(&psi_,  n * sizeof(double));
    }

    void release() {
        if (rho_)  cudaFree(rho_);
        if (rhou_) cudaFree(rhou_);
        if (rhov_) cudaFree(rhov_);
        if (rhow_) cudaFree(rhow_);
        if (Bx_)   cudaFree(Bx_);
        if (By_)   cudaFree(By_);
        if (Bz_)   cudaFree(Bz_);
        if (E_)    cudaFree(E_);
        if (psi_)  cudaFree(psi_);
        rho_  = nullptr; rhou_ = nullptr; rhov_ = nullptr; rhow_ = nullptr;
        Bx_   = nullptr; By_   = nullptr; Bz_   = nullptr;
        E_    = nullptr; psi_  = nullptr;
        nx__ = 0;  ny__ = 0;  ng__ = 0;
        x_min__ = 0.0;  x_max__ = 1.0;
        y_min__ = 0.0;  y_max__ = 1.0;
        dx__ = 0.0;  dy__ = 0.0;
    }

    int    nx()       const { return nx__; }
    int    ny()       const { return ny__; }
    int    ng()       const { return ng__; }
    int    total_nx() const { return nx__ + 2 * ng__; }
    int    total_ny() const { return ny__ + 2 * ng__; }
    double x_min()    const { return x_min__; }
    double x_max()    const { return x_max__; }
    double y_min()    const { return y_min__; }
    double y_max()    const { return y_max__; }
    double dx()       const { return dx__; }
    double dy()       const { return dy__; }
    int    i_begin()  const { return ng__; }
    int    i_end()    const { return ng__ + nx__; }
    int    j_begin()  const { return ng__; }
    int    j_end()    const { return ng__ + ny__; }

    std::size_t num_cells() const {
        return static_cast<std::size_t>(total_nx()) * total_ny();
    }

    // Field accessors
    double*       rho_ptr()        { return rho_; }
    double*       rhou_ptr()       { return rhou_; }
    double*       rhov_ptr()       { return rhov_; }
    double*       rhow_ptr()       { return rhow_; }
    double*       Bx_ptr()         { return Bx_; }
    double*       By_ptr()         { return By_; }
    double*       Bz_ptr()         { return Bz_; }
    double*       E_ptr()          { return E_; }
    double*       psi_ptr()        { return psi_; }
    const double* rho_ptr()  const { return rho_; }
    const double* rhou_ptr() const { return rhou_; }
    const double* rhov_ptr() const { return rhov_; }
    const double* rhow_ptr() const { return rhow_; }
    const double* Bx_ptr()   const { return Bx_; }
    const double* By_ptr()   const { return By_; }
    const double* Bz_ptr()   const { return Bz_; }
    const double* E_ptr()    const { return E_; }
    const double* psi_ptr()  const { return psi_; }

    // Upload all 9 fields from a host AoS vector (length = num_cells())
    void upload_from_aos(const std::vector<Conserved>& host_data) {
        const std::size_t nc    = num_cells();
        const std::size_t bytes = nc * sizeof(double);

        std::vector<double> h_rho(nc), h_rhou(nc), h_rhov(nc), h_rhow(nc);
        std::vector<double> h_Bx(nc),  h_By(nc),   h_Bz(nc);
        std::vector<double> h_E(nc),   h_psi(nc);

        for (std::size_t k = 0; k < nc; ++k) {
            h_rho[k]  = host_data[k].rho;
            h_rhou[k] = host_data[k].rhou;
            h_rhov[k] = host_data[k].rhov;
            h_rhow[k] = host_data[k].rhow;
            h_Bx[k]   = host_data[k].Bx;
            h_By[k]   = host_data[k].By;
            h_Bz[k]   = host_data[k].Bz;
            h_E[k]    = host_data[k].E;
            h_psi[k]  = host_data[k].psi;
        }
        cudaMemcpy(rho_,  h_rho.data(),  bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(rhou_, h_rhou.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(rhov_, h_rhov.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(rhow_, h_rhow.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(Bx_,   h_Bx.data(),  bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(By_,   h_By.data(),  bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(Bz_,   h_Bz.data(),  bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(E_,    h_E.data(),   bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(psi_,  h_psi.data(), bytes, cudaMemcpyHostToDevice);
    }

    // Download all 9 fields back to a host AoS vector
    void download_to_aos(std::vector<Conserved>& host_data) const {
        const std::size_t nc    = num_cells();
        const std::size_t bytes = nc * sizeof(double);

        host_data.resize(nc);
        std::vector<double> h_rho(nc), h_rhou(nc), h_rhov(nc), h_rhow(nc);
        std::vector<double> h_Bx(nc),  h_By(nc),   h_Bz(nc);
        std::vector<double> h_E(nc),   h_psi(nc);

        cudaMemcpy(h_rho.data(),  rho_,  bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rhou.data(), rhou_, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rhov.data(), rhov_, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rhow.data(), rhow_, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Bx.data(),   Bx_,   bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_By.data(),   By_,   bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Bz.data(),   Bz_,   bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_E.data(),    E_,    bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_psi.data(),  psi_,  bytes, cudaMemcpyDeviceToHost);

        for (std::size_t k = 0; k < nc; ++k) {
            host_data[k] = Conserved(
                h_rho[k], h_rhou[k], h_rhov[k], h_rhow[k],
                h_Bx[k],  h_By[k],  h_Bz[k],
                h_E[k],   h_psi[k]
            );
        }
    }

    void swap(Grid2DGPU& other) {
        std::swap(rho_,  other.rho_);
        std::swap(rhou_, other.rhou_);
        std::swap(rhov_, other.rhov_);
        std::swap(rhow_, other.rhow_);
        std::swap(Bx_,   other.Bx_);
        std::swap(By_,   other.By_);
        std::swap(Bz_,   other.Bz_);
        std::swap(E_,    other.E_);
        std::swap(psi_,  other.psi_);
    }

private:
    int    nx__ = 0, ny__ = 0, ng__ = 0;
    double x_min__ = 0.0, x_max__ = 1.0;
    double y_min__ = 0.0, y_max__ = 1.0;
    double dx__ = 0.0, dy__ = 0.0;

    double* rho_  = nullptr;
    double* rhou_ = nullptr;
    double* rhov_ = nullptr;
    double* rhow_ = nullptr;
    double* Bx_   = nullptr;
    double* By_   = nullptr;
    double* Bz_   = nullptr;
    double* E_    = nullptr;
    double* psi_  = nullptr;

    void move_from(Grid2DGPU&& o) {
        nx__ = o.nx__;  ny__ = o.ny__;  ng__ = o.ng__;
        x_min__ = o.x_min__;  x_max__ = o.x_max__;
        y_min__ = o.y_min__;  y_max__ = o.y_max__;
        dx__ = o.dx__;  dy__ = o.dy__;
        rho_  = o.rho_;  rhou_ = o.rhou_; rhov_ = o.rhov_; rhow_ = o.rhow_;
        Bx_   = o.Bx_;   By_   = o.By_;   Bz_   = o.Bz_;
        E_    = o.E_;    psi_  = o.psi_;
        o.rho_=nullptr; o.rhou_=nullptr; o.rhov_=nullptr; o.rhow_=nullptr;
        o.Bx_ =nullptr; o.By_ =nullptr;  o.Bz_ =nullptr;
        o.E_  =nullptr; o.psi_=nullptr;
        o.nx__=0; o.ny__=0; o.ng__=0;
        o.x_min__=0.0; o.x_max__=1.0;
        o.y_min__=0.0; o.y_max__=1.0;
        o.dx__=0.0; o.dy__=0.0;
    }
};

// ============================================================
// Grid2DGPUView — mutable non-owning view (write kernels)
// ============================================================
struct Grid2DGPUView {
    int    nx, ny, ng;
    double x_min, x_max, y_min, y_max, dx, dy;
    double* rho;
    double* rhou;
    double* rhov;
    double* rhow;
    double* Bx;
    double* By;
    double* Bz;
    double* E;
    double* psi;

    __host__ __device__ int total_nx() const { return nx + 2 * ng; }
    __host__ __device__ int total_ny() const { return ny + 2 * ng; }
    __host__ __device__ int i_begin()  const { return ng; }
    __host__ __device__ int i_end()    const { return ng + nx; }
    __host__ __device__ int j_begin()  const { return ng; }
    __host__ __device__ int j_end()    const { return ng + ny; }
    __host__ __device__ int flat_index(int i, int j) const {
        return j * total_nx() + i;
    }
};

// ============================================================
// ConstGrid2DGPUView — read-only non-owning view (read kernels)
// ============================================================
struct ConstGrid2DGPUView {
    int    nx, ny, ng;
    double x_min, x_max, y_min, y_max, dx, dy;
    const double* rho;
    const double* rhou;
    const double* rhov;
    const double* rhow;
    const double* Bx;
    const double* By;
    const double* Bz;
    const double* E;
    const double* psi;

    __host__ __device__ int total_nx() const { return nx + 2 * ng; }
    __host__ __device__ int total_ny() const { return ny + 2 * ng; }
    __host__ __device__ int i_begin()  const { return ng; }
    __host__ __device__ int i_end()    const { return ng + nx; }
    __host__ __device__ int j_begin()  const { return ng; }
    __host__ __device__ int j_end()    const { return ng + ny; }
    __host__ __device__ int flat_index(int i, int j) const {
        return j * total_nx() + i;
    }
};

// ============================================================
// make_view helpers
// ============================================================

inline Grid2DGPUView make_view(Grid2DGPU& grid) {
    return Grid2DGPUView{
        grid.nx(), grid.ny(), grid.ng(),
        grid.x_min(), grid.x_max(),
        grid.y_min(), grid.y_max(),
        grid.dx(), grid.dy(),
        grid.rho_ptr(), grid.rhou_ptr(), grid.rhov_ptr(), grid.rhow_ptr(),
        grid.Bx_ptr(),  grid.By_ptr(),   grid.Bz_ptr(),
        grid.E_ptr(),   grid.psi_ptr()
    };
}

inline ConstGrid2DGPUView make_view(const Grid2DGPU& grid) {
    return ConstGrid2DGPUView{
        grid.nx(), grid.ny(), grid.ng(),
        grid.x_min(), grid.x_max(),
        grid.y_min(), grid.y_max(),
        grid.dx(), grid.dy(),
        grid.rho_ptr(), grid.rhou_ptr(), grid.rhov_ptr(), grid.rhow_ptr(),
        grid.Bx_ptr(),  grid.By_ptr(),   grid.Bz_ptr(),
        grid.E_ptr(),   grid.psi_ptr()
    };
}
