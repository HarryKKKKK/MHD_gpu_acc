#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "types.hpp"

// Deliberately simple AoS storage for the unoptimised 3D GPU baseline.
// One allocation and one flat index make the implementation easy to audit.
class Grid3DGPU {
public:
    Grid3DGPU() = default;

    Grid3DGPU(int nx, int ny, int nz, int ng,
              double x_min, double x_max,
              double y_min, double y_max,
              double z_min, double z_max) {
        allocate(nx, ny, nz, ng, x_min, x_max, y_min, y_max, z_min, z_max);
    }

    Grid3DGPU(const Grid3DGPU&) = delete;
    Grid3DGPU& operator=(const Grid3DGPU&) = delete;
    Grid3DGPU(Grid3DGPU&& other) noexcept { move_from(std::move(other)); }
    Grid3DGPU& operator=(Grid3DGPU&& other) noexcept {
        if (this != &other) {
            release();
            move_from(std::move(other));
        }
        return *this;
    }
    ~Grid3DGPU() { release(); }

    void allocate(int nx, int ny, int nz, int ng,
                  double x_min, double x_max,
                  double y_min, double y_max,
                  double z_min, double z_max) {
        release();
        if (nx <= 0 || ny <= 0 || nz <= 0 || ng < 2)
            throw std::runtime_error("Grid3DGPU requires positive dimensions and ng >= 2.");
        nx_ = nx; ny_ = ny; nz_ = nz; ng_ = ng;
        x_min_ = x_min; x_max_ = x_max;
        y_min_ = y_min; y_max_ = y_max;
        z_min_ = z_min; z_max_ = z_max;
        dx_ = (x_max_ - x_min_) / nx_;
        dy_ = (y_max_ - y_min_) / ny_;
        dz_ = (z_max_ - z_min_) / nz_;
        const cudaError_t err = cudaMalloc(
            reinterpret_cast<void**>(&cells_), num_cells() * sizeof(Conserved));
        if (err != cudaSuccess)
            throw std::runtime_error("Grid3DGPU cudaMalloc failed.");
    }

    void release() {
        if (cells_) cudaFree(cells_);
        cells_ = nullptr;
        nx_ = ny_ = nz_ = ng_ = 0;
    }

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    int ng() const { return ng_; }
    int total_nx() const { return nx_ + 2 * ng_; }
    int total_ny() const { return ny_ + 2 * ng_; }
    int total_nz() const { return nz_ + 2 * ng_; }
    int i_begin() const { return ng_; }
    int j_begin() const { return ng_; }
    int k_begin() const { return ng_; }
    int i_end() const { return ng_ + nx_; }
    int j_end() const { return ng_ + ny_; }
    int k_end() const { return ng_ + nz_; }
    double x_min() const { return x_min_; }
    double x_max() const { return x_max_; }
    double y_min() const { return y_min_; }
    double y_max() const { return y_max_; }
    double z_min() const { return z_min_; }
    double z_max() const { return z_max_; }
    double dx() const { return dx_; }
    double dy() const { return dy_; }
    double dz() const { return dz_; }
    std::size_t num_cells() const {
        return static_cast<std::size_t>(total_nx()) * total_ny() * total_nz();
    }
    Conserved* data() { return cells_; }
    const Conserved* data() const { return cells_; }

    void upload(const std::vector<Conserved>& host) {
        if (host.size() != num_cells())
            throw std::runtime_error("Grid3DGPU upload size mismatch.");
        const cudaError_t err = cudaMemcpy(cells_, host.data(),
            host.size() * sizeof(Conserved), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw std::runtime_error("Grid3DGPU upload failed.");
    }

    void download(std::vector<Conserved>& host) const {
        host.resize(num_cells());
        const cudaError_t err = cudaMemcpy(host.data(), cells_,
            host.size() * sizeof(Conserved), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            throw std::runtime_error("Grid3DGPU download failed.");
    }

    void swap(Grid3DGPU& other) { std::swap(cells_, other.cells_); }

private:
    int nx_ = 0, ny_ = 0, nz_ = 0, ng_ = 0;
    double x_min_ = 0.0, x_max_ = 1.0;
    double y_min_ = 0.0, y_max_ = 1.0;
    double z_min_ = 0.0, z_max_ = 1.0;
    double dx_ = 0.0, dy_ = 0.0, dz_ = 0.0;
    Conserved* cells_ = nullptr;

    void move_from(Grid3DGPU&& other) {
        nx_ = other.nx_; ny_ = other.ny_; nz_ = other.nz_; ng_ = other.ng_;
        x_min_ = other.x_min_; x_max_ = other.x_max_;
        y_min_ = other.y_min_; y_max_ = other.y_max_;
        z_min_ = other.z_min_; z_max_ = other.z_max_;
        dx_ = other.dx_; dy_ = other.dy_; dz_ = other.dz_;
        cells_ = other.cells_;
        other.cells_ = nullptr;
        other.nx_ = other.ny_ = other.nz_ = other.ng_ = 0;
    }
};

struct Grid3DGPUView {
    int nx, ny, nz, ng;
    double dx, dy, dz;
    Conserved* cells;

    __host__ __device__ int total_nx() const { return nx + 2 * ng; }
    __host__ __device__ int total_ny() const { return ny + 2 * ng; }
    __host__ __device__ int total_nz() const { return nz + 2 * ng; }
    __host__ __device__ int i_begin() const { return ng; }
    __host__ __device__ int j_begin() const { return ng; }
    __host__ __device__ int k_begin() const { return ng; }
    __host__ __device__ std::size_t flat_index(int i, int j, int k) const {
        return (static_cast<std::size_t>(k) * total_ny() + j) * total_nx() + i;
    }
};

struct ConstGrid3DGPUView {
    int nx, ny, nz, ng;
    double dx, dy, dz;
    const Conserved* cells;

    __host__ __device__ int total_nx() const { return nx + 2 * ng; }
    __host__ __device__ int total_ny() const { return ny + 2 * ng; }
    __host__ __device__ int total_nz() const { return nz + 2 * ng; }
    __host__ __device__ int i_begin() const { return ng; }
    __host__ __device__ int j_begin() const { return ng; }
    __host__ __device__ int k_begin() const { return ng; }
    __host__ __device__ std::size_t flat_index(int i, int j, int k) const {
        return (static_cast<std::size_t>(k) * total_ny() + j) * total_nx() + i;
    }
};

inline Grid3DGPUView make_view(Grid3DGPU& g) {
    return {g.nx(), g.ny(), g.nz(), g.ng(), g.dx(), g.dy(), g.dz(), g.data()};
}

inline ConstGrid3DGPUView make_view(const Grid3DGPU& g) {
    return {g.nx(), g.ny(), g.nz(), g.ng(), g.dx(), g.dy(), g.dz(), g.data()};
}
