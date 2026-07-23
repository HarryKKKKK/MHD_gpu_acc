#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "types.hpp"

// Cartesian 3D grid with ng ghost layers around the active volume.
class Grid3D {
public:
    Grid3D() = default;

    Grid3D(int nx, int ny, int nz, int ng,
           double x_min, double x_max,
           double y_min, double y_max,
           double z_min, double z_max)
        : nx_(nx), ny_(ny), nz_(nz), ng_(ng),
          x_min_(x_min), x_max_(x_max),
          y_min_(y_min), y_max_(y_max),
          z_min_(z_min), z_max_(z_max),
          dx_((x_max - x_min) / nx),
          dy_((y_max - y_min) / ny),
          dz_((z_max - z_min) / nz)
    {
        U_.assign(static_cast<std::size_t>(total_nx()) * total_ny() * total_nz(),
                  Conserved{});
    }

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    int ng() const { return ng_; }
    int total_nx() const { return nx_ + 2*ng_; }
    int total_ny() const { return ny_ + 2*ng_; }
    int total_nz() const { return nz_ + 2*ng_; }
    int i_begin() const { return ng_; }
    int i_end() const { return ng_ + nx_; }
    int j_begin() const { return ng_; }
    int j_end() const { return ng_ + ny_; }
    int k_begin() const { return ng_; }
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
    double x_center(int i) const { return x_min_ + (i - ng_ + 0.5)*dx_; }
    double y_center(int j) const { return y_min_ + (j - ng_ + 0.5)*dy_; }
    double z_center(int k) const { return z_min_ + (k - ng_ + 0.5)*dz_; }

    Conserved& operator()(int i, int j, int k) { return U_[flat_index(i,j,k)]; }
    const Conserved& operator()(int i, int j, int k) const {
        return U_[flat_index(i,j,k)];
    }
    std::vector<Conserved>& data() { return U_; }
    const std::vector<Conserved>& data() const { return U_; }
    void fill(const Conserved& value) { std::fill(U_.begin(), U_.end(), value); }

private:
    int nx_ = 0, ny_ = 0, nz_ = 0, ng_ = 0;
    double x_min_ = 0.0, x_max_ = 1.0;
    double y_min_ = 0.0, y_max_ = 1.0;
    double z_min_ = 0.0, z_max_ = 1.0;
    double dx_ = 0.0, dy_ = 0.0, dz_ = 0.0;
    std::vector<Conserved> U_;

    std::size_t flat_index(int i, int j, int k) const {
        return (static_cast<std::size_t>(k)*total_ny() + j)*total_nx() + i;
    }
};

