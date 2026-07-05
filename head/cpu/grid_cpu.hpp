#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include "types.hpp"

// ============================================================
// 2D Cartesian grid of MHD conserved states.
// Ghost cells of width ng surround the active domain.
// ============================================================
class Grid2D {
public:
    Grid2D() = default;

    // nx_global_in/ny_global_in/i_start_in/j_start_in describe this grid's
    // position within a larger global index space. Single, self-contained
    // grids (CPU serial path, GPU host-side grid via src/init.cpp) leave
    // all four at their defaults (0): dx_/dy_ are then computed from
    // nx_/ny_ directly and i_start_/j_start_ are 0, reproducing the
    // previous single-domain formula exactly, unchanged.
    // An MPI rank's local subgrid instead passes the true global
    // x_min/x_max/y_min/y_max, the true global nx_global_in/ny_global_in,
    // and this rank's i_start_in/j_start_in (dom.i_start/dom.j_start), so
    // that dx_/dy_ and every cell's physical coordinate are produced by
    // this exact same formula and the exact same global dx/dy as the
    // single-domain case, instead of a separately re-derived local dx/dy.
    Grid2D(int nx_, int ny_, int ng_,
           double x_min_, double x_max_,
           double y_min_, double y_max_,
           int nx_global_in = 0, int ny_global_in = 0,
           int i_start_in = 0, int j_start_in = 0)
        : nx_(nx_), ny_(ny_), ng_(ng_),
          x_min_(x_min_), x_max_(x_max_),
          y_min_(y_min_), y_max_(y_max_),
          i_start_(i_start_in), j_start_(j_start_in)
    {
        const int nx_global = (nx_global_in > 0) ? nx_global_in : nx_;
        const int ny_global = (ny_global_in > 0) ? ny_global_in : ny_;
        dx_ = (x_max_ - x_min_) / static_cast<double>(nx_global);
        dy_ = (y_max_ - y_min_) / static_cast<double>(ny_global);

        const int total_x = nx_ + 2 * ng_;
        const int total_y = ny_ + 2 * ng_;
        // Conserved default constructor zero-initialises all 9 fields
        U_.assign(static_cast<std::size_t>(total_x * total_y), Conserved{});
    }

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int ng() const { return ng_; }

    int total_nx() const { return nx_ + 2 * ng_; }
    int total_ny() const { return ny_ + 2 * ng_; }

    double x_min() const { return x_min_; }
    double x_max() const { return x_max_; }
    double y_min() const { return y_min_; }
    double y_max() const { return y_max_; }

    double dx() const { return dx_; }
    double dy() const { return dy_; }

    int i_begin() const { return ng_; }
    int i_end()   const { return ng_ + nx_; }
    int j_begin() const { return ng_; }
    int j_end()   const { return ng_ + ny_; }

    double x_center(int i) const {
        return x_min_ + (static_cast<double>(i_start_ + i - ng_) + 0.5) * dx_;
    }

    double y_center(int j) const {
        return y_min_ + (static_cast<double>(j_start_ + j - ng_) + 0.5) * dy_;
    }

    Conserved& operator()(int i, int j) {
        return U_[flat_index(i, j)];
    }

    const Conserved& operator()(int i, int j) const {
        return U_[flat_index(i, j)];
    }

    std::vector<Conserved>& data() { return U_; }
    const std::vector<Conserved>& data() const { return U_; }

    void fill(const Conserved& value) {
        std::fill(U_.begin(), U_.end(), value);
    }

private:
    int    nx_ = 0, ny_ = 0, ng_ = 0;
    double x_min_ = 0.0, x_max_ = 1.0;
    double y_min_ = 0.0, y_max_ = 1.0;
    double dx_ = 0.0, dy_ = 0.0;
    int    i_start_ = 0, j_start_ = 0;

    std::vector<Conserved> U_;

    std::size_t flat_index(int i, int j) const {
        return static_cast<std::size_t>(j * total_nx() + i);
    }
};
