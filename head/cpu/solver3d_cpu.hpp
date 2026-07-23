#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "cpu/boundary3d_cpu.hpp"
#include "cpu/grid3d_cpu.hpp"
#include "riemann.hpp"

struct CpuWorkspace3D {
    int nx = 0, ny = 0, nz = 0;
    std::vector<Conserved> face_flux;
    std::vector<Conserved> state;
    std::vector<Conserved> recon_L;
    std::vector<Conserved> recon_R;

    void init(int nx_, int ny_, int nz_) {
        nx=nx_; ny=ny_; nz=nz_;
        const std::size_t max_faces = std::max({
            static_cast<std::size_t>(nx+1)*ny*nz,
            static_cast<std::size_t>(nx)*(ny+1)*nz,
            static_cast<std::size_t>(nx)*ny*(nz+1)
        });
        face_flux.resize(max_faces);
    }
    bool is_initialized() const { return nx>0 && ny>0 && nz>0 && !face_flux.empty(); }
};

double compute_dt_cpu(const Grid3D& grid, double cfl);

void advance_cpu(
    const Grid3D& Uold, Grid3D& Ux, Grid3D& Uy, Grid3D& Unew,
    double dt, CpuWorkspace3D& ws, RiemannSolver solver,
    const BoundaryConfig3D& bc
);
