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

// Optional z-halo exchange used by the slab-decomposed MPI driver. The
// callback must fill both z ghost layers. x/y boundaries remain local and are
// handled by the solver.
using ZHaloExchange3D = void (*)(Grid3D&, void*);

void advance_cpu(
    const Grid3D& Uold, Grid3D& Ux, Grid3D& Uy, Grid3D& Unew,
    double dt, CpuWorkspace3D& ws, RiemannSolver solver,
    const BoundaryConfig3D& bc
);

void advance_cpu_distributed_z(
    const Grid3D& Uold, Grid3D& Ux, Grid3D& Uy, Grid3D& Unew,
    double dt, CpuWorkspace3D& ws, RiemannSolver solver,
    const BoundaryConfig3D& bc, ZHaloExchange3D exchange_z, void* context
);
