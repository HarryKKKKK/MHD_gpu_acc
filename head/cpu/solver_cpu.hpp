#pragma once

#include "cpu/grid_cpu.hpp"
#include "riemann.hpp"
#include "test_cases.hpp"
#include "types.hpp"

#include <cstddef>
#include <vector>

// ============================================================
// Pre-allocated workspace for second-order advance.
// Allocate once before the time loop; pass into every
// advance_second_order call to avoid per-step heap allocation.
// ============================================================
struct CpuWorkspace {
    int nx = 0;
    int ny = 0;

    // x-face flux cache: (nx+1) * ny entries
    std::vector<Conserved> fx_cache;

    // y-face flux cache: nx * (ny+1) entries
    std::vector<Conserved> fy_cache;

    // Full-grid primitive cache (including ghost cells): total_nx * total_ny entries.
    // Populated before each directional sweep to avoid redundant cons_to_prim calls.
    std::vector<Primitive> prim_cache;

    // Per-cell MUSCL-Hancock half-stepped face states.
    // recon_L[i,j] = left-face (lower-index) half-stepped state of cell (i,j).
    // recon_R[i,j] = right-face (higher-index) half-stepped state of cell (i,j).
    // Shared between x and y sweeps (used one at a time); lazily sized like prim_cache.
    std::vector<Conserved> recon_L_cache;
    std::vector<Conserved> recon_R_cache;

    void init(int nx_, int ny_) {
        nx = nx_;
        ny = ny_;
        fx_cache.resize(static_cast<std::size_t>(nx + 1) * ny);
        fy_cache.resize(static_cast<std::size_t>(nx) * (ny + 1));
        // prim_cache is lazily sized in advance_second_order (needs grid's ng).
    }

    bool is_initialized() const {
        return nx > 0 && ny > 0 &&
               fx_cache.size() == static_cast<std::size_t>(nx + 1) * ny &&
               fy_cache.size() == static_cast<std::size_t>(nx) * (ny + 1);
    }
};

// ============================================================
// CFL timestep.
// Also sets phys::ch_glm = max signal speed for this step.
//
// out_max_speed, if non-null, receives the max signal speed computed
// this step *before* it is written into phys::ch_glm (diagnostic use
// only — passing nullptr, the default, reproduces the exact prior
// signature/behaviour).
// ============================================================
double compute_dt(const Grid2D& grid, double cfl, double* out_max_speed = nullptr);

// ============================================================
// Second-order MUSCL-Hancock with dimensional (Strang) splitting.
// ws must be initialised with ws.init(cfg.nx, cfg.ny) before the loop.
// ============================================================
void advance_second_order(
    const Grid2D&        Uold,
    Grid2D&              Utmp,
    Grid2D&              Unew,
    double               dt,
    CpuWorkspace&        ws,
    RiemannSolver        solver,
    const BoundaryConfig& bc
);

// Convenience overload: HLL + all-transmissive BC
void advance_second_order(
    const Grid2D& Uold,
    Grid2D&       Utmp,
    Grid2D&       Unew,
    double        dt,
    CpuWorkspace& ws
);
