#pragma once

#include <mpi.h>

#include <cstddef>
#include <string>
#include <vector>

#include "cpu/grid_cpu.hpp"
#include "riemann.hpp"
#include "test_cases.hpp"
#include "types.hpp"

// ============================================================
// Pure-MPI counterpart of solver_cpu.hpp / solver_cpu.cpp.
//
// Algorithm and optimisations are identical to the OpenMP CPU solver:
// second-order MUSCL-Hancock reconstruction with x-then-y dimensional
// splitting, a precomputed primitive-variable cache, a precomputed
// per-cell left/right reconstruction cache, and precomputed x-/y-face
// Riemann flux caches (see src/cpu/solver_mpi.cpp).
//
// The only difference from the OpenMP version is *how ghost cells are
// filled*: instead of a single self-contained grid whose ghosts are
// refreshed via copy_ghost_cells()/apply_boundary(), the global grid is
// split into a 2D Cartesian grid of MPI ranks. Each rank owns a Grid2D
// covering its own sub-domain plus an ng-wide ghost border, and ghost
// cells are filled either by halo-exchanging with a real neighbour rank
// (interior partition boundary, or periodic wrap-around handled by the
// Cartesian communicator itself) or, on ranks that sit on the true
// global domain edge, by applying the case's physical boundary
// condition locally (identical formulas to boundary_cpu.hpp).
//
// This file intentionally does not depend on solver_cpu.hpp/.cpp so it
// can be built and reasoned about on its own (see Makefile: the `mpi`
// target links solver_mpi.o, not solver_cpu.o).
// ============================================================

// ------------------------------------------------------------
// 2D Cartesian domain decomposition.
// ------------------------------------------------------------
struct MpiDomain {
    MPI_Comm cart_comm = MPI_COMM_NULL;
    int      rank      = 0;
    int      nprocs    = 1;

    int dims[2]   = {1, 1};   // {px, py} ranks along x and y
    int coords[2] = {0, 0};   // this rank's (px_idx, py_idx)

    int nbr_left = MPI_PROC_NULL, nbr_right = MPI_PROC_NULL;
    int nbr_down = MPI_PROC_NULL, nbr_up    = MPI_PROC_NULL;

    int nx_global = 0, ny_global = 0;
    int nx_local  = 0, ny_local  = 0;
    int i_start   = 0, j_start   = 0;  // this rank's offset into the global index space
    int ng        = 0;

    bool periodic_x = false, periodic_y = false;

    // Committed derived datatypes for halo exchange (built once, reused every step).
    MPI_Datatype conserved_type = MPI_DATATYPE_NULL;  // 9 contiguous doubles == one Conserved
    MPI_Datatype col_halo_type  = MPI_DATATYPE_NULL;   // ng-wide column strip, total_ny rows

    bool is_root() const { return rank == 0; }
};

// Build the 2D Cartesian topology, block-decompose [nx_global x ny_global]
// across ranks, and commit the halo-exchange datatypes. An axis is made
// periodic in the Cartesian communicator only if BOTH of its sides use
// BoundaryType::Periodic (matching what a single-rank run would do).
MpiDomain mpi_domain_create(
    MPI_Comm world, int nx_global, int ny_global, int ng,
    const BoundaryConfig& bc
);

void mpi_domain_destroy(MpiDomain& dom);

// Allocate this rank's local grid (physical sub-extent offset to match
// its block of the global domain) and fill it via the existing
// initial_state_at()/initialise_grid() machinery evaluated at each local
// cell's *global* physical coordinate. No communication is required:
// the analytic initial condition evaluated at a neighbouring rank's
// coordinate reproduces exactly what that rank computes for its own
// interior cell, so per-rank init is bit-for-bit consistent with a
// single-rank run.
Grid2D make_local_grid(
    const MpiDomain& dom, const CaseConfig& cfg, const std::string& case_name
);

// Exchange the ng-wide left/right (x) or bottom/top (y) ghost layers with
// real neighbour ranks. On ranks that sit on the true global boundary in
// that direction (neighbour == MPI_PROC_NULL), apply the physical BC
// locally instead; Dirichlet ghosts are copied from dirichlet_src
// (mirroring copy_ghost_cells() in boundary_cpu.hpp).
void exchange_halo_x(
    Grid2D& grid, const MpiDomain& dom, const BoundaryConfig& bc,
    const Grid2D* dirichlet_src = nullptr
);
void exchange_halo_y(
    Grid2D& grid, const MpiDomain& dom, const BoundaryConfig& bc,
    const Grid2D* dirichlet_src = nullptr
);
inline void exchange_halo_full(
    Grid2D& grid, const MpiDomain& dom, const BoundaryConfig& bc,
    const Grid2D* dirichlet_src = nullptr
) {
    exchange_halo_x(grid, dom, bc, dirichlet_src);
    exchange_halo_y(grid, dom, bc, dirichlet_src);
}

// Gather every rank's interior cells into a full [nx_global x ny_global]
// grid on rank 0 (used only for CSV snapshot output). Returns a default,
// empty Grid2D on all non-root ranks.
Grid2D gather_global_grid(const MpiDomain& dom, const Grid2D& local, const CaseConfig& cfg);

// ============================================================
// Pre-allocated per-rank workspace (same role/fields as CpuWorkspace in
// solver_cpu.hpp, duplicated here to keep this file self-contained).
// ============================================================
struct MpiWorkspace {
    int nx = 0;
    int ny = 0;

    // x-face flux cache: (nx+1) * ny entries
    std::vector<Conserved> fx_cache;
    // y-face flux cache: nx * (ny+1) entries
    std::vector<Conserved> fy_cache;
    // Full-grid primitive cache (including ghost cells).
    std::vector<Primitive> prim_cache;
    // Per-cell MUSCL-Hancock half-stepped face states.
    std::vector<Conserved> recon_L_cache;
    std::vector<Conserved> recon_R_cache;

    void init(int nx_, int ny_) {
        nx = nx_;
        ny = ny_;
        fx_cache.resize(static_cast<std::size_t>(nx + 1) * ny);
        fy_cache.resize(static_cast<std::size_t>(nx) * (ny + 1));
    }

    bool is_initialized() const {
        return nx > 0 && ny > 0 &&
               fx_cache.size() == static_cast<std::size_t>(nx + 1) * ny &&
               fy_cache.size() == static_cast<std::size_t>(nx) * (ny + 1);
    }
};

// ============================================================
// CFL timestep, MPI-reduced.
// Each rank computes its own local maximum signal speed over its
// interior cells (identical loop/formula to compute_dt() in
// solver_cpu.cpp), then MPI_Allreduce(MAX) gives every rank the same
// global maximum, which is used to set phys::ch_glm identically
// everywhere before flux computation.
// ============================================================
double compute_dt_mpi(const Grid2D& grid, double cfl, MPI_Comm comm);

// ============================================================
// Second-order MUSCL-Hancock, x-then-y dimensional splitting.
// Identical algorithm/caching structure to advance_second_order() in
// solver_cpu.cpp; ghost cells are refreshed via MPI halo exchange
// instead of copy_ghost_cells()/apply_boundary().
// ============================================================
void advance_second_order_mpi(
    const Grid2D&         Uold,
    Grid2D&                Utmp,
    Grid2D&                Unew,
    double                 dt,
    MpiWorkspace&          ws,
    RiemannSolver          solver,
    const BoundaryConfig&  bc,
    const MpiDomain&       dom
);
