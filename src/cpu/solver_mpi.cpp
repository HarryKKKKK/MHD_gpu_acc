#include "cpu/solver_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "init.hpp"
#include "physics.hpp"
#include "riemann.hpp"
#include "types.hpp"

// ============================================================
// NOTE ON ALGORITHMIC PARITY WITH solver_cpu.cpp
// ------------------------------------------------------------
// Every function in the anonymous namespace below (minmod, the MUSCL-
// Hancock half-step reconstruction, the reconstruction/flux caches and
// the psi damping) is a verbatim copy of the corresponding function in
// src/cpu/solver_cpu.cpp, including the #ifdef _OPENMP guards. The pure
// "mpi" Makefile target compiles this file without -fopenmp, so those
// pragmas are inert and every rank runs the exact same serial per-cell
// kernel that a single OpenMP thread would; the optional "mpi_omp"
// target compiles the same file with -fopenmp, additionally threading
// each rank's local sweep. Only the ghost-cell handling differs from
// solver_cpu.cpp (MPI halo exchange instead of apply_boundary() on a
// single self-contained grid) — see exchange_halo_x/y below.
// ============================================================

namespace {

// ---- Minmod limiter for all 9 conserved components ----

inline double minmod_scalar(double a, double b) {
    const double abs_a = std::fabs(a);
    const double abs_b = std::fabs(b);
    const double limited = std::fmin(abs_a, abs_b);
    const double signed_limited = (a > 0.0) ? limited : -limited;
    constexpr double eps = 1.0e-12;
    return (abs_b < eps || a*b <= 0.0) ? 0.0 : signed_limited;
}

inline Conserved minmod_conserved(
    const Conserved& L, const Conserved& C, const Conserved& R
) {
    return Conserved(
        minmod_scalar(C.rho  - L.rho,  R.rho  - C.rho),
        minmod_scalar(C.rhou - L.rhou, R.rhou - C.rhou),
        minmod_scalar(C.rhov - L.rhov, R.rhov - C.rhov),
        minmod_scalar(C.rhow - L.rhow, R.rhow - C.rhow),
        minmod_scalar(C.Bx   - L.Bx,   R.Bx   - C.Bx),
        minmod_scalar(C.By   - L.By,   R.By   - C.By),
        minmod_scalar(C.Bz   - L.Bz,   R.Bz   - C.Bz),
        minmod_scalar(C.E    - L.E,    R.E    - C.E),
        minmod_scalar(C.psi  - L.psi,  R.psi  - C.psi)
    );
}

// ---- Conservative positivity check used by the GPU predictor ----

inline bool positive_conserved(const Conserved& U) {
    const double msq = U.rhou*U.rhou + U.rhov*U.rhov + U.rhow*U.rhow;
    const double mag = 0.5 * (U.Bx*U.Bx + U.By*U.By + U.Bz*U.Bz);
    const double lhs = U.rho * (U.E - mag) - 0.5 * msq;
    return U.rho > 0.0 && lhs > 0.0;
}

// ---- MUSCL-Hancock half-step reconstruction for one cell triplet ----

inline void reconstruct_cell_muscl_hancock(
    const Conserved& Um,
    const Conserved& Uc,
    const Conserved& Up,
    double           dt_over_d,
    Direction        dir,
    Conserved&       U_left_star,
    Conserved&       U_right_star
) {
    const Conserved slope = minmod_conserved(Um, Uc, Up);
    const Conserved half_slope = 0.5 * slope;
    const Conserved U_left = Uc - half_slope;
    const Conserved U_right = Uc + half_slope;

    const double ch = phys::get_ch_glm();
    const Conserved F_left  = (dir == Direction::X) ? phys::flux_x(U_left, ch)
                                                     : phys::flux_y(U_left, ch);
    const Conserved F_right = (dir == Direction::X) ? phys::flux_x(U_right, ch)
                                                     : phys::flux_y(U_right, ch);
    const Conserved base = Uc + 0.5 * dt_over_d * (F_left - F_right);

    U_left_star = base - half_slope;
    U_right_star = base + half_slope;
    if (!positive_conserved(U_left_star) || !positive_conserved(U_right_star)) {
        U_left_star = Uc;
        U_right_star = Uc;
    }
}

// ---- Cache index helpers ----

inline int xface_idx(int local_j, int local_i_face, int nx_faces) {
    return local_j * nx_faces + local_i_face;
}

inline int yface_idx(int local_j_face, int local_i, int nx_cells) {
    return local_j_face * nx_cells + local_i;
}

// ---- Reconstruction fill (x-direction): cells i in [ib-1, ie], rows j in [jb, je) ----

void fill_recon_x_cache(
    const std::vector<Conserved>& U,
    int ib, int ie, int jb, int je,
    int total_nx,
    double dt_over_dx,
    std::vector<Conserved>& recon_L,
    std::vector<Conserved>& recon_R
) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = jb; j < je; ++j) {
        for (int i = ib - 1; i <= ie; ++i) {
            const std::size_t idx = static_cast<std::size_t>(j) * total_nx + i;
            reconstruct_cell_muscl_hancock(
                U[static_cast<std::size_t>(j) * total_nx + (i - 1)],
                U[idx],
                U[static_cast<std::size_t>(j) * total_nx + (i + 1)],
                dt_over_dx, Direction::X,
                recon_L[idx], recon_R[idx]
            );
        }
    }
}

// ---- Reconstruction fill (y-direction): cells j in [jb-1, je], cols i in [ib, ie) ----

void fill_recon_y_cache(
    const std::vector<Conserved>& U,
    int ib, int ie, int jb, int je,
    int total_nx,
    double dt_over_dy,
    std::vector<Conserved>& recon_L,
    std::vector<Conserved>& recon_R
) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = jb - 1; j <= je; ++j) {
        for (int i = ib; i < ie; ++i) {
            const std::size_t idx = static_cast<std::size_t>(j) * total_nx + i;
            reconstruct_cell_muscl_hancock(
                U[static_cast<std::size_t>(j - 1) * total_nx + i],
                U[idx],
                U[static_cast<std::size_t>(j + 1) * total_nx + i],
                dt_over_dy, Direction::Y,
                recon_L[idx], recon_R[idx]
            );
        }
    }
}

// ---- Fill x-face flux cache using precomputed reconstruction states ----

void fill_x_face_cache(
    const Grid2D& Uin,
    const std::vector<Conserved>& recon_L,
    const std::vector<Conserved>& recon_R,
    std::vector<Conserved>& fx_cache,
    RiemannSolver solver
) {
    const int ib = Uin.i_begin();
    const int ie = Uin.i_end();
    const int jb = Uin.j_begin();
    const int je = Uin.j_end();

    const int nx_faces = (ie - ib) + 1;
    const int total_nx = Uin.total_nx();

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = jb; j < je; ++j) {
        for (int i = ib - 1; i < ie; ++i) {
            const int local_j      = j - jb;
            const int local_i_face = i - (ib - 1);

            const std::size_t idx_i   = static_cast<std::size_t>(j) * total_nx + i;
            const std::size_t idx_ip1 = static_cast<std::size_t>(j) * total_nx + (i + 1);

            fx_cache[xface_idx(local_j, local_i_face, nx_faces)] =
                riemann_flux(recon_R[idx_i], recon_L[idx_ip1], Direction::X, solver, phys::ch_glm);
        }
    }
}

// ---- Fill y-face flux cache using precomputed reconstruction states ----

void fill_y_face_cache(
    const Grid2D& Uin,
    const std::vector<Conserved>& recon_L,
    const std::vector<Conserved>& recon_R,
    std::vector<Conserved>& fy_cache,
    RiemannSolver solver
) {
    const int ib = Uin.i_begin();
    const int ie = Uin.i_end();
    const int jb = Uin.j_begin();
    const int je = Uin.j_end();

    const int nx_cells = ie - ib;
    const int total_nx = Uin.total_nx();

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = jb - 1; j < je; ++j) {
        for (int i = ib; i < ie; ++i) {
            const int local_j_face = j - (jb - 1);
            const int local_i      = i - ib;

            const std::size_t idx_j   = static_cast<std::size_t>(j)     * total_nx + i;
            const std::size_t idx_jp1 = static_cast<std::size_t>(j + 1) * total_nx + i;

            fy_cache[yface_idx(local_j_face, local_i, nx_cells)] =
                riemann_flux(recon_R[idx_j], recon_L[idx_jp1], Direction::Y, solver, phys::ch_glm);
        }
    }
}

void apply_psi_damping(Grid2D& grid, double dt) {
    // Dedner et al. (2002): c_r := c_p^2/c_h ~= 0.18 gave optimal results
    // "regardless of the grid resolution" (also confirmed by Bard & Dorelli
    // 2014, JCP 259, who use the same fixed value in all simulations).
    // l_d is therefore used directly as this fixed length, not scaled by dx/dy.
    const double l_d = phys::cr_glm;
    if (l_d <= 0.0) return;  // cr_glm <= 0: no damping

    const double factor = std::exp(-dt * phys::ch_glm / l_d);
    if (factor >= 1.0) return;  // ch=0: no damping needed

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            grid(i, j).psi *= factor;
        }
    }
}

// ============================================================
// Local (per-rank) physical boundary application. Same formulas as
// apply_boundary() in head/cpu/boundary_cpu.hpp, restricted to one side
// at a time so exchange_halo_x/y can call them only on ranks that own a
// genuine global boundary (MPI_PROC_NULL neighbour). Periodic
// BoundaryType never reaches these functions: a periodic axis is given
// periods=1 in MPI_Cart_create, so every rank always has a real
// neighbour (itself, if running on a single rank along that axis) and
// the wrap-around is handled by the halo exchange itself, exactly
// reproducing the single-rank periodic case.
// ============================================================

void apply_physical_left(Grid2D& grid, BoundaryType type) {
    const int ng = grid.ng();
    const int ib = grid.i_begin();
    const int total_ny = grid.total_ny();
    for (int j = 0; j < total_ny; ++j) {
        for (int g = 0; g < ng; ++g) {
            if (type == BoundaryType::Transmissive) {
                grid(ib - 1 - g, j) = grid(ib, j);
            }
        }
    }
}

void apply_physical_right(Grid2D& grid, BoundaryType type) {
    const int ng = grid.ng();
    const int ie = grid.i_end();
    const int total_ny = grid.total_ny();
    for (int j = 0; j < total_ny; ++j) {
        for (int g = 0; g < ng; ++g) {
            if (type == BoundaryType::Transmissive) {
                grid(ie + g, j) = grid(ie - 1, j);
            }
        }
    }
}

void apply_physical_bottom(Grid2D& grid, BoundaryType type) {
    const int ng = grid.ng();
    const int jb = grid.j_begin();
    const int total_nx = grid.total_nx();
    for (int i = 0; i < total_nx; ++i) {
        for (int g = 0; g < ng; ++g) {
            if (type == BoundaryType::Transmissive) {
                grid(i, jb - 1 - g) = grid(i, jb);
            }
        }
    }
}

void apply_physical_top(Grid2D& grid, BoundaryType type) {
    const int ng = grid.ng();
    const int je = grid.j_end();
    const int total_nx = grid.total_nx();
    for (int i = 0; i < total_nx; ++i) {
        for (int g = 0; g < ng; ++g) {
            if (type == BoundaryType::Transmissive) {
                grid(i, je + g) = grid(i, je - 1);
            }
        }
    }
}

// Split n_global as evenly as possible across nparts ranks: the first
// (n_global % nparts) ranks get one extra cell.
void block_decompose(int n_global, int nparts, int idx, int& n_local, int& start) {
    const int base = n_global / nparts;
    const int rem  = n_global % nparts;
    n_local = base + (idx < rem ? 1 : 0);
    start   = idx * base + std::min(idx, rem);
}

} // namespace

// ============================================================
// Domain decomposition
// ============================================================

MpiDomain mpi_domain_create(
    MPI_Comm world, int nx_global, int ny_global, int ng,
    const BoundaryConfig& bc
) {
    MpiDomain dom;
    dom.nx_global = nx_global;
    dom.ny_global = ny_global;
    dom.ng        = ng;

    MPI_Comm_size(world, &dom.nprocs);

    dom.dims[0] = 0;
    dom.dims[1] = 0;
    MPI_Dims_create(dom.nprocs, 2, dom.dims);

    dom.periodic_x = (bc.left   == BoundaryType::Periodic && bc.right == BoundaryType::Periodic);
    dom.periodic_y = (bc.bottom == BoundaryType::Periodic && bc.top   == BoundaryType::Periodic);
    int periods[2] = { dom.periodic_x ? 1 : 0, dom.periodic_y ? 1 : 0 };

    // reorder=0: this rank's number in cart_comm is guaranteed identical
    // to its rank in `world`, so rank 0 always means "the process that
    // does I/O", and gather_global_grid() below can recompute every
    // rank's (coords, offsets) purely locally via MPI_Cart_coords.
    MPI_Cart_create(world, 2, dom.dims, periods, /*reorder=*/0, &dom.cart_comm);
    MPI_Comm_rank(dom.cart_comm, &dom.rank);
    MPI_Cart_coords(dom.cart_comm, dom.rank, 2, dom.coords);

    MPI_Cart_shift(dom.cart_comm, 0, 1, &dom.nbr_left, &dom.nbr_right);
    MPI_Cart_shift(dom.cart_comm, 1, 1, &dom.nbr_down, &dom.nbr_up);

    block_decompose(nx_global, dom.dims[0], dom.coords[0], dom.nx_local, dom.i_start);
    block_decompose(ny_global, dom.dims[1], dom.coords[1], dom.ny_local, dom.j_start);

    // Block sizes can differ by one cell between ranks (remainder distribution),
    // so check collectively rather than throwing locally: either every rank
    // agrees the decomposition is valid, or every rank throws together. A
    // rank-local throw here would let some ranks skip the later Allreduce/
    // Gatherv calls in the time loop while others wait on them forever.
    const int local_ok = (dom.nx_local >= ng && dom.ny_local >= ng) ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, dom.cart_comm);
    if (!global_ok) {
        throw std::runtime_error(
            "mpi_domain_create: local sub-domain is narrower than the ghost "
            "width on at least one axis; use fewer ranks or a larger grid.");
    }

    // A Conserved is exactly 9 contiguous doubles (no padding).
    MPI_Type_contiguous(9, MPI_DOUBLE, &dom.conserved_type);
    MPI_Type_commit(&dom.conserved_type);

    const int total_nx = dom.nx_local + 2 * ng;
    const int total_ny = dom.ny_local + 2 * ng;
    MPI_Type_vector(total_ny, ng, total_nx, dom.conserved_type, &dom.col_halo_type);
    MPI_Type_commit(&dom.col_halo_type);

    return dom;
}

void mpi_domain_destroy(MpiDomain& dom) {
    if (dom.col_halo_type  != MPI_DATATYPE_NULL) MPI_Type_free(&dom.col_halo_type);
    if (dom.conserved_type != MPI_DATATYPE_NULL) MPI_Type_free(&dom.conserved_type);
    if (dom.cart_comm      != MPI_COMM_NULL)     MPI_Comm_free(&dom.cart_comm);
}

Grid2D make_local_grid(
    const MpiDomain& dom, const CaseConfig& cfg, const std::string& case_name
) {
    // Match make_grid_from_config() in src/init.cpp: gamma must be set
    // before initialise_grid() evaluates the initial state.
    phys::gamma = cfg.gamma;

    // Pass the true global domain bounds, the true global cell counts, and
    // this rank's offset into the global index space (dom.i_start/
    // dom.j_start). Grid2D computes dx_/dy_ from the global extent and
    // every cell's physical coordinate via the exact same x_center()/
    // y_center() formula used by the single-domain CPU/GPU grid in
    // src/init.cpp, instead of re-deriving a per-rank local x_min/x_max/dx.
    Grid2D grid(
        dom.nx_local, dom.ny_local, dom.ng,
        cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max,
        cfg.nx, cfg.ny,
        dom.i_start, dom.j_start
    );
    initialise_grid(grid, case_name);
    return grid;
}

// ============================================================
// Halo exchange
// ============================================================

void exchange_halo_x(
    Grid2D& grid, const MpiDomain& dom, const BoundaryConfig& bc
) {
    const int ng = dom.ng;
    const int ib = grid.i_begin();
    const int ie = grid.i_end();

    MPI_Status st;
    constexpr int kTagPlusX  = 101;  // shift toward +x
    constexpr int kTagMinusX = 102;  // shift toward -x

    // +x: send my right-interior edge to nbr_right; receive into my left ghost from nbr_left.
    MPI_Sendrecv(
        &grid(ie - ng, 0), 1, dom.col_halo_type, dom.nbr_right, kTagPlusX,
        &grid(ib - ng, 0), 1, dom.col_halo_type, dom.nbr_left,  kTagPlusX,
        dom.cart_comm, &st
    );
    // -x: send my left-interior edge to nbr_left; receive into my right ghost from nbr_right.
    MPI_Sendrecv(
        &grid(ib, 0), 1, dom.col_halo_type, dom.nbr_left,  kTagMinusX,
        &grid(ie, 0), 1, dom.col_halo_type, dom.nbr_right, kTagMinusX,
        dom.cart_comm, &st
    );

    if (dom.nbr_left == MPI_PROC_NULL) {
        apply_physical_left(grid, bc.left);
    }
    if (dom.nbr_right == MPI_PROC_NULL) {
        apply_physical_right(grid, bc.right);
    }
}

void exchange_halo_y(
    Grid2D& grid, const MpiDomain& dom, const BoundaryConfig& bc
) {
    const int ng = dom.ng;
    const int jb = grid.j_begin();
    const int je = grid.j_end();
    const int total_nx = grid.total_nx();
    const int count = ng * total_nx;  // ng full rows are contiguous in memory

    MPI_Status st;
    constexpr int kTagPlusY  = 201;  // shift toward +y
    constexpr int kTagMinusY = 202;  // shift toward -y

    // +y: send my top-interior rows to nbr_up; receive into my bottom ghost from nbr_down.
    MPI_Sendrecv(
        &grid(0, je - ng), count, dom.conserved_type, dom.nbr_up,   kTagPlusY,
        &grid(0, jb - ng), count, dom.conserved_type, dom.nbr_down, kTagPlusY,
        dom.cart_comm, &st
    );
    // -y: send my bottom-interior rows to nbr_down; receive into my top ghost from nbr_up.
    MPI_Sendrecv(
        &grid(0, jb), count, dom.conserved_type, dom.nbr_down, kTagMinusY,
        &grid(0, je), count, dom.conserved_type, dom.nbr_up,   kTagMinusY,
        dom.cart_comm, &st
    );

    if (dom.nbr_down == MPI_PROC_NULL) {
        apply_physical_bottom(grid, bc.bottom);
    }
    if (dom.nbr_up == MPI_PROC_NULL) {
        apply_physical_top(grid, bc.top);
    }
}

// ============================================================
// Gather (for CSV snapshot output only; not on the hot path)
// ============================================================

Grid2D gather_global_grid(const MpiDomain& dom, const Grid2D& local, const CaseConfig& cfg) {
    std::vector<int> recvcounts(dom.nprocs, 0), displs(dom.nprocs, 0);
    std::vector<int> i_starts(dom.nprocs, 0), j_starts(dom.nprocs, 0);
    std::vector<int> nxls(dom.nprocs, 0),     nyls(dom.nprocs, 0);

    if (dom.is_root()) {
        int offset = 0;
        for (int r = 0; r < dom.nprocs; ++r) {
            int c[2];
            MPI_Cart_coords(dom.cart_comm, r, 2, c);
            int nxl, i0, nyl, j0;
            block_decompose(cfg.nx, dom.dims[0], c[0], nxl, i0);
            block_decompose(cfg.ny, dom.dims[1], c[1], nyl, j0);
            i_starts[r] = i0; j_starts[r] = j0;
            nxls[r] = nxl;   nyls[r] = nyl;
            recvcounts[r] = nxl * nyl;
            displs[r] = offset;
            offset += nxl * nyl;
        }
    }

    // Pack this rank's interior cells into a contiguous row-major buffer.
    std::vector<Conserved> send_buf(static_cast<std::size_t>(dom.nx_local) * dom.ny_local);
    {
        const int ib = local.i_begin(), jb = local.j_begin();
        for (int jj = 0; jj < dom.ny_local; ++jj)
            for (int ii = 0; ii < dom.nx_local; ++ii)
                send_buf[static_cast<std::size_t>(jj) * dom.nx_local + ii] = local(ib + ii, jb + jj);
    }

    std::vector<Conserved> recv_buf;
    if (dom.is_root()) recv_buf.resize(static_cast<std::size_t>(cfg.nx) * cfg.ny);

    MPI_Gatherv(
        send_buf.data(), static_cast<int>(send_buf.size()), dom.conserved_type,
        recv_buf.data(), recvcounts.data(), displs.data(), dom.conserved_type,
        0, dom.cart_comm
    );

    Grid2D global;
    if (dom.is_root()) {
        global = Grid2D(cfg.nx, cfg.ny, cfg.ng, cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max);
        for (int r = 0; r < dom.nprocs; ++r) {
            const int base = displs[r];
            for (int jj = 0; jj < nyls[r]; ++jj) {
                for (int ii = 0; ii < nxls[r]; ++ii) {
                    global(global.i_begin() + i_starts[r] + ii, global.j_begin() + j_starts[r] + jj) =
                        recv_buf[static_cast<std::size_t>(base) + static_cast<std::size_t>(jj) * nxls[r] + ii];
                }
            }
        }
    }
    return global;
}

// ============================================================
// CFL timestep (MPI-reduced)
// ============================================================

double compute_dt_mpi(const Grid2D& grid, double cfl, MPI_Comm comm) {
    double max_speed = 0.0;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(max:max_speed) schedule(static)
#endif
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const Primitive V = phys::cons_to_prim(grid(i, j));

            if (!std::isfinite(V.rho) || !std::isfinite(V.p) ||
                V.rho <= 0.0 || V.p <= 0.0) continue;

            const double sx = phys::max_signal_speed_x(V, 0.0);
            const double sy = phys::max_signal_speed_y(V, 0.0);
            if (std::isfinite(sx) && std::isfinite(sy))
                max_speed = std::max(max_speed, std::max(sx, sy));
        }
    }

    double global_max = 0.0;
    MPI_Allreduce(&max_speed, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (global_max <= 0.0) {
        throw std::runtime_error("compute_dt_mpi: non-positive maximum wave speed.");
    }

    // Set GLM cleaning speed = max MHD signal speed, identically on every rank.
    phys::ch_glm = global_max;

    return cfl * std::min(grid.dx(), grid.dy()) / global_max;
}

// ============================================================
// Second-order MUSCL-Hancock, x-then-y dimensional splitting (MPI).
//
// Steps (identical to advance_cpu() in solver_cpu.cpp, except
// steps 3 and 7 use directional MPI halo exchange in place of the matching
// single-domain boundary refresh):
//   1. Fill x-face cache from Uold
//   2. x-update: Uold -> Utmp (interior only)
//   3. y-halo exchange on Utmp
//   4. Fill y-face cache from Utmp
//   5. y-update: Utmp -> Unew (interior only)
//   6. Mixed-GLM psi damping on Unew
//   7. x-halo exchange on Unew for the next timestep
// ============================================================

void advance_mpi(
    const Grid2D&        Uold,
    Grid2D&              Utmp,
    Grid2D&              Unew,
    double               dt,
    MpiWorkspace&        ws,
    RiemannSolver        solver,
    const BoundaryConfig& bc,
    const MpiDomain&      dom
) {
    if (!ws.is_initialized()) {
        throw std::runtime_error(
            "advance_mpi: MpiWorkspace not initialised. "
            "Call ws.init(dom.nx_local, dom.ny_local) before the time loop.");
    }

    const int ib = Uold.i_begin();
    const int ie = Uold.i_end();
    const int jb = Uold.j_begin();
    const int je = Uold.j_end();

    const int nx_faces = (ie - ib) + 1;
    const int nx_cells  = ie - ib;

    const double dt_over_dx = dt / Uold.dx();
    const double dt_over_dy = dt / Uold.dy();

    const int total_nx = Uold.total_nx();
    const int total_ny = Uold.total_ny();
    const std::size_t total_cells = static_cast<std::size_t>(total_nx) * total_ny;
    ws.state_cache.resize(total_cells);
    ws.recon_L_cache.resize(total_cells);
    ws.recon_R_cache.resize(total_cells);

    // ----------------------------------------------------------
    // Steps 1-2: x-sweep  (Uold -> Utmp)
    // ----------------------------------------------------------

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int jj = 0; jj < total_ny; ++jj)
        for (int ii = 0; ii < total_nx; ++ii)
            ws.state_cache[static_cast<std::size_t>(jj) * total_nx + ii] =
                Uold(ii, jj);

    fill_recon_x_cache(ws.state_cache, ib, ie, jb, je, total_nx, dt_over_dx,
                       ws.recon_L_cache, ws.recon_R_cache);
    fill_x_face_cache(Uold, ws.recon_L_cache, ws.recon_R_cache, ws.fx_cache, solver);

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = jb; j < je; ++j) {
        for (int i = ib; i < ie; ++i) {
            const int local_j        = j - jb;
            const int local_i_face_m = (i - 1) - (ib - 1);
            const int local_i_face_p =  i      - (ib - 1);

            Utmp(i, j) = Uold(i, j)
                - dt_over_dx * (
                    ws.fx_cache[xface_idx(local_j, local_i_face_p, nx_faces)] -
                    ws.fx_cache[xface_idx(local_j, local_i_face_m, nx_faces)]
                );
        }
    }

    // Step 3: the y sweep only needs bottom/top halo rows of Utmp.
    exchange_halo_y(Utmp, dom, bc);

    // ----------------------------------------------------------
    // Steps 4-5: y-sweep  (Utmp -> Unew)
    // ----------------------------------------------------------

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int jj = 0; jj < total_ny; ++jj)
        for (int ii = 0; ii < total_nx; ++ii)
            ws.state_cache[static_cast<std::size_t>(jj) * total_nx + ii] =
                Utmp(ii, jj);

    fill_recon_y_cache(ws.state_cache, ib, ie, jb, je, total_nx, dt_over_dy,
                       ws.recon_L_cache, ws.recon_R_cache);
    fill_y_face_cache(Utmp, ws.recon_L_cache, ws.recon_R_cache, ws.fy_cache, solver);

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = jb; j < je; ++j) {
        for (int i = ib; i < ie; ++i) {
            const int local_i        = i - ib;
            const int local_j_face_m = (j - 1) - (jb - 1);
            const int local_j_face_p =  j      - (jb - 1);

            Unew(i, j) = Utmp(i, j)
                - dt_over_dy * (
                    ws.fy_cache[yface_idx(local_j_face_p, local_i, nx_cells)] -
                    ws.fy_cache[yface_idx(local_j_face_m, local_i, nx_cells)]
                );
        }
    }

    // Step 6: Mixed-GLM psi damping (Dedner eq. 45)
    apply_psi_damping(Unew, dt);

    // Step 7: the next timestep starts with an x sweep.  Refresh only the
    // left/right halo columns, after damping, so halo psi is current.
    exchange_halo_x(Unew, dom, bc);
}
