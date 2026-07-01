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
// solver_cpu.cpp (MPI halo exchange instead of copy_ghost_cells() /
// apply_boundary() on a single self-contained grid) — see
// exchange_halo_x/y below.
// ============================================================

namespace {

constexpr double kRhoFloor = 1.0e-12;
constexpr double kPFloor   = 1.0e-12;

// ---- Minmod limiter for scalar and all 9 primitive components ----

inline double minmod_scalar(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (a > 0.0) ? std::min(a, b) : std::max(a, b);
}

inline Primitive minmod_primitive(const Primitive& a, const Primitive& b) {
    return Primitive(
        minmod_scalar(a.rho, b.rho),
        minmod_scalar(a.u,   b.u),
        minmod_scalar(a.v,   b.v),
        minmod_scalar(a.w,   b.w),
        minmod_scalar(a.Bx,  b.Bx),
        minmod_scalar(a.By,  b.By),
        minmod_scalar(a.Bz,  b.Bz),
        minmod_scalar(a.p,   b.p),
        minmod_scalar(a.psi, b.psi)
    );
}

// ---- Physical validity checks for MHD primitive state ----

inline bool is_physical(const Primitive& V) {
    return std::isfinite(V.rho) && V.rho > kRhoFloor &&
           std::isfinite(V.p)   && V.p   > kPFloor   &&
           std::isfinite(V.u)   && std::isfinite(V.v) &&
           std::isfinite(V.w)   &&
           std::isfinite(V.Bx)  && std::isfinite(V.By) &&
           std::isfinite(V.Bz)  && std::isfinite(V.psi);
}

inline Primitive enforce_physical_primitive(
    const Primitive& candidate,
    const Primitive& fallback
) {
    return is_physical(candidate) ? candidate : fallback;
}

inline Conserved enforce_physical_conserved(
    const Conserved& candidate,
    const Conserved& fallback
) {
    return is_physical(phys::cons_to_prim(candidate)) ? candidate : fallback;
}

// ---- Minmod slope estimate in the given direction ----

inline Primitive limited_slope(
    const Primitive& Wm,
    const Primitive& Wc,
    const Primitive& Wp
) {
    return minmod_primitive(Wc - Wm, Wp - Wc);
}

// ---- MUSCL-Hancock half-step reconstruction for one cell triplet ----

inline void reconstruct_cell_muscl_hancock(
    const Primitive& Wm,
    const Primitive& Wc,
    const Primitive& Wp,
    double           dt_over_d,
    Direction        dir,
    Conserved&       U_left_star,
    Conserved&       U_right_star
) {
    const Primitive slope = limited_slope(Wm, Wc, Wp);

    Primitive W_left  = Wc - 0.5 * slope;
    Primitive W_right = Wc + 0.5 * slope;

    W_left  = enforce_physical_primitive(W_left,  Wc);
    W_right = enforce_physical_primitive(W_right, Wc);

    const Conserved U_left  = phys::prim_to_cons(W_left);
    const Conserved U_right = phys::prim_to_cons(W_right);

    // Half-step in time using the directional physical flux
    const double ch = phys::ch_glm;
    const Conserved F_left  = (dir == Direction::X) ? phys::flux_x(U_left,  ch)
                                                     : phys::flux_y(U_left,  ch);
    const Conserved F_right = (dir == Direction::X) ? phys::flux_x(U_right, ch)
                                                     : phys::flux_y(U_right, ch);

    const Conserved half_update = 0.5 * dt_over_d * (F_right - F_left);

    U_left_star  = enforce_physical_conserved(U_left  - half_update, U_left);
    U_right_star = enforce_physical_conserved(U_right - half_update, U_right);
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
    const std::vector<Primitive>& W,
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
                W[static_cast<std::size_t>(j) * total_nx + (i - 1)],
                W[idx],
                W[static_cast<std::size_t>(j) * total_nx + (i + 1)],
                dt_over_dx, Direction::X,
                recon_L[idx], recon_R[idx]
            );
        }
    }
}

// ---- Reconstruction fill (y-direction): cells j in [jb-1, je], cols i in [ib, ie) ----

void fill_recon_y_cache(
    const std::vector<Primitive>& W,
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
                W[static_cast<std::size_t>(j - 1) * total_nx + i],
                W[idx],
                W[static_cast<std::size_t>(j + 1) * total_nx + i],
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
    const double factor = std::exp(-dt * phys::ch_glm / phys::cr_glm);
    if (factor >= 1.0) return;  // ch=0 or cr=0: no damping needed

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
// apply_boundary()/copy_ghost_cells() in head/cpu/boundary_cpu.hpp,
// restricted to one side at a time so exchange_halo_x/y can call them
// only on ranks that own a genuine global boundary (MPI_PROC_NULL
// neighbour). Periodic BoundaryType never reaches these functions: a
// periodic axis is given periods=1 in MPI_Cart_create, so every rank
// always has a real neighbour (itself, if running on a single rank
// along that axis) and the wrap-around is handled by the halo exchange
// itself, exactly reproducing the single-rank periodic case.
// ============================================================

void apply_physical_left(Grid2D& grid, BoundaryType type) {
    const int ng = grid.ng();
    const int ib = grid.i_begin();
    const int total_ny = grid.total_ny();
    for (int j = 0; j < total_ny; ++j) {
        for (int g = 0; g < ng; ++g) {
            if (type == BoundaryType::Transmissive) {
                grid(ib - 1 - g, j) = grid(ib, j);
            } else if (type == BoundaryType::Reflecting) {
                Conserved m = grid(ib + g, j);
                m.rhou = -m.rhou;
                m.Bx   = -m.Bx;
                grid(ib - 1 - g, j) = m;
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
            } else if (type == BoundaryType::Reflecting) {
                Conserved m = grid(ie - 1 - g, j);
                m.rhou = -m.rhou;
                m.Bx   = -m.Bx;
                grid(ie + g, j) = m;
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
            } else if (type == BoundaryType::Reflecting) {
                Conserved m = grid(i, jb + g);
                m.rhov = -m.rhov;
                m.By   = -m.By;
                grid(i, jb - 1 - g) = m;
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
            } else if (type == BoundaryType::Reflecting) {
                Conserved m = grid(i, je - 1 - g);
                m.rhov = -m.rhov;
                m.By   = -m.By;
                grid(i, je + g) = m;
            }
        }
    }
}

// Dirichlet ghost cells: carry forward the previous-step values from a
// reference grid, exactly like copy_ghost_cells() in boundary_cpu.hpp.

void copy_dirichlet_left(const Grid2D& src, Grid2D& dst) {
    const int ng = src.ng();
    const int ib = src.i_begin();
    const int total_ny = src.total_ny();
    for (int j = 0; j < total_ny; ++j)
        for (int g = 0; g < ng; ++g)
            dst(ib - 1 - g, j) = src(ib - 1 - g, j);
}

void copy_dirichlet_right(const Grid2D& src, Grid2D& dst) {
    const int ng = src.ng();
    const int ie = src.i_end();
    const int total_ny = src.total_ny();
    for (int j = 0; j < total_ny; ++j)
        for (int g = 0; g < ng; ++g)
            dst(ie + g, j) = src(ie + g, j);
}

void copy_dirichlet_bottom(const Grid2D& src, Grid2D& dst) {
    const int ng = src.ng();
    const int jb = src.j_begin();
    const int total_nx = src.total_nx();
    for (int i = 0; i < total_nx; ++i)
        for (int g = 0; g < ng; ++g)
            dst(i, jb - 1 - g) = src(i, jb - 1 - g);
}

void copy_dirichlet_top(const Grid2D& src, Grid2D& dst) {
    const int ng = src.ng();
    const int je = src.j_end();
    const int total_nx = src.total_nx();
    for (int i = 0; i < total_nx; ++i)
        for (int g = 0; g < ng; ++g)
            dst(i, je + g) = src(i, je + g);
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

    const double dx_global = (cfg.x_max - cfg.x_min) / cfg.nx;
    const double dy_global = (cfg.y_max - cfg.y_min) / cfg.ny;

    const double x_min_local = cfg.x_min + dom.i_start * dx_global;
    const double x_max_local = x_min_local + dom.nx_local * dx_global;
    const double y_min_local = cfg.y_min + dom.j_start * dy_global;
    const double y_max_local = y_min_local + dom.ny_local * dy_global;

    Grid2D grid(
        dom.nx_local, dom.ny_local, dom.ng,
        x_min_local, x_max_local, y_min_local, y_max_local
    );
    initialise_grid(grid, case_name);
    return grid;
}

// ============================================================
// Halo exchange
// ============================================================

void exchange_halo_x(
    Grid2D& grid, const MpiDomain& dom, const BoundaryConfig& bc,
    const Grid2D* dirichlet_src
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
        if (bc.left == BoundaryType::Dirichlet) {
            if (dirichlet_src) copy_dirichlet_left(*dirichlet_src, grid);
        } else {
            apply_physical_left(grid, bc.left);
        }
    }
    if (dom.nbr_right == MPI_PROC_NULL) {
        if (bc.right == BoundaryType::Dirichlet) {
            if (dirichlet_src) copy_dirichlet_right(*dirichlet_src, grid);
        } else {
            apply_physical_right(grid, bc.right);
        }
    }
}

void exchange_halo_y(
    Grid2D& grid, const MpiDomain& dom, const BoundaryConfig& bc,
    const Grid2D* dirichlet_src
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
        if (bc.bottom == BoundaryType::Dirichlet) {
            if (dirichlet_src) copy_dirichlet_bottom(*dirichlet_src, grid);
        } else {
            apply_physical_bottom(grid, bc.bottom);
        }
    }
    if (dom.nbr_up == MPI_PROC_NULL) {
        if (bc.top == BoundaryType::Dirichlet) {
            if (dirichlet_src) copy_dirichlet_top(*dirichlet_src, grid);
        } else {
            apply_physical_top(grid, bc.top);
        }
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

            const double sx = phys::max_signal_speed_x(V, phys::ch_glm);
            const double sy = phys::max_signal_speed_y(V, phys::ch_glm);
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
// Steps (identical to advance_second_order() in solver_cpu.cpp, except
// steps 3 and 6 use MPI halo exchange in place of
// copy_ghost_cells()+apply_boundary()):
//   1. Fill x-face cache from Uold
//   2. x-update: Uold -> Utmp (interior only)
//   3. Halo exchange on Utmp
//   4. Fill y-face cache from Utmp
//   5. y-update: Utmp -> Unew (interior only)
//   6. Halo exchange on Unew
//   7. Mixed-GLM psi damping on Unew
// ============================================================

void advance_second_order_mpi(
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
            "advance_second_order_mpi: MpiWorkspace not initialised. "
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
    ws.prim_cache.resize(total_cells);
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
            ws.prim_cache[static_cast<std::size_t>(jj) * total_nx + ii] =
                phys::cons_to_prim(Uold(ii, jj));

    fill_recon_x_cache(ws.prim_cache, ib, ie, jb, je, total_nx, dt_over_dx,
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
            if (!std::isfinite(Utmp(i,j).rho) || !std::isfinite(Utmp(i,j).E))
                Utmp(i,j) = Uold(i,j);
        }
    }

    // Step 3: refresh Utmp's ghost cells (halo exchange + local physical BC)
    exchange_halo_full(Utmp, dom, bc, &Uold);

    // ----------------------------------------------------------
    // Steps 4-5: y-sweep  (Utmp -> Unew)
    // ----------------------------------------------------------

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int jj = 0; jj < total_ny; ++jj)
        for (int ii = 0; ii < total_nx; ++ii)
            ws.prim_cache[static_cast<std::size_t>(jj) * total_nx + ii] =
                phys::cons_to_prim(Utmp(ii, jj));

    fill_recon_y_cache(ws.prim_cache, ib, ie, jb, je, total_nx, dt_over_dy,
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
            if (!std::isfinite(Unew(i,j).rho) || !std::isfinite(Unew(i,j).E))
                Unew(i,j) = Utmp(i,j);
        }
    }

    // Step 6: refresh Unew's ghost cells (halo exchange + local physical BC)
    exchange_halo_full(Unew, dom, bc, &Utmp);

    // Step 7: Mixed-GLM psi damping (Dedner eq. 45)
    apply_psi_damping(Unew, dt);
}
