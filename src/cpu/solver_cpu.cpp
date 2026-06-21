#include "cpu/solver_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cpu/boundary_cpu.hpp"
#include "physics.hpp"
#include "riemann.hpp"
#include "types.hpp"

namespace {

constexpr double kRhoFloor = 1.0e-12;
constexpr double kPFloor   = 1.0e-12;

// ============================================================
// Minmod limiter for scalar and all 9 primitive components
// ============================================================

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

// ============================================================
// Physical validity checks for MHD primitive state
// ============================================================

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

// ============================================================
// Minmod slope estimate in the given direction
// ============================================================

inline Primitive limited_slope(
    const Primitive& Wm,
    const Primitive& Wc,
    const Primitive& Wp
) {
    return minmod_primitive(Wc - Wm, Wp - Wc);
}

// ============================================================
// MUSCL-Hancock half-step reconstruction for one cell triplet.
// ============================================================

inline void reconstruct_cell_muscl_hancock(
    const Conserved& Um,
    const Conserved& Uc,
    const Conserved& Up,
    double           dt_over_d,
    Direction        dir,
    Conserved&       U_left_star,
    Conserved&       U_right_star
) {
    const Primitive Wm = phys::cons_to_prim(Um);
    const Primitive Wc = phys::cons_to_prim(Uc);
    const Primitive Wp = phys::cons_to_prim(Up);

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

// ============================================================
// MUSCL-Hancock interface fluxes — each calls reconstruct on
// two neighbouring cells and passes the interface states to the
// chosen Riemann solver.
// ============================================================

inline Conserved muscl_hancock_flux_x(
    const Grid2D& U,
    int i, int j,
    double dt_over_dx,
    RiemannSolver solver
) {
    Conserved Ui_L, Ui_R, Uip1_L, Uip1_R;

    reconstruct_cell_muscl_hancock(
        U(i - 1, j), U(i, j), U(i + 1, j),
        dt_over_dx, Direction::X,
        Ui_L, Ui_R
    );
    reconstruct_cell_muscl_hancock(
        U(i, j), U(i + 1, j), U(i + 2, j),
        dt_over_dx, Direction::X,
        Uip1_L, Uip1_R
    );

    return riemann_flux(Ui_R, Uip1_L, Direction::X, solver, phys::ch_glm);
}

inline Conserved muscl_hancock_flux_y(
    const Grid2D& U,
    int i, int j,
    double dt_over_dy,
    RiemannSolver solver
) {
    Conserved Uj_L, Uj_R, Ujp1_L, Ujp1_R;

    reconstruct_cell_muscl_hancock(
        U(i, j - 1), U(i, j), U(i, j + 1),
        dt_over_dy, Direction::Y,
        Uj_L, Uj_R
    );
    reconstruct_cell_muscl_hancock(
        U(i, j), U(i, j + 1), U(i, j + 2),
        dt_over_dy, Direction::Y,
        Ujp1_L, Ujp1_R
    );

    return riemann_flux(Uj_R, Ujp1_L, Direction::Y, solver, phys::ch_glm);
}

// ============================================================
// Cache index helpers
// ============================================================

inline int xface_idx(int local_j, int local_i_face, int nx_faces) {
    return local_j * nx_faces + local_i_face;
}

inline int yface_idx(int local_j_face, int local_i, int nx_cells) {
    return local_j_face * nx_cells + local_i;
}

// ============================================================
// Fill x-face flux cache for all interior faces.
// Face local_i_face = 0 is the left ghost interface at i = ib-1.
// ============================================================

void fill_x_face_cache(
    const Grid2D& Uin,
    double dt,
    std::vector<Conserved>& fx_cache,
    RiemannSolver solver
) {
    const int ib = Uin.i_begin();
    const int ie = Uin.i_end();
    const int jb = Uin.j_begin();
    const int je = Uin.j_end();

    const int    nx_faces    = (ie - ib) + 1;
    const double dt_over_dx  = dt / Uin.dx();

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = jb; j < je; ++j) {
        for (int i = ib - 1; i < ie; ++i) {
            const int local_j      = j - jb;
            const int local_i_face = i - (ib - 1);

            fx_cache[xface_idx(local_j, local_i_face, nx_faces)] =
                muscl_hancock_flux_x(Uin, i, j, dt_over_dx, solver);
        }
    }
}

// ============================================================
// Fill y-face flux cache for all interior faces.
// Face local_j_face = 0 is the bottom ghost interface at j = jb-1.
// ============================================================

void fill_y_face_cache(
    const Grid2D& Uin,
    double dt,
    std::vector<Conserved>& fy_cache,
    RiemannSolver solver
) {
    const int ib = Uin.i_begin();
    const int ie = Uin.i_end();
    const int jb = Uin.j_begin();
    const int je = Uin.j_end();

    const int    nx_cells   = ie - ib;
    const double dt_over_dy = dt / Uin.dy();

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = jb - 1; j < je; ++j) {
        for (int i = ib; i < ie; ++i) {
            const int local_j_face = j - (jb - 1);
            const int local_i      = i - ib;

            fy_cache[yface_idx(local_j_face, local_i, nx_cells)] =
                muscl_hancock_flux_y(Uin, i, j, dt_over_dy, solver);
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

} // namespace

// ============================================================
// Public API
// ============================================================

double compute_dt(const Grid2D& grid, double cfl) {
    double max_speed = 0.0;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(max:max_speed) schedule(static)
#endif
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const Primitive V = phys::cons_to_prim(grid(i, j));

            // Skip non-physical cells (shouldn't occur but be safe)
            if (!std::isfinite(V.rho) || !std::isfinite(V.p) ||
                V.rho <= 0.0 || V.p <= 0.0) continue;

            const double sx = phys::max_signal_speed_x(V, phys::ch_glm);
            const double sy = phys::max_signal_speed_y(V, phys::ch_glm);
            if (std::isfinite(sx) && std::isfinite(sy))
                max_speed = std::max(max_speed, std::max(sx, sy));
        }
    }

    if (max_speed <= 0.0) {
        throw std::runtime_error("compute_dt: non-positive maximum wave speed.");
    }

    // Set GLM cleaning speed = max MHD signal speed (eq. 30 / Section 4)
    phys::ch_glm = max_speed;

    return cfl * std::min(grid.dx(), grid.dy()) / max_speed;
}

// ============================================================
// Second-order MUSCL-Hancock, x-then-y dimensional splitting.
//
// Steps:
//   1. Fill x-face cache from Uold
//   2. x-update: Uold → Utmp (interior only)
//   3. BC on Utmp
//   4. Fill y-face cache from Utmp
//   5. y-update: Utmp → Unew (interior only)
//   6. BC on Unew
//   7. Mixed-GLM ψ damping on Unew
// ============================================================

void advance_second_order(
    const Grid2D&        Uold,
    Grid2D&              Utmp,
    Grid2D&              Unew,
    double               dt,
    CpuWorkspace&        ws,
    RiemannSolver        solver,
    const BoundaryConfig& bc
) {
    if (!ws.is_initialized()) {
        throw std::runtime_error(
            "advance_second_order: CpuWorkspace not initialised. "
            "Call ws.init(cfg.nx, cfg.ny) before the time loop."
        );
    }

    const int ib = Uold.i_begin();
    const int ie = Uold.i_end();
    const int jb = Uold.j_begin();
    const int je = Uold.j_end();

    const int nx_faces = (ie - ib) + 1;
    const int nx_cells =  ie - ib;

    const double dt_over_dx = dt / Uold.dx();
    const double dt_over_dy = dt / Uold.dy();

    // ----------------------------------------------------------
    // Steps 1-2: x-sweep  (Uold → Utmp)
    // ----------------------------------------------------------

    fill_x_face_cache(Uold, dt, ws.fx_cache, solver);

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

    // Safety: reset any NaN cells from the x-sweep to their old values.
    // One NaN flux can silently corrupt every cell in a column via the
    // y-sweep (fmax(NaN,x)=x masks the damage in diagnostics).
    for (int j = jb; j < je; ++j)
        for (int i = ib; i < ie; ++i)
            if (!std::isfinite(Utmp(i,j).rho) || !std::isfinite(Utmp(i,j).E))
                Utmp(i,j) = Uold(i,j);

    // Step 3: BC on Utmp — Dirichlet cells come from Uold
    copy_ghost_cells(Uold, Utmp);
    apply_boundary(Utmp, bc);

    // ----------------------------------------------------------
    // Steps 4-5: y-sweep  (Utmp → Unew)
    // ----------------------------------------------------------

    fill_y_face_cache(Utmp, dt, ws.fy_cache, solver);

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

    // Safety: reset any NaN cells from the y-sweep.
    for (int j = jb; j < je; ++j)
        for (int i = ib; i < ie; ++i)
            if (!std::isfinite(Unew(i,j).rho) || !std::isfinite(Unew(i,j).E))
                Unew(i,j) = Utmp(i,j);

    // Step 6: BC on Unew — Dirichlet cells come from Utmp
    copy_ghost_cells(Utmp, Unew);
    apply_boundary(Unew, bc);

    // Step 7: Mixed-GLM ψ damping (Dedner eq. 45)
    apply_psi_damping(Unew, dt);
}

// Backward-compatible convenience overload: HLL + all-transmissive
void advance_second_order(
    const Grid2D& Uold,
    Grid2D&       Utmp,
    Grid2D&       Unew,
    double        dt,
    CpuWorkspace& ws
) {
    static const BoundaryConfig all_transmissive{};
    advance_second_order(Uold, Utmp, Unew, dt, ws, RiemannSolver::HLL, all_transmissive);
}
