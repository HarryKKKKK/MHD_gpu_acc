#include "cpu/solver_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cpu/boundary_cpu.hpp"
#include "diagnostics.hpp"
#include "physics.hpp"
#include "riemann.hpp"
#include "types.hpp"

namespace {

constexpr double kRhoFloor = 1.0e-12;
constexpr double kPFloor   = 1.0e-12;

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
    if (is_physical(candidate)) return candidate;
    diag::count_floor_trigger();
    return fallback;
}

inline Conserved enforce_physical_conserved(
    const Conserved& candidate,
    const Conserved& fallback
) {
    if (is_physical(phys::cons_to_prim(candidate))) return candidate;
    diag::count_floor_trigger();
    return fallback;
}

inline Primitive limited_slope(
    const Primitive& Wm,
    const Primitive& Wc,
    const Primitive& Wp
) {
    return minmod_primitive(Wc - Wm, Wp - Wc);
}

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

    // ch=0 here intentionally (matches src/gpu/solver_gpu.cu predictor step)
    const Conserved F_left  = (dir == Direction::X) ? phys::flux_x(U_left,  0.0)
                                                     : phys::flux_y(U_left,  0.0);
    const Conserved F_right = (dir == Direction::X) ? phys::flux_x(U_right, 0.0)
                                                     : phys::flux_y(U_right, 0.0);

    const Conserved half_update = 0.5 * dt_over_d * (F_right - F_left);

    U_left_star  = enforce_physical_conserved(U_left  - half_update, U_left);
    U_right_star = enforce_physical_conserved(U_right - half_update, U_right);
}

inline int xface_idx(int local_j, int local_i_face, int nx_faces) {
    return local_j * nx_faces + local_i_face;
}

inline int yface_idx(int local_j_face, int local_i, int nx_cells) {
    return local_j_face * nx_cells + local_i;
}

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
    if (factor >= 1.0) return;

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

double compute_dt(const Grid2D& grid, double cfl, double* out_max_speed) {
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

    if (out_max_speed) *out_max_speed = max_speed;

    if (max_speed <= 0.0) {
        throw std::runtime_error("compute_dt: non-positive maximum wave speed.");
    }

    phys::ch_glm = max_speed;

    return cfl * std::min(grid.dx(), grid.dy()) / max_speed;
}

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

    const int total_nx = Uold.total_nx();
    const int total_ny = Uold.total_ny();
    const std::size_t total_cells = static_cast<std::size_t>(total_nx) * total_ny;
    ws.prim_cache.resize(total_cells);
    ws.recon_L_cache.resize(total_cells);
    ws.recon_R_cache.resize(total_cells);

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
            Utmp(i,j) = enforce_physical_conserved(Utmp(i,j), Uold(i,j));
        }
    }

    copy_ghost_cells(Uold, Utmp);
    apply_boundary(Utmp, bc);

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
            Unew(i,j) = enforce_physical_conserved(Unew(i,j), Utmp(i,j));
        }
    }

    copy_ghost_cells(Utmp, Unew);
    apply_boundary(Unew, bc);

    apply_psi_damping(Unew, dt);
}

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
