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

inline bool positive_conserved(const Conserved& U) {
    const double msq = U.rhou*U.rhou + U.rhov*U.rhov + U.rhow*U.rhow;
    const double mag = 0.5 * (U.Bx*U.Bx + U.By*U.By + U.Bz*U.Bz);
    const double lhs = U.rho * (U.E - mag) - 0.5 * msq;
    return U.rho > 0.0 && lhs > 0.0;
}

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

inline int xface_idx(int local_j, int local_i_face, int nx_faces) {
    return local_j * nx_faces + local_i_face;
}

inline int yface_idx(int local_j_face, int local_i, int nx_cells) {
    return local_j_face * nx_cells + local_i;
}

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

} // namespace

double compute_dt_cpu(const Grid2D& grid, double cfl) {
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

    if (max_speed <= 0.0) {
        throw std::runtime_error("compute_dt_cpu: non-positive maximum wave speed.");
    }

    phys::ch_glm = max_speed;

    return cfl * std::min(grid.dx(), grid.dy()) / max_speed;
}

void advance_cpu(
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
            "advance_cpu: CpuWorkspace not initialised. "
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
    ws.state_cache.resize(total_cells);
    ws.recon_L_cache.resize(total_cells);
    ws.recon_R_cache.resize(total_cells);

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

    // The y sweep only reads bottom/top ghosts of Utmp.
    apply_boundary_y(Utmp, bc);

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

    apply_psi_damping(Unew, dt);
    // The next timestep starts with an x sweep, so only left/right ghosts
    // are required.  Refresh them after damping so ghost psi is current.
    apply_boundary_x(Unew, bc);
}

void advance_cpu(
    const Grid2D& Uold,
    Grid2D&       Utmp,
    Grid2D&       Unew,
    double        dt,
    CpuWorkspace& ws
) {
    static const BoundaryConfig all_transmissive{};
    advance_cpu(Uold, Utmp, Unew, dt, ws, RiemannSolver::HLL, all_transmissive);
}
