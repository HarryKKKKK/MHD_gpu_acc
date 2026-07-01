#pragma once

// ============================================================
// Diagnostic instrumentation for the HLLC ch_glm-ratchet investigation
// (--diag-interval driver flag in main_cpu.cpp / main_gpu.cu).
//
// Everything here is either a pure read over already-computed state
// (compute_snapshot) or a side-channel counter that does not change
// which branch the caller takes (count_floor_trigger). Enabling
// diagnostics must not alter any solver numerics.
// ============================================================

#include <algorithm>
#include <cmath>
#include <limits>

#include "physics.hpp"
#include "types.hpp"

namespace diag {

// ------------------------------------------------------------
// is_physical() fallback-trigger counter (CPU / host side).
// Reconstruction code calls diag::count_floor_trigger() only from the
// "candidate rejected, falling back" branch of enforce_physical_primitive
// / enforce_physical_conserved (solver_cpu.cpp) — it is a pure side
// effect and never changes which value is returned. The GPU equivalent
// (device-side atomicAdd on d_floor_trigger_count) lives in
// src/gpu/solver_gpu.cu / head/gpu/solver_gpu.cuh instead, since it
// needs cudaMemcpyToSymbol/FromSymbol rather than a host global.
// ------------------------------------------------------------
inline long long floor_trigger_count = 0;

inline void reset_floor_trigger_count() { floor_trigger_count = 0; }

inline void count_floor_trigger() {
#ifdef _OPENMP
    #pragma omp atomic
    floor_trigger_count += 1;
#else
    ++floor_trigger_count;
#endif
}

// ------------------------------------------------------------
// Full-grid diagnostic snapshot: field extrema + discrete div(B).
// ------------------------------------------------------------
struct Snapshot {
    double max_cf_x = 0.0, max_cf_y = 0.0;
    double max_rho = -std::numeric_limits<double>::infinity();
    double min_rho =  std::numeric_limits<double>::infinity();
    double max_p   = -std::numeric_limits<double>::infinity();
    double min_p   =  std::numeric_limits<double>::infinity();
    double max_abs_Bx = 0.0, max_abs_By = 0.0, max_abs_Bz = 0.0, max_abs_psi = 0.0;
    double divB_L1 = 0.0, divB_max = 0.0;
};

// Accessor(i, j) must return a Conserved (by value or const-ref) for any
// i/j in [ib-1, ie] x [jb-1, je] — the divB centred difference reads one
// ghost cell beyond the interior range on every side, so the caller's
// ghost cells must already be valid (i.e. call this after apply_boundary
// / apply_boundary_gpu has run at least once on the state being probed).
template <typename Accessor>
Snapshot compute_snapshot(
    const Accessor& U,
    int ib, int ie, int jb, int je,
    double dx, double dy
) {
    Snapshot s;
    for (int j = jb; j < je; ++j) {
        for (int i = ib; i < ie; ++i) {
            const Conserved Cij = U(i, j);
            const Primitive V   = phys::cons_to_prim(Cij);

            s.max_rho = std::max(s.max_rho, V.rho);
            s.min_rho = std::min(s.min_rho, V.rho);
            s.max_p   = std::max(s.max_p,   V.p);
            s.min_p   = std::min(s.min_p,   V.p);
            s.max_abs_Bx  = std::max(s.max_abs_Bx,  std::fabs(V.Bx));
            s.max_abs_By  = std::max(s.max_abs_By,  std::fabs(V.By));
            s.max_abs_Bz  = std::max(s.max_abs_Bz,  std::fabs(V.Bz));
            s.max_abs_psi = std::max(s.max_abs_psi, std::fabs(V.psi));

            // Same admissibility test compute_dt()/compute_dt_gpu() use
            // before calling fast_speed_x/y, so max_cf_x/y only reflect
            // cells that actually contribute to the real CFL reduction.
            if (std::isfinite(V.rho) && V.rho > 0.0 &&
                std::isfinite(V.p)   && V.p   > 0.0) {
                s.max_cf_x = std::max(s.max_cf_x, phys::fast_speed_x(V));
                s.max_cf_y = std::max(s.max_cf_y, phys::fast_speed_y(V));
            }

            // Discrete div(B) via centred difference (needs 1 valid ghost
            // layer on every side; see Accessor contract above).
            const Conserved Cxm = U(i - 1, j);
            const Conserved Cxp = U(i + 1, j);
            const Conserved Cym = U(i, j - 1);
            const Conserved Cyp = U(i, j + 1);
            const double dBxdx = (Cxp.Bx - Cxm.Bx) / (2.0 * dx);
            const double dBydy = (Cyp.By - Cym.By) / (2.0 * dy);
            const double divB  = dBxdx + dBydy;

            s.divB_L1 += std::fabs(divB);
            s.divB_max = std::max(s.divB_max, std::fabs(divB));
        }
    }
    return s;
}

} // namespace diag
