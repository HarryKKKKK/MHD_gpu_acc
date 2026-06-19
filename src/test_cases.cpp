#include "test_cases.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <stdexcept>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "physics.hpp"

// ============================================================
// Helpers
// ============================================================

namespace {

// Build a conserved MHD state from primitive variables.
// psi = 0 at t=0 (no divergence error initially).
inline Conserved make_mhd(
    double rho,
    double ux, double uy, double uz,
    double Bx,  double By,  double Bz,
    double p,
    double gamma
) {
    const double Bmag2 = Bx*Bx + By*By + Bz*Bz;
    const double ke    = 0.5 * rho * (ux*ux + uy*uy + uz*uz);
    const double E     = p / (gamma - 1.0) + ke + 0.5*Bmag2;
    return Conserved(rho, rho*ux, rho*uy, rho*uz, Bx, By, Bz, E, 0.0);
}

// ============================================================
// 1. Peak in Bx  (Table I, γ = 5/3)
//
// Domain: [-0.5, 1.5] × [-0.5, 1.5], periodic.
// Initial conditions: uniform flow with a localised peak in Bx.
//
// rho = 1,  ux = uy = 1,  uz = 0
// Bx  = r(x²+y²) / sqrt(4π)
// By  = 0
// Bz  = 1 / sqrt(4π)
// p   = 6
//
// r(s) = 4096 s⁴ - 128 s² + 1   (smooth bump centred at origin)
// ============================================================
Conserved peak_bx_state(double x, double y) {
    constexpr double gamma = 5.0 / 3.0;
    constexpr double sqrt4pi = 2.0 * 1.7724538509055159;  // sqrt(4π)

    const double s  = x*x + y*y;
    // Compact-support bump: (64s-1)^2 for s<1/64, zero outside.
    // Radius of support: r = 1/8 = 0.125. Without the cutoff the
    // function grows as s^2 and produces unphysical B at domain edges.
    const double rs = (s < 1.0/64.0) ? (4096.0*s*s - 128.0*s + 1.0) : 0.0;

    const double rho = 1.0;
    const double ux  = 1.0;
    const double uy  = 1.0;
    const double uz  = 0.0;
    const double Bx  = rs / sqrt4pi;
    const double By  = 0.0;
    const double Bz  = 1.0 / sqrt4pi;
    const double p   = 6.0;

    return make_mhd(rho, ux, uy, uz, Bx, By, Bz, p, gamma);
}

// ============================================================
// 2. 1D Riemann problem  (Table I, γ = 5/3)
//
// Domain: [-0.5, 0.5] × [-0.25, 0.25]
// BC: Neumann on left/right, periodic on top/bottom.
//
// Originally from Dai & Woodward (1994); also used by Tóth (2000).
// ============================================================
Conserved riemann1d_state(double x, double /*y*/) {
    constexpr double gamma   = 5.0 / 3.0;
    constexpr double sqrt4pi = 2.0 * 1.7724538509055159;

    if (x < 0.0) {
        return make_mhd(
            1.0,  10.0, 0.0, 0.0,
            5.0/sqrt4pi,  5.0/sqrt4pi, 0.0,
            20.0, gamma
        );
    } else {
        return make_mhd(
            1.0, -10.0, 0.0, 0.0,
            5.0/sqrt4pi, -5.0/sqrt4pi, 0.0,
            1.0, gamma
        );
    }
}

// ============================================================
// 3. Shock reflection  (Table I, γ = 1.4)
//
// Domain: [-1, 1] × [-0.5, 0.5]
// BC: Dirichlet (Ul) on top + left, reflecting bottom, Neumann right.
//
// Incident shock at 29° to the x-axis.
// ============================================================
Conserved shock_reflection_state(double /*x*/, double /*y*/) {
    // Entire domain initialised with upstream (left) state
    constexpr double gamma = 1.4;
    return make_mhd(
        1.0, 2.9, 0.0, 0.0,
        0.5, 0.0, 0.0,
        5.0/7.0, gamma
    );
}

// ============================================================
// 4. 2D Riemann problem  (Table I, γ = 5/3)
//
// Domain: [-1, 1] × [-1, 1]
// BC: Dirichlet from 1D Riemann solutions on all four sides.
// Four quadrants with different initial states.
// ============================================================
Conserved riemann2d_state(double x, double y) {
    // Quadrant data (Table I, conserved variables; gamma=5/3 implicit in E values)
    // Quadrant I: x>0, y>0
    // Quadrant II: x<0, y>0
    // Quadrant III: x<0, y<0
    // Quadrant IV: x>0, y<0

    double rho, rhou, rhov, rhow, Bx, By, Bz, E;

    if (x >= 0.0 && y >= 0.0) {
        // Quadrant I
        rho  = 0.9308; rhou = 1.4557; rhov = -0.4633; rhow = 0.0575;
        Bx   = 0.3501; By   = 0.9830; Bz   = 0.3050;
        E    = 5.0838;
    } else if (x < 0.0 && y >= 0.0) {
        // Quadrant II
        rho  = 1.0304; rhou = 1.5774; rhov = -1.0455; rhow = -0.1016;
        Bx   = 0.3501; By   = 0.5078; Bz   = 0.1576;
        E    = 5.7813;
    } else if (x < 0.0 && y < 0.0) {
        // Quadrant III
        rho  = 1.0000; rhou = 1.7500; rhov = -1.0000; rhow = 0.0000;
        Bx   = 0.5642; By   = 0.5078; Bz   = 0.2539;
        E    = 6.0000;
    } else {
        // Quadrant IV (x>=0, y<0)
        rho  = 1.8887; rhou = 0.2334; rhov = -1.7422; rhow = 0.0733;
        Bx   = 0.5642; By   = 0.9830; Bz   = 0.4915;
        E    = 12.999;
    }

    return Conserved(rho, rhou, rhov, rhow, Bx, By, Bz, E, 0.0);
}

// ============================================================
// 5. Kelvin-Helmholtz instability  (Table I, γ = 1.4)
//
// Domain: [0, 1] × [-1, 1], periodic.
// Shear layer with a sinusoidal perturbation to trigger KH.
// ============================================================
Conserved kh_state(double x, double y) {
    constexpr double gamma = 1.4;

    // u_x = 5 * (tanh(20*(y+0.5)) - (tanh(20*(y-0.5)) + 1))
    const double ux = 5.0 * (tanh(20.0*(y + 0.5))
                           - (tanh(20.0*(y - 0.5)) + 1.0));

    // u_y = 0.25 * sin(2π*x) * (exp(-100*(y+0.5)^2) - exp(-100*(y-0.5)^2))
    const double uy = 0.25 * sin(2.0 * M_PI * x)
                    * (exp(-100.0*(y + 0.5)*(y + 0.5))
                     - exp(-100.0*(y - 0.5)*(y - 0.5)));

    const double rho = 1.0;
    const double uz  = 0.0;
    const double Bx  = 1.0;
    const double By  = 0.0;
    const double Bz  = 0.0;
    const double p   = 50.0;

    return make_mhd(rho, ux, uy, uz, Bx, By, Bz, p, gamma);
}

} // namespace

// ============================================================
// Public API
// ============================================================

CaseId parse_case_id(const std::string& name) {
    if (name == "peak_bx")           return CaseId::PeakBx;
    if (name == "riemann1d")         return CaseId::Riemann1D;
    if (name == "shock_reflection")  return CaseId::ShockReflection;
    if (name == "riemann2d")         return CaseId::Riemann2D;
    if (name == "kelvin_helmholtz")  return CaseId::KelvinHelmholtz;

    throw std::runtime_error(
        "Unknown MHD test case: '" + name + "'. "
        "Valid names: peak_bx, riemann1d, shock_reflection, "
        "riemann2d, kelvin_helmholtz."
    );
}

std::string case_id_to_string(CaseId id) {
    switch (id) {
        case CaseId::PeakBx:          return "peak_bx";
        case CaseId::Riemann1D:       return "riemann1d";
        case CaseId::ShockReflection: return "shock_reflection";
        case CaseId::Riemann2D:       return "riemann2d";
        case CaseId::KelvinHelmholtz: return "kelvin_helmholtz";
    }
    throw std::runtime_error("Unhandled CaseId in case_id_to_string.");
}

CaseConfig get_case_config(CaseId id) {
    switch (id) {

        case CaseId::PeakBx:
            // Periodic domain; 65536 triangles ≈ 256×256 cells on a square
            return CaseConfig{
                256, 256, 2,
                -0.5, 1.5, -0.5, 1.5,
                /*cfl=*/0.3, /*t_end=*/1.0,
                /*gamma=*/5.0/3.0,
                BoundaryConfig{
                    BoundaryType::Periodic, BoundaryType::Periodic,
                    BoundaryType::Periodic, BoundaryType::Periodic
                }
            };

        case CaseId::Riemann1D:
            // Neumann L/R, periodic top/bottom; 16384 triangles ≈ 256×128
            return CaseConfig{
                256, 128, 2,
                -0.5, 0.5, -0.25, 0.25,
                /*cfl=*/0.3, /*t_end=*/0.08,
                /*gamma=*/5.0/3.0,
                BoundaryConfig{
                    BoundaryType::Transmissive, BoundaryType::Transmissive,
                    BoundaryType::Periodic,     BoundaryType::Periodic
                }
            };

        case CaseId::ShockReflection:
            // Reflecting bottom wall; transmissive on other sides
            return CaseConfig{
                200, 100, 2,
                -1.0, 1.0, -0.5, 0.5,
                /*cfl=*/0.3, /*t_end=*/2.0,
                /*gamma=*/1.4,
                BoundaryConfig{
                    BoundaryType::Transmissive, BoundaryType::Transmissive,
                    BoundaryType::Reflecting,   BoundaryType::Transmissive
                }
            };

        case CaseId::Riemann2D:
            // Transmissive on all sides (standard for 4-quadrant Riemann)
            return CaseConfig{
                200, 200, 2,
                -1.0, 1.0, -1.0, 1.0,
                /*cfl=*/0.3, /*t_end=*/0.25,
                /*gamma=*/5.0/3.0,
                BoundaryConfig{
                    BoundaryType::Transmissive, BoundaryType::Transmissive,
                    BoundaryType::Transmissive, BoundaryType::Transmissive
                }
            };

        case CaseId::KelvinHelmholtz:
            // Fully periodic
            return CaseConfig{
                200, 400, 2,
                0.0, 1.0, -1.0, 1.0,
                /*cfl=*/0.3, /*t_end=*/0.5,
                /*gamma=*/1.4,
                BoundaryConfig{
                    BoundaryType::Periodic, BoundaryType::Periodic,
                    BoundaryType::Periodic, BoundaryType::Periodic
                }
            };
    }
    throw std::runtime_error("Unhandled CaseId in get_case_config.");
}

CaseConfig get_case_config(const std::string& name) {
    return get_case_config(parse_case_id(name));
}

CaseConfig get_n_case_config(CaseId id, int n) {
    if (n < 1)
        throw std::runtime_error("get_n_case_config: n must be >= 1.");
    CaseConfig cfg = get_case_config(id);
    cfg.nx *= n;
    cfg.ny *= n;
    return cfg;
}

CaseConfig get_n_case_config(const std::string& name, int n) {
    return get_n_case_config(parse_case_id(name), n);
}

Conserved initial_state_at(CaseId id, double x, double y) {
    switch (id) {
        case CaseId::PeakBx:          return peak_bx_state(x, y);
        case CaseId::Riemann1D:       return riemann1d_state(x, y);
        case CaseId::ShockReflection: return shock_reflection_state(x, y);
        case CaseId::Riemann2D:       return riemann2d_state(x, y);
        case CaseId::KelvinHelmholtz: return kh_state(x, y);
    }
    throw std::runtime_error("Unhandled CaseId in initial_state_at.");
}

Conserved initial_state_at(const std::string& name, double x, double y) {
    return initial_state_at(parse_case_id(name), x, y);
}
