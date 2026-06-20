#include "test_cases.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <iomanip>
#include <sstream>
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
// Kelvin-Helmholtz instability  (Table I, γ = 1.4)
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

// ============================================================
// Mach 1.22 shock–bubble interaction  (Table 1, γ = 1.4)
//
// A planar shock (Ms = 1.22) in air impinges on a spherical
// helium bubble.  Initial shock is at x = 0.005 m; the bubble
// of radius R = 0.025 m is centred at (0.035, 0.0445) m.
// No magnetic field (pure hydrodynamics embedded in the MHD solver).
//
// Post-shock state derived from Rankine–Hugoniot relations.
// Domain: [0, 0.225] × [0, 0.089] m, 500 × 197 cells.
// ============================================================
Conserved shock_bubble_state(double x, double y) {
    constexpr double gamma   = 1.4;
    constexpr double rho_air = 1.29;          // kg/m³
    constexpr double rho_He  = 0.214;         // kg/m³
    constexpr double p0      = 1.01325e5;     // Pa
    constexpr double Ms      = 1.22;

    // Bubble geometry
    constexpr double xc = 0.035, yc = 0.0445, R = 0.025;
    // Initial shock location
    constexpr double xs = 0.005;

    // Post-shock state via Rankine–Hugoniot (shock moves in +x direction)
    static const double c1   = std::sqrt(gamma * p0 / rho_air);
    static const double vs   = Ms * c1;
    static const double rho2 = rho_air * (gamma + 1.0) * Ms * Ms
                               / ((gamma - 1.0) * Ms * Ms + 2.0);
    static const double p2   = p0 * (2.0 * gamma * Ms * Ms - (gamma - 1.0))
                               / (gamma + 1.0);
    // Piston velocity: lab-frame velocity of post-shock gas
    static const double u2   = vs * (1.0 - rho_air / rho2);

    if (x < xs) {
        return make_mhd(rho2, u2, 0.0, 0.0, 0.0, 0.0, 0.0, p2, gamma);
    }
    const double r2 = (x - xc)*(x - xc) + (y - yc)*(y - yc);
    if (r2 < R * R) {
        return make_mhd(rho_He, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p0, gamma);
    }
    return make_mhd(rho_air, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p0, gamma);
}

// ============================================================
// Brio–Wu 1D MHD shock tube  (Brio & Wu 1988, γ = 2.0)
//
// Classic MHD shock-tube test with a compound wave structure.
// Domain: [0, 1] in x, thin in y (1D).  Discontinuity at x = 0.5.
//
//   Left  (x < 0.5): ρ=1,      p=1,   Bx=0.75, By= 1
//   Right (x ≥ 0.5): ρ=0.125,  p=0.1, Bx=0.75, By=-1
// ============================================================
Conserved brio_wu_state(double x, double /*y*/) {
    constexpr double gamma = 2.0;
    constexpr double Bx    = 0.75;

    if (x < 0.5) {
        return make_mhd(1.0,   0.0, 0.0, 0.0,  Bx,  1.0, 0.0, 1.0,  gamma);
    } else {
        return make_mhd(0.125, 0.0, 0.0, 0.0,  Bx, -1.0, 0.0, 0.1,  gamma);
    }
}

// ============================================================
// Orszag–Tang 2D MHD vortex  (Orszag & Tang 1979, γ = 5/3)
//
// Standard 2D MHD benchmark for transition to MHD turbulence.
// Domain: [0, 2π] × [0, 2π], fully periodic.
//
// Initial conditions (Dahlburg & Picone 1989 normalisation):
//   ρ = γ²,  p = γ,  cs = 1
//   u = −sin y,  v = sin x
//   Bx = −sin y,  By = sin 2x
// ============================================================
Conserved orszag_tang_state(double x, double y) {
    constexpr double gamma = 5.0 / 3.0;
    const double rho = gamma * gamma;         // = 25/9
    const double p   = gamma;                 // = 5/3, gives cs = 1
    const double u   = -std::sin(y);
    const double v   =  std::sin(x);
    const double Bx  = -std::sin(y);
    const double By  =  std::sin(2.0 * x);
    return make_mhd(rho, u, v, 0.0, Bx, By, 0.0, p, gamma);
}

} // namespace

// ============================================================
// Public API
// ============================================================

CaseId parse_case_id(const std::string& name) {
    if (name == "kelvin_helmholtz") return CaseId::KelvinHelmholtz;
    if (name == "shock_bubble")     return CaseId::ShockBubble;
    if (name == "brio_wu")          return CaseId::BrioWu;
    if (name == "orszag_tang")      return CaseId::OrszagTang;

    throw std::runtime_error(
        "Unknown MHD test case: '" + name + "'. "
        "Valid names: kelvin_helmholtz, shock_bubble, brio_wu, orszag_tang."
    );
}

std::string case_id_to_string(CaseId id) {
    switch (id) {
        case CaseId::KelvinHelmholtz: return "kelvin_helmholtz";
        case CaseId::ShockBubble:     return "shock_bubble";
        case CaseId::BrioWu:          return "brio_wu";
        case CaseId::OrszagTang:      return "orszag_tang";
    }
    throw std::runtime_error("Unhandled CaseId in case_id_to_string.");
}

CaseConfig get_case_config(CaseId id) {
    switch (id) {

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

        case CaseId::ShockBubble: {
            // Transmissive inflow/outflow; reflecting channel walls (top/bottom)
            CaseConfig cfg{
                500, 197, 2,
                0.0, 0.225, 0.0, 0.089,
                /*cfl=*/0.4, /*t_end=*/0.0,  // set below from snapshot schedule
                /*gamma=*/1.4,
                BoundaryConfig{
                    BoundaryType::Transmissive, BoundaryType::Transmissive,
                    BoundaryType::Reflecting,   BoundaryType::Reflecting
                }
            };

            // Physical snapshot times derived from Haas & Sturtevant (1987) Fig. 4.
            // Dimensionless time: t = T * Ms * s / R  =>  T = T0 + t_paper * tau
            //   T0  = time for shock to reach the bubble left edge = (xc-R-xs)/vs
            //   tau = time for shock to traverse one bubble radius  = R/vs
            constexpr double gam_sb  = 1.4;
            constexpr double p0_sb   = 1.01325e5;   // Pa
            constexpr double rho_sb  = 1.29;         // kg/m³
            constexpr double Ms_sb   = 1.22;
            constexpr double xc_sb   = 0.035;        // bubble centre x, m
            constexpr double R_sb    = 0.025;        // bubble radius, m
            constexpr double xs_sb   = 0.005;        // initial shock position, m
            const double vs_sb  = Ms_sb * std::sqrt(gam_sb * p0_sb / rho_sb);
            const double T0_sb  = (xc_sb - R_sb - xs_sb) / vs_sb;
            const double tau_sb = R_sb / vs_sb;

            // Dimensionless snapshot times from the paper (Fig. 4, Ms = 1.22)
            const double t_paper[] = {0.6, 1.2, 1.8, 3.0, 4.6, 6.2, 7.8, 12.6, 19.0};
            for (double tp : t_paper) {
                cfg.snapshot_times.push_back(T0_sb + tp * tau_sb);
                // Tag format: "t" + dimensionless_time*10 zero-padded to 3 digits
                // e.g. t=0.6 -> "t006", t=12.6 -> "t126", t=19.0 -> "t190"
                std::ostringstream oss;
                oss << "t" << std::setw(3) << std::setfill('0')
                    << static_cast<int>(std::round(tp * 10.0));
                cfg.snapshot_tags.push_back(oss.str());
            }
            cfg.t_end = cfg.snapshot_times.back();
            return cfg;
        }

        case CaseId::BrioWu:
            // 1D problem: periodic in y, transmissive in x
            return CaseConfig{
                800, 4, 2,
                0.0, 1.0, 0.0, 4.0 / 800.0,
                /*cfl=*/0.4, /*t_end=*/0.1,
                /*gamma=*/2.0,
                BoundaryConfig{
                    BoundaryType::Transmissive, BoundaryType::Transmissive,
                    BoundaryType::Periodic,     BoundaryType::Periodic
                }
            };

        case CaseId::OrszagTang:
            // Fully periodic, domain [0, 2π]²
            return CaseConfig{
                192, 192, 2,
                0.0, 2.0 * M_PI, 0.0, 2.0 * M_PI,
                /*cfl=*/0.4, /*t_end=*/M_PI,
                /*gamma=*/5.0 / 3.0,
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
        case CaseId::KelvinHelmholtz: return kh_state(x, y);
        case CaseId::ShockBubble:     return shock_bubble_state(x, y);
        case CaseId::BrioWu:          return brio_wu_state(x, y);
        case CaseId::OrszagTang:      return orszag_tang_state(x, y);
    }
    throw std::runtime_error("Unhandled CaseId in initial_state_at.");
}

Conserved initial_state_at(const std::string& name, double x, double y) {
    return initial_state_at(parse_case_id(name), x, y);
}
