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

} // namespace

// ============================================================
// Public API
// ============================================================

CaseId parse_case_id(const std::string& name) {
    if (name == "kelvin_helmholtz")  return CaseId::KelvinHelmholtz;

    throw std::runtime_error(
        "Unknown MHD test case: '" + name + "'. "
        "Valid names: kelvin_helmholtz."
    );
}

std::string case_id_to_string(CaseId id) {
    switch (id) {
        case CaseId::KelvinHelmholtz: return "kelvin_helmholtz";
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
    }
    throw std::runtime_error("Unhandled CaseId in initial_state_at.");
}

Conserved initial_state_at(const std::string& name, double x, double y) {
    return initial_state_at(parse_case_id(name), x, y);
}
