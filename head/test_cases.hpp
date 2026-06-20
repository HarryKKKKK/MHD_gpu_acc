#pragma once

#include <string>
#include <vector>

#include "types.hpp"

// ============================================================
// Boundary condition types
// ============================================================
enum class BoundaryType {
    Periodic,      // periodic wrap-around
    Transmissive,  // zero-gradient (Neumann) outflow
    Reflecting,    // symmetry (reflecting) wall
    Dirichlet      // fixed value from initial data (applied externally)
};

struct BoundaryConfig {
    BoundaryType left   = BoundaryType::Transmissive;
    BoundaryType right  = BoundaryType::Transmissive;
    BoundaryType bottom = BoundaryType::Transmissive;
    BoundaryType top    = BoundaryType::Transmissive;
};

// ============================================================
// Test case identifiers
// ============================================================
enum class CaseId {
    KelvinHelmholtz,  // §5: Kelvin-Helmholtz instability, γ=1.4
    ShockBubble,      // Mach 1.22 shock–bubble interaction, γ=1.4
    BrioWu,           // Brio–Wu 1D MHD shock tube, γ=2.0
    OrszagTang        // Orszag–Tang 2D MHD vortex, γ=5/3
};

// ============================================================
// Case configuration (domain, grid, time, physics)
// ============================================================
struct CaseConfig {
    int    nx;
    int    ny;
    int    ng;       // number of ghost layers (≥2 for MUSCL-Hancock)
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    double cfl;
    double t_end;
    double gamma;    // adiabatic exponent for this case
    BoundaryConfig bc;
    // Ordered physical times at which to write field snapshots.
    // Tags are file-name labels (e.g. "t006" for dimensionless t=0.6).
    // If empty, only the final state is written.
    std::vector<double>      snapshot_times = {};
    std::vector<std::string> snapshot_tags  = {};
};

// ============================================================
// API
// ============================================================
CaseId     parse_case_id(const std::string& case_name);
std::string case_id_to_string(CaseId case_id);

CaseConfig  get_case_config(CaseId case_id);
CaseConfig  get_case_config(const std::string& case_name);

// Scale grid resolution by factor n (nx *= n, ny *= n)
CaseConfig  get_n_case_config(CaseId case_id, int n);
CaseConfig  get_n_case_config(const std::string& case_name, int n);

// Return initial conserved state at cell center (x, y)
Conserved   initial_state_at(CaseId case_id, double x, double y);
Conserved   initial_state_at(const std::string& case_name, double x, double y);
