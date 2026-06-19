#pragma once

#include <cmath>

#include "types.hpp"

// ============================================================
// MHD physics for the GLM-MHD system (Dedner et al. 2002).
//
// Conservative variables: U = (rho, rho*ux, rho*uy, rho*uz,
//                               Bx, By, Bz, E, psi)
//
// Equation of state (perfect gas, eq. 3):
//   p = (gamma - 1) * (E - rho|u|^2/2 - |B|^2/2)
//
// GLM divergence-cleaning scalar psi propagates div(B) errors
// out of the domain at speed ch (hyperbolic GLM, Section 2).
// Mixed GLM adds exponential damping to psi (eq. 19, 45).
// ============================================================

namespace phys {

// adiabatic exponent — set per test case before initialisation
// GPU: set via set_gpu_physics_gamma() before launching kernels
#ifdef __CUDACC__
__device__ static double gamma = 5.0 / 3.0;
#else
inline double gamma = 5.0 / 3.0;
#endif

// GLM cleaning speed ch (set each timestep from max signal speed, Section 4)
// Mixed-GLM damping ratio c_r = c_p^2 / c_h (optimal ~0.18, Fig. 2)
#ifndef __CUDACC__
inline double ch_glm = 0.0;
inline double cr_glm = 0.18;
#endif

// ------------------------------------------------------------
// Conserved → Primitive conversion
// ------------------------------------------------------------
HD inline Primitive cons_to_prim(const Conserved& U) {
    const double inv_rho = 1.0 / U.rho;
    const double ux = U.rhou * inv_rho;
    const double uy = U.rhov * inv_rho;
    const double uz = U.rhow * inv_rho;
    const double Bmag2 = U.Bx*U.Bx + U.By*U.By + U.Bz*U.Bz;
    const double ke    = 0.5 * U.rho * (ux*ux + uy*uy + uz*uz);
    const double p     = (gamma - 1.0) * (U.E - ke - 0.5*Bmag2);
    return Primitive(U.rho, ux, uy, uz, U.Bx, U.By, U.Bz, p, U.psi);
}

// ------------------------------------------------------------
// Primitive → Conserved conversion
// ------------------------------------------------------------
HD inline Conserved prim_to_cons(const Primitive& V) {
    const double Bmag2 = V.Bx*V.Bx + V.By*V.By + V.Bz*V.Bz;
    const double ke    = 0.5 * V.rho * (V.u*V.u + V.v*V.v + V.w*V.w);
    const double E     = V.p / (gamma - 1.0) + ke + 0.5*Bmag2;
    return Conserved(V.rho, V.rho*V.u, V.rho*V.v, V.rho*V.w,
                     V.Bx, V.By, V.Bz, E, V.psi);
}

// ------------------------------------------------------------
// GLM-MHD flux in x-direction (eqs. 1a-1d with GLM modification,
// eq. 24a-24e from Dedner et al. 2002).
//
// F_x = (rho*ux,
//         rho*ux^2 + p_tot - Bx^2,
//         rho*ux*uy - Bx*By,
//         rho*ux*uz - Bx*Bz,
//         psi,                      <- GLM: replaces 0
//         By*ux - Bx*uy,
//         Bz*ux - Bx*uz,
//         (E + p_tot)*ux - Bx*(u.B),
//         ch^2 * Bx)                <- GLM
// ------------------------------------------------------------
HD inline Conserved flux_x(const Conserved& U, double ch) {
    const double inv_rho = 1.0 / U.rho;
    const double ux = U.rhou * inv_rho;
    const double uy = U.rhov * inv_rho;
    const double uz = U.rhow * inv_rho;
    const double Bmag2 = U.Bx*U.Bx + U.By*U.By + U.Bz*U.Bz;
    const double ke    = 0.5 * U.rho * (ux*ux + uy*uy + uz*uz);
    const double p     = (gamma - 1.0) * (U.E - ke - 0.5*Bmag2);
    const double p_tot = p + 0.5*Bmag2;
    const double BdotU = U.Bx*ux + U.By*uy + U.Bz*uz;

    return Conserved(
        U.rhou,
        U.rhou*ux + p_tot - U.Bx*U.Bx,
        U.rhou*uy         - U.Bx*U.By,
        U.rhou*uz         - U.Bx*U.Bz,
        U.psi,
        U.By*ux - U.Bx*uy,
        U.Bz*ux - U.Bx*uz,
        (U.E + p_tot)*ux  - U.Bx*BdotU,
        ch*ch * U.Bx
    );
}

// ------------------------------------------------------------
// GLM-MHD flux in y-direction
//
// F_y = (rho*uy,
//         rho*ux*uy - Bx*By,
//         rho*uy^2 + p_tot - By^2,
//         rho*uy*uz - By*Bz,
//         Bx*uy - By*ux,
//         psi,                      <- GLM
//         Bz*uy - By*uz,
//         (E + p_tot)*uy - By*(u.B),
//         ch^2 * By)                <- GLM
// ------------------------------------------------------------
HD inline Conserved flux_y(const Conserved& U, double ch) {
    const double inv_rho = 1.0 / U.rho;
    const double ux = U.rhou * inv_rho;
    const double uy = U.rhov * inv_rho;
    const double uz = U.rhow * inv_rho;
    const double Bmag2 = U.Bx*U.Bx + U.By*U.By + U.Bz*U.Bz;
    const double ke    = 0.5 * U.rho * (ux*ux + uy*uy + uz*uz);
    const double p     = (gamma - 1.0) * (U.E - ke - 0.5*Bmag2);
    const double p_tot = p + 0.5*Bmag2;
    const double BdotU = U.Bx*ux + U.By*uy + U.Bz*uz;

    return Conserved(
        U.rhov,
        U.rhov*ux         - U.By*U.Bx,
        U.rhov*uy + p_tot - U.By*U.By,
        U.rhov*uz         - U.By*U.Bz,
        U.Bx*uy - U.By*ux,
        U.psi,
        U.Bz*uy - U.By*uz,
        (U.E + p_tot)*uy  - U.By*BdotU,
        ch*ch * U.By
    );
}

// ------------------------------------------------------------
// Fast magnetosonic speed in the x-direction (eqs. 29-32).
//
//   a^2  = gamma*p/rho  (sound speed squared)
//   b^2  = |B|^2/rho    (total Alfvén speed squared)
//   b_x^2 = Bx^2/rho   (x-Alfvén speed squared)
//
//   c_f = sqrt( 1/2 * (a^2 + b^2 + sqrt((a^2+b^2)^2 - 4*a^2*b_x^2)) )
// ------------------------------------------------------------
HD inline double fast_speed_x(const Primitive& V) {
    const double a2  = gamma * V.p / V.rho;
    const double b2  = (V.Bx*V.Bx + V.By*V.By + V.Bz*V.Bz) / V.rho;
    const double bx2 = V.Bx * V.Bx / V.rho;
    const double disc = (a2 + b2)*(a2 + b2) - 4.0*a2*bx2;
    return sqrt(0.5 * (a2 + b2 + sqrt(disc > 0.0 ? disc : 0.0)));
}

// Fast magnetosonic speed in the y-direction (swap Bx ↔ By)
HD inline double fast_speed_y(const Primitive& V) {
    const double a2  = gamma * V.p / V.rho;
    const double b2  = (V.Bx*V.Bx + V.By*V.By + V.Bz*V.Bz) / V.rho;
    const double by2 = V.By * V.By / V.rho;
    const double disc = (a2 + b2)*(a2 + b2) - 4.0*a2*by2;
    return sqrt(0.5 * (a2 + b2 + sqrt(disc > 0.0 ? disc : 0.0)));
}

// Sound speed (for reference / diagnostics)
HD inline double sound_speed(const Primitive& V) {
    return sqrt(gamma * V.p / V.rho);
}

// ------------------------------------------------------------
// Maximum signal speed in each direction.
// Includes GLM waves traveling at ±ch (eigenvalues λ1, λ9 of
// the GLM-MHD system, eq. 33).
// ------------------------------------------------------------
HD inline double max_signal_speed_x(const Primitive& V, double ch) {
    const double cf = fast_speed_x(V);
    return fmax(fabs(V.u) + cf, ch);
}

HD inline double max_signal_speed_y(const Primitive& V, double ch) {
    const double cf = fast_speed_y(V);
    return fmax(fabs(V.v) + cf, ch);
}

// Thermal pressure from conserved state
HD inline double pressure(const Conserved& U) {
    return cons_to_prim(U).p;
}

} // namespace phys
