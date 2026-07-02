#pragma once

#include <cmath>

#include "types.hpp"

namespace phys {

#ifdef __CUDACC__
__device__ static double d_gamma = 5.0 / 3.0;
inline double gamma = 5.0 / 3.0;
#else
inline double gamma = 5.0 / 3.0;
#endif

#ifdef __CUDACC__
__device__ static double d_ch_glm = 0.0;
__device__ static double d_cr_glm = 0.18;
inline double ch_glm = 0.0;
inline double cr_glm = 0.18;
#else
inline double ch_glm = 0.0;
inline double cr_glm = 0.18;
#endif

#ifdef __CUDACC__
HD inline double get_gamma() {
#ifdef __CUDA_ARCH__
    return d_gamma;
#else
    return gamma;
#endif
}
HD inline double get_ch_glm() {
#ifdef __CUDA_ARCH__
    return d_ch_glm;
#else
    return ch_glm;
#endif
}
#else
HD inline double get_gamma()  { return gamma; }
HD inline double get_ch_glm() { return ch_glm; }
#endif

HD inline Primitive cons_to_prim(const Conserved& U) {
    const double inv_rho = 1.0 / U.rho;
    const double ux = U.rhou * inv_rho;
    const double uy = U.rhov * inv_rho;
    const double uz = U.rhow * inv_rho;
    const double Bmag2 = U.Bx*U.Bx + U.By*U.By + U.Bz*U.Bz;
    const double ke    = 0.5 * U.rho * (ux*ux + uy*uy + uz*uz);
    const double p     = (get_gamma() - 1.0) * (U.E - ke - 0.5*Bmag2);
    return Primitive(U.rho, ux, uy, uz, U.Bx, U.By, U.Bz, p, U.psi);
}

HD inline Conserved prim_to_cons(const Primitive& V) {
    const double Bmag2 = V.Bx*V.Bx + V.By*V.By + V.Bz*V.Bz;
    const double ke    = 0.5 * V.rho * (V.u*V.u + V.v*V.v + V.w*V.w);
    const double E     = V.p / (get_gamma() - 1.0) + ke + 0.5*Bmag2;
    return Conserved(V.rho, V.rho*V.u, V.rho*V.v, V.rho*V.w,
                     V.Bx, V.By, V.Bz, E, V.psi);
}

HD inline Conserved flux_x(const Conserved& U, double ch) {
    const double inv_rho = 1.0 / U.rho;
    const double ux = U.rhou * inv_rho;
    const double uy = U.rhov * inv_rho;
    const double uz = U.rhow * inv_rho;
    const double Bmag2 = U.Bx*U.Bx + U.By*U.By + U.Bz*U.Bz;
    const double ke    = 0.5 * U.rho * (ux*ux + uy*uy + uz*uz);
    const double p     = (get_gamma() - 1.0) * (U.E - ke - 0.5*Bmag2);
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

HD inline Conserved flux_y(const Conserved& U, double ch) {
    const double inv_rho = 1.0 / U.rho;
    const double ux = U.rhou * inv_rho;
    const double uy = U.rhov * inv_rho;
    const double uz = U.rhow * inv_rho;
    const double Bmag2 = U.Bx*U.Bx + U.By*U.By + U.Bz*U.Bz;
    const double ke    = 0.5 * U.rho * (ux*ux + uy*uy + uz*uz);
    const double p     = (get_gamma() - 1.0) * (U.E - ke - 0.5*Bmag2);
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

HD inline double fast_speed_x(const Primitive& V) {
    const double a2  = get_gamma() * V.p / V.rho;
    const double b2  = (V.Bx*V.Bx + V.By*V.By + V.Bz*V.Bz) / V.rho;
    const double bx2 = V.Bx * V.Bx / V.rho;
    const double disc = (a2 + b2)*(a2 + b2) - 4.0*a2*bx2;
    return sqrt(0.5 * (a2 + b2 + sqrt(disc > 0.0 ? disc : 0.0)));
}

HD inline double fast_speed_y(const Primitive& V) {
    const double a2  = get_gamma() * V.p / V.rho;
    const double b2  = (V.Bx*V.Bx + V.By*V.By + V.Bz*V.Bz) / V.rho;
    const double by2 = V.By * V.By / V.rho;
    const double disc = (a2 + b2)*(a2 + b2) - 4.0*a2*by2;
    return sqrt(0.5 * (a2 + b2 + sqrt(disc > 0.0 ? disc : 0.0)));
}

HD inline double sound_speed(const Primitive& V) {
    return sqrt(get_gamma() * V.p / V.rho);
}

HD inline double max_signal_speed_x(const Primitive& V, double ch) {
    const double cf = fast_speed_x(V);
    return fmax(fabs(V.u) + cf, ch);
}

HD inline double max_signal_speed_y(const Primitive& V, double ch) {
    const double cf = fast_speed_y(V);
    return fmax(fabs(V.v) + cf, ch);
}

HD inline double pressure(const Conserved& U) {
    return cons_to_prim(U).p;
}

} // namespace phys
