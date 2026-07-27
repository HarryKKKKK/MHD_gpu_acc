#pragma once

#include <cmath>
#include "types.hpp"

namespace phys {

#ifdef __CUDACC__
__device__ static double d_gamma = 1.4;
inline double gamma = 1.4;
#else
inline double gamma = 1.4;
#endif

HD inline double get_gamma() {
#if defined(__CUDA_ARCH__)
    return d_gamma;
#else
    return gamma;
#endif
}

HD inline Primitive cons_to_prim(const Conserved& q) {
    const double inv_rho=1.0/q.rho;
    const double u=q.rhou*inv_rho, v=q.rhov*inv_rho, w=q.rhow*inv_rho;
    const double kinetic=0.5*q.rho*(u*u+v*v+w*w);
    return {q.rho,u,v,w,(get_gamma()-1.0)*(q.E-kinetic)};
}

HD inline Conserved prim_to_cons(const Primitive& v) {
    const double kinetic=0.5*v.rho*(v.u*v.u+v.v*v.v+v.w*v.w);
    return {v.rho,v.rho*v.u,v.rho*v.v,v.rho*v.w,
            v.p/(get_gamma()-1.0)+kinetic};
}

HD inline Conserved flux_x(const Conserved& q) {
    const Primitive v=cons_to_prim(q);
    return {q.rhou,q.rhou*v.u+v.p,q.rhou*v.v,q.rhou*v.w,
            (q.E+v.p)*v.u};
}
HD inline Conserved flux_y(const Conserved& q) {
    const Primitive v=cons_to_prim(q);
    return {q.rhov,q.rhov*v.u,q.rhov*v.v+v.p,q.rhov*v.w,
            (q.E+v.p)*v.v};
}
HD inline Conserved flux_z(const Conserved& q) {
    const Primitive v=cons_to_prim(q);
    return {q.rhow,q.rhow*v.u,q.rhow*v.v,q.rhow*v.w+v.p,
            (q.E+v.p)*v.w};
}

HD inline double sound_speed(const Primitive& v) {
    return sqrt(get_gamma()*v.p/v.rho);
}
HD inline double max_signal_speed_x(const Primitive& v) {
    return fabs(v.u)+sound_speed(v);
}
HD inline double max_signal_speed_y(const Primitive& v) {
    return fabs(v.v)+sound_speed(v);
}
HD inline double max_signal_speed_z(const Primitive& v) {
    return fabs(v.w)+sound_speed(v);
}
HD inline double pressure(const Conserved& q) { return cons_to_prim(q).p; }

} // namespace phys
