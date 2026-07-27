#pragma once

#include <cmath>
#include "physics.hpp"

enum class Direction { X, Y };
enum class RiemannSolver { HLL, HLLC, FORCE };

HD inline bool finite_number(double x) {
#ifdef __CUDA_ARCH__
    return isfinite(x);
#else
    return std::isfinite(x);
#endif
}

HD inline bool primitive_is_physical(const Primitive& v) {
    return finite_number(v.rho) && finite_number(v.p) &&
           v.rho>0.0 && v.p>0.0;
}

HD inline double normal_velocity(const Primitive& v,Direction d) {
    return d==Direction::X?v.u:v.v;
}
HD inline Conserved physical_flux(const Conserved& q,Direction d) {
    return d==Direction::X?phys::flux_x(q):phys::flux_y(q);
}

HD inline Conserved hll_flux(const Conserved& ql,const Conserved& qr,
                             Direction d) {
    const Primitive vl=phys::cons_to_prim(ql),vr=phys::cons_to_prim(qr);
    const Conserved fl=physical_flux(ql,d),fr=physical_flux(qr,d);
    if(!primitive_is_physical(vl)||!primitive_is_physical(vr))
        return 0.5*(fl+fr);
    const double ul=normal_velocity(vl,d),ur=normal_velocity(vr,d);
    const double sl=fmin(ul-phys::sound_speed(vl),ur-phys::sound_speed(vr));
    const double sr=fmax(ul+phys::sound_speed(vl),ur+phys::sound_speed(vr));
    if(sl>=0.0)return fl;
    if(sr<=0.0)return fr;
    const double den=sr-sl;
    if(fabs(den)<1e-14)return 0.5*(fl+fr);
    return (sr*fl-sl*fr+(sl*sr)*(qr-ql))/den;
}

HD inline Conserved hllc_star(const Conserved& q,const Primitive& v,
                              Direction d,double s,double sm) {
    const double un=normal_velocity(v,d);
    const double factor=v.rho*(s-un)/(s-sm);
    const double pstar=v.p+v.rho*(s-un)*(sm-un);
    if(d==Direction::X) {
        return {factor,factor*sm,factor*v.v,factor*v.w,
                ((s-un)*q.E-v.p*un+pstar*sm)/(s-sm)};
    }
    return {factor,factor*v.u,factor*sm,factor*v.w,
            ((s-un)*q.E-v.p*un+pstar*sm)/(s-sm)};
}

HD inline Conserved hllc_flux(const Conserved& ql,const Conserved& qr,
                              Direction d) {
    const Primitive vl=phys::cons_to_prim(ql),vr=phys::cons_to_prim(qr);
    if(!primitive_is_physical(vl)||!primitive_is_physical(vr))
        return hll_flux(ql,qr,d);
    const double ul=normal_velocity(vl,d),ur=normal_velocity(vr,d);
    const double sl=fmin(ul-phys::sound_speed(vl),ur-phys::sound_speed(vr));
    const double sr=fmax(ul+phys::sound_speed(vl),ur+phys::sound_speed(vr));
    const Conserved fl=physical_flux(ql,d),fr=physical_flux(qr,d);
    if(sl>=0.0)return fl;
    if(sr<=0.0)return fr;
    const double den=vl.rho*(sl-ul)-vr.rho*(sr-ur);
    if(fabs(den)<1e-14)return hll_flux(ql,qr,d);
    const double sm=(vr.p-vl.p+vl.rho*ul*(sl-ul)-
                     vr.rho*ur*(sr-ur))/den;
    if(!finite_number(sm)||fabs(sl-sm)<1e-14||fabs(sr-sm)<1e-14)
        return hll_flux(ql,qr,d);
    const Conserved qs=sm>=0.0?hllc_star(ql,vl,d,sl,sm)
                                :hllc_star(qr,vr,d,sr,sm);
    const Conserved f=sm>=0.0?fl+sl*(qs-ql):fr+sr*(qs-qr);
    if(!finite_number(f.rho)||!finite_number(f.E))
        return hll_flux(ql,qr,d);
    return f;
}

HD inline Conserved force_flux(const Conserved& ql,const Conserved& qr,
                               Direction d) {
    const Primitive vl=phys::cons_to_prim(ql),vr=phys::cons_to_prim(qr);
    const Conserved fl=physical_flux(ql,d),fr=physical_flux(qr,d);
    if(!primitive_is_physical(vl)||!primitive_is_physical(vr))
        return 0.5*(fl+fr);
    const double alpha=fmax(fabs(normal_velocity(vl,d))+phys::sound_speed(vl),
                            fabs(normal_velocity(vr,d))+phys::sound_speed(vr));
    if(alpha<1e-14)return 0.5*(fl+fr);
    const Conserved flf=0.5*(fl+fr)-0.5*alpha*(qr-ql);
    const Conserved qri=0.5*(ql+qr)-0.5/alpha*(fr-fl);
    if(!primitive_is_physical(phys::cons_to_prim(qri)))return flf;
    return 0.5*(flf+physical_flux(qri,d));
}

HD inline Conserved riemann_flux(const Conserved& ql,const Conserved& qr,
                                 Direction d,RiemannSolver solver) {
    if(solver==RiemannSolver::HLL)return hll_flux(ql,qr,d);
    if(solver==RiemannSolver::HLLC)return hllc_flux(ql,qr,d);
    return force_flux(ql,qr,d);
}

template<RiemannSolver Solver>
HD inline Conserved riemann_flux(const Conserved& ql,const Conserved& qr,
                                 Direction d) {
    if constexpr(Solver==RiemannSolver::HLL)return hll_flux(ql,qr,d);
    if constexpr(Solver==RiemannSolver::HLLC)return hllc_flux(ql,qr,d);
    return force_flux(ql,qr,d);
}
