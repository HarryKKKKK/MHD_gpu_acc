#pragma once

#include <cmath>

#include "physics.hpp"
#include "types.hpp"

enum class Direction { X, Y };

enum class RiemannSolver { HLL, HLLC, HLLD, FORCE };

HD inline Conserved physical_flux(const Conserved& U, Direction dir) {
    const double ch = phys::get_ch_glm();
    return (dir == Direction::X) ? phys::flux_x(U, ch)
                                 : phys::flux_y(U, ch);
}

HD inline double normal_velocity(const Primitive& V, Direction dir) {
    return (dir == Direction::X) ? V.u : V.v;
}

HD inline double normal_B(const Primitive& V, Direction dir) {
    return (dir == Direction::X) ? V.Bx : V.By;
}

HD inline bool finite_number(double x) {
#ifdef __CUDA_ARCH__
    return isfinite(x);
#else
    return std::isfinite(x);
#endif
}

HD inline bool primitive_is_physical(const Primitive& V) {
    return finite_number(V.rho) && finite_number(V.p) &&
           V.rho > 0.0 && V.p > 0.0;
}

struct GlmStar {
    double Bn;
    double psi;
};

HD inline GlmStar glm_resolve(
    double BnL, double BnR, double psiL, double psiR, double ch
) {
    GlmStar s;
    if (ch > 0.0) {
        s.Bn  = 0.5*(BnL + BnR) - 0.5/ch*(psiR - psiL);
        s.psi = 0.5*(psiL + psiR) - 0.5*ch*(BnR - BnL);
    } else {
        s.Bn  = 0.5*(BnL + BnR);
        s.psi = 0.5*(psiL + psiR);
    }
    return s;
}

HD inline Conserved hll_flux(
    const Conserved& UL,
    const Conserved& UR,
    Direction        dir,
    double           ch
) {
    const Primitive VL = phys::cons_to_prim(UL);
    const Primitive VR = phys::cons_to_prim(UR);

    const Conserved FL = (dir == Direction::X) ? phys::flux_x(UL, ch)
                                               : phys::flux_y(UL, ch);
    const Conserved FR = (dir == Direction::X) ? phys::flux_x(UR, ch)
                                               : phys::flux_y(UR, ch);

    if (!primitive_is_physical(VL) || !primitive_is_physical(VR)) {
        return 0.5 * (FL + FR);
    }

    const double cfL = (dir == Direction::X) ? phys::fast_speed_x(VL)
                                             : phys::fast_speed_y(VL);
    const double cfR = (dir == Direction::X) ? phys::fast_speed_x(VR)
                                             : phys::fast_speed_y(VR);

    const double unL = normal_velocity(VL, dir);
    const double unR = normal_velocity(VR, dir);

    const double SL = fmin(fmin(unL - cfL, unR - cfR), -ch);
    const double SR = fmax(fmax(unL + cfL, unR + cfR),  ch);

    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    const double denom = SR - SL;
    if (fabs(denom) < 1.0e-14) return 0.5 * (FL + FR);

    return (SR * FL - SL * FR + (SL * SR) * (UR - UL)) / denom;
}

HD inline Conserved hllc_flux(
    const Conserved& UL,
    const Conserved& UR,
    Direction        dir,
    double           ch
) {
    const Primitive VL = phys::cons_to_prim(UL);
    const Primitive VR = phys::cons_to_prim(UR);

    if (!primitive_is_physical(VL) || !primitive_is_physical(VR)) {
        return hll_flux(UL, UR, dir, ch);
    }

    const double rhoL = VL.rho, rhoR = VR.rho;
    const double pL   = VL.p,   pR   = VR.p;

    double unL, unR, utL, utR, uwL, uwR;
    double BnL_raw, BnR_raw, BtL, BtR, BwL, BwR;
    const double psiL = VL.psi, psiR = VR.psi;

    if (dir == Direction::X) {
        unL = VL.u;  unR = VR.u;
        utL = VL.v;  utR = VR.v;
        uwL = VL.w;  uwR = VR.w;
        BnL_raw = VL.Bx; BnR_raw = VR.Bx;
        BtL = VL.By; BtR = VR.By;
        BwL = VL.Bz; BwR = VR.Bz;
    } else {
        unL = VL.v;  unR = VR.v;
        utL = VL.u;  utR = VR.u;
        uwL = VL.w;  uwR = VR.w;
        BnL_raw = VL.By; BnR_raw = VR.By;
        BtL = VL.Bx; BtR = VR.Bx;
        BwL = VL.Bz; BwR = VR.Bz;
    }

    const double cfL = (dir == Direction::X) ? phys::fast_speed_x(VL)
                                             : phys::fast_speed_y(VL);
    const double cfR = (dir == Direction::X) ? phys::fast_speed_x(VR)
                                             : phys::fast_speed_y(VR);

    const double SL = fmin(unL - cfL, unR - cfR);
    const double SR = fmax(unL + cfL, unR + cfR);

    const Conserved FL = (dir == Direction::X) ? phys::flux_x(UL, ch)
                                               : phys::flux_y(UL, ch);
    const Conserved FR = (dir == Direction::X) ? phys::flux_x(UR, ch)
                                               : phys::flux_y(UR, ch);

    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    const GlmStar glm  = glm_resolve(BnL_raw, BnR_raw, psiL, psiR, ch);
    const double  Bn   = glm.Bn;
    const double  psiM = glm.psi;

    const double BmagL2 = Bn*Bn + BtL*BtL + BwL*BwL;
    const double BmagR2 = Bn*Bn + BtR*BtR + BwR*BwR;
    const double ptL = pL + 0.5*BmagL2;
    const double ptR = pR + 0.5*BmagR2;

    const double numerSM = (SR - unR)*rhoR*unR - (SL - unL)*rhoL*unL + ptL - ptR;
    const double denomSM = (SR - unR)*rhoR     - (SL - unL)*rhoL;
    if (fabs(denomSM) < 1.0e-14) return hll_flux(UL, UR, dir, ch);
    const double SM = numerSM / denomSM;

    const double rhoLs = rhoL * (SL - unL) / (SL - SM);
    const double rhoRs = rhoR * (SR - unR) / (SR - SM);
    if (rhoLs <= 0.0 || rhoRs <= 0.0 ||
        !finite_number(rhoLs) || !finite_number(rhoRs)) {
        return hll_flux(UL, UR, dir, ch);
    }

    const double ptLs = ptL + rhoL*(SL - unL)*(SM - unL);
    const double ptRs = ptR + rhoR*(SR - unR)*(SM - unR);

    const double FtL = unL*BtL - Bn*utL, FtR = unR*BtR - Bn*utR;
    const double FwL = unL*BwL - Bn*uwL, FwR = unR*BwR - Bn*uwR;
    const double Bt_star = (SR*BtR - SL*BtL - (FtR - FtL)) / (SR - SL);
    const double Bw_star = (SR*BwR - SL*BwL - (FwR - FwL)) / (SR - SL);

    const double BdotUL  = Bn*unL + BtL*utL + BwL*uwL;
    const double BdotULs = Bn*SM  + Bt_star*utL + Bw_star*uwL;
    const double BdotUR  = Bn*unR + BtR*utR + BwR*uwR;
    const double BdotURs = Bn*SM  + Bt_star*utR + Bw_star*uwR;

    const double ELs = ((SL - unL)*UL.E - ptL*unL + ptLs*SM
                        + Bn*(BdotULs - BdotUL)) / (SL - SM);
    const double ERs = ((SR - unR)*UR.E - ptR*unR + ptRs*SM
                        + Bn*(BdotURs - BdotUR)) / (SR - SM);

    auto build_conserved = [&](
        double rhos, double uns, double uts, double uws,
        double Bns,  double Bts, double Bws, double Es, double psis
    ) -> Conserved {
        if (dir == Direction::X) {
            return Conserved(rhos, rhos*uns, rhos*uts, rhos*uws,
                             Bns, Bts, Bws, Es, psis);
        } else {
            return Conserved(rhos, rhos*uts, rhos*uns, rhos*uws,
                             Bts, Bns, Bws, Es, psis);
        }
    };

    if (SM >= 0.0) {
        const Conserved ULs = build_conserved(
            rhoLs, SM, utL, uwL, Bn, Bt_star, Bw_star, ELs, psiM);
        if (!primitive_is_physical(phys::cons_to_prim(ULs))) {
            return hll_flux(UL, UR, dir, ch);
        }
        return FL + SL * (ULs - UL);
    } else {
        const Conserved URs = build_conserved(
            rhoRs, SM, utR, uwR, Bn, Bt_star, Bw_star, ERs, psiM);
        if (!primitive_is_physical(phys::cons_to_prim(URs))) {
            return hll_flux(UL, UR, dir, ch);
        }
        return FR + SR * (URs - UR);
    }
}

HD inline Conserved hlld_flux(
    const Conserved& UL,
    const Conserved& UR,
    Direction        dir,
    double           ch
) {
    const Primitive VL = phys::cons_to_prim(UL);
    const Primitive VR = phys::cons_to_prim(UR);

    if (!primitive_is_physical(VL) || !primitive_is_physical(VR)) {
        return hll_flux(UL, UR, dir, ch);
    }

    const double rhoL = VL.rho, rhoR = VR.rho;
    const double pL   = VL.p,   pR   = VR.p;

    double unL, unR, utL, utR, uwL, uwR;
    double BnL_raw, BnR_raw, BtL, BtR, BwL, BwR;
    const double psiL = VL.psi, psiR = VR.psi;

    if (dir == Direction::X) {
        unL = VL.u;  unR = VR.u;
        utL = VL.v;  utR = VR.v;
        uwL = VL.w;  uwR = VR.w;
        BnL_raw = VL.Bx; BnR_raw = VR.Bx;
        BtL = VL.By; BtR = VR.By;
        BwL = VL.Bz; BwR = VR.Bz;
    } else {
        unL = VL.v;  unR = VR.v;
        utL = VL.u;  utR = VR.u;
        uwL = VL.w;  uwR = VR.w;
        BnL_raw = VL.By; BnR_raw = VR.By;
        BtL = VL.Bx; BtR = VR.Bx;
        BwL = VL.Bz; BwR = VR.Bz;
    }

    const double cfL = (dir == Direction::X) ? phys::fast_speed_x(VL)
                                             : phys::fast_speed_y(VL);
    const double cfR = (dir == Direction::X) ? phys::fast_speed_x(VR)
                                             : phys::fast_speed_y(VR);

    const double SL = fmin(unL - cfL, unR - cfR);
    const double SR = fmax(unL + cfL, unR + cfR);

    const double denom_RL = SR - SL;
    if (fabs(denom_RL) < 1.0e-14) {
        return hllc_flux(UL, UR, dir, ch);
    }

    const Conserved FL = (dir == Direction::X) ? phys::flux_x(UL, ch)
                                               : phys::flux_y(UL, ch);
    const Conserved FR = (dir == Direction::X) ? phys::flux_x(UR, ch)
                                               : phys::flux_y(UR, ch);

    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    const GlmStar glm  = glm_resolve(BnL_raw, BnR_raw, psiL, psiR, ch);
    const double  Bn   = glm.Bn;
    const double  psiM = glm.psi;

    const double BmagL2 = Bn*Bn + BtL*BtL + BwL*BwL;
    const double BmagR2 = Bn*Bn + BtR*BtR + BwR*BwR;
    const double ptL = pL + 0.5*BmagL2;
    const double ptR = pR + 0.5*BmagR2;

    const double numerSM = (SR - unR)*rhoR*unR - (SL - unL)*rhoL*unL
                         + ptL - ptR;
    const double denomSM = (SR - unR)*rhoR     - (SL - unL)*rhoL;

    if (fabs(denomSM) < 1.0e-14) {
        return hllc_flux(UL, UR, dir, ch);
    }
    const double SM = numerSM / denomSM;

    const double rhoLs = rhoL * (SL - unL) / (SL - SM);
    const double rhoRs = rhoR * (SR - unR) / (SR - SM);

    if (rhoLs <= 0.0 || rhoRs <= 0.0 ||
        !finite_number(rhoLs) || !finite_number(rhoRs)) {
        return hllc_flux(UL, UR, dir, ch);
    }

    const double ptstar = ptL + rhoL*(SL - unL)*(SM - unL);

    const double sqrtRhoLs = sqrt(rhoLs);
    const double sqrtRhoRs = sqrt(rhoRs);
    const double SLs = SM - fabs(Bn) / sqrtRhoLs;
    const double SRs = SM + fabs(Bn) / sqrtRhoRs;

    double utLs, uwLs, BtLs, BwLs, ELs;
    {
        const double BdotUL = Bn*unL + BtL*utL + BwL*uwL;
        const double d      = rhoL*(SL - unL)*(SL - SM) - Bn*Bn;
        if (fabs(d) < 1.0e-14) {
            utLs = utL; uwLs = uwL; BtLs = BtL; BwLs = BwL;
        } else {
            utLs = utL - Bn*BtL*(SM - unL) / d;
            uwLs = uwL - Bn*BwL*(SM - unL) / d;
            BtLs = BtL * (rhoL*(SL-unL)*(SL-unL) - Bn*Bn) / d;
            BwLs = BwL * (rhoL*(SL-unL)*(SL-unL) - Bn*Bn) / d;
        }
        const double BdotULs = Bn*SM + BtLs*utLs + BwLs*uwLs;
        ELs = (UL.E*(SL - unL) - ptL*unL + ptstar*SM
               + Bn*(BdotULs - BdotUL)) / (SL - SM);
    }

    double utRs, uwRs, BtRs, BwRs, ERs;
    {
        const double BdotUR = Bn*unR + BtR*utR + BwR*uwR;
        const double d      = rhoR*(SR - unR)*(SR - SM) - Bn*Bn;
        if (fabs(d) < 1.0e-14) {
            utRs = utR; uwRs = uwR; BtRs = BtR; BwRs = BwR;
        } else {
            utRs = utR - Bn*BtR*(SM - unR) / d;
            uwRs = uwR - Bn*BwR*(SM - unR) / d;
            BtRs = BtR * (rhoR*(SR-unR)*(SR-unR) - Bn*Bn) / d;
            BwRs = BwR * (rhoR*(SR-unR)*(SR-unR) - Bn*Bn) / d;
        }
        const double BdotURs = Bn*SM + BtRs*utRs + BwRs*uwRs;
        ERs = (UR.E*(SR - unR) - ptR*unR + ptstar*SM
               + Bn*(BdotURs - BdotUR)) / (SR - SM);
    }

    const double signBn = (Bn >= 0.0) ? 1.0 : -1.0;

    const double sqRhoLs = sqrtRhoLs;
    const double sqRhoRs = sqrtRhoRs;
    const double denom2  = sqRhoLs + sqRhoRs;

    double utLss, uwLss, BtLss, BwLss;
    double utRss, uwRss, BtRss, BwRss;
    double ELss, ERss;

    if (fabs(denom2) < 1.0e-14) {
        utLss = utLs; uwLss = uwLs; BtLss = BtLs; BwLss = BwLs; ELss = ELs;
        utRss = utRs; uwRss = uwRs; BtRss = BtRs; BwRss = BwRs; ERss = ERs;
    } else {
        const double inv_d2 = 1.0 / denom2;
        utLss = utRss = (sqRhoLs*utLs + sqRhoRs*utRs + signBn*(BtRs - BtLs)) * inv_d2;
        uwLss = uwRss = (sqRhoLs*uwLs + sqRhoRs*uwRs + signBn*(BwRs - BwLs)) * inv_d2;
        BtLss = BtRss = (sqRhoLs*BtRs + sqRhoRs*BtLs + signBn*sqRhoLs*sqRhoRs*(utRs - utLs)) * inv_d2;
        BwLss = BwRss = (sqRhoLs*BwRs + sqRhoRs*BwLs + signBn*sqRhoLs*sqRhoRs*(uwRs - uwLs)) * inv_d2;

        const double BdotULss = Bn*SM + BtLss*utLss + BwLss*uwLss;
        const double BdotULs_ = Bn*SM + BtLs *utLs  + BwLs *uwLs;
        ELss = ELs - sqRhoLs*signBn*(BdotULss - BdotULs_);

        const double BdotURss = Bn*SM + BtRss*utRss + BwRss*uwRss;
        const double BdotURs_ = Bn*SM + BtRs *utRs  + BwRs *uwRs;
        ERss = ERs + sqRhoRs*signBn*(BdotURss - BdotURs_);
    }

    auto build_conserved = [&](
        double rhos_, double uns_, double uts_, double uws_,
        double Bns_, double Bts_, double Bws_, double Es_,
        double psis_
    ) -> Conserved {
        if (dir == Direction::X) {
            return Conserved(
                rhos_,
                rhos_*uns_, rhos_*uts_, rhos_*uws_,
                Bns_, Bts_, Bws_,
                Es_, psis_
            );
        } else {
            return Conserved(
                rhos_,
                rhos_*uts_, rhos_*uns_, rhos_*uws_,
                Bts_, Bns_, Bws_,
                Es_, psis_
            );
        }
    };

    auto state_ok = [&](const Conserved& U) -> bool {
        return primitive_is_physical(phys::cons_to_prim(U));
    };

    if (SM >= 0.0) {
        if (SLs >= 0.0) {
            const Conserved ULs = build_conserved(
                rhoLs, SM, utLs, uwLs, Bn, BtLs, BwLs, ELs, psiM);
            if (!state_ok(ULs)) return hllc_flux(UL, UR, dir, ch);
            return FL + SL * (ULs - UL);
        } else {
            const Conserved ULs  = build_conserved(
                rhoLs, SM, utLs,  uwLs,  Bn, BtLs,  BwLs,  ELs,  psiM);
            const Conserved ULss = build_conserved(
                rhoLs, SM, utLss, uwLss, Bn, BtLss, BwLss, ELss, psiM);
            if (!state_ok(ULss)) return hllc_flux(UL, UR, dir, ch);
            return FL + SL*(ULs - UL) + SLs*(ULss - ULs);
        }
    } else {
        if (SRs <= 0.0) {
            const Conserved URs = build_conserved(
                rhoRs, SM, utRs, uwRs, Bn, BtRs, BwRs, ERs, psiM);
            if (!state_ok(URs)) return hllc_flux(UL, UR, dir, ch);
            return FR + SR * (URs - UR);
        } else {
            const Conserved URs  = build_conserved(
                rhoRs, SM, utRs,  uwRs,  Bn, BtRs,  BwRs,  ERs,  psiM);
            const Conserved URss = build_conserved(
                rhoRs, SM, utRss, uwRss, Bn, BtRss, BwRss, ERss, psiM);
            if (!state_ok(URss)) return hllc_flux(UL, UR, dir, ch);
            return FR + SR*(URs - UR) + SRs*(URss - URs);
        }
    }
}

HD inline Conserved force_flux(
    const Conserved& UL,
    const Conserved& UR,
    Direction        dir,
    double           ch
) {
    const Primitive VL = phys::cons_to_prim(UL);
    const Primitive VR = phys::cons_to_prim(UR);

    const Conserved FL = (dir == Direction::X) ? phys::flux_x(UL, ch)
                                               : phys::flux_y(UL, ch);
    const Conserved FR = (dir == Direction::X) ? phys::flux_x(UR, ch)
                                               : phys::flux_y(UR, ch);

    if (!primitive_is_physical(VL) || !primitive_is_physical(VR)) {
        return 0.5 * (FL + FR);
    }

    const double cfL = (dir == Direction::X) ? phys::fast_speed_x(VL)
                                             : phys::fast_speed_y(VL);
    const double cfR = (dir == Direction::X) ? phys::fast_speed_x(VR)
                                             : phys::fast_speed_y(VR);

    const double unL = normal_velocity(VL, dir);
    const double unR = normal_velocity(VR, dir);

    const double alpha = fmax(fmax(fabs(unL) + cfL, fabs(unR) + cfR), ch);

    if (alpha < 1.0e-14) return 0.5 * (FL + FR);

    const Conserved F_lf = 0.5*(FL + FR) - 0.5*alpha*(UR - UL);
    const Conserved U_ri = 0.5*(UL + UR) - 0.5*(1.0/alpha)*(FR - FL);

    const Primitive V_ri = phys::cons_to_prim(U_ri);
    if (!primitive_is_physical(V_ri)) return F_lf;

    const Conserved F_ri = (dir == Direction::X) ? phys::flux_x(U_ri, ch)
                                                  : phys::flux_y(U_ri, ch);
    return 0.5*(F_lf + F_ri);
}

HD inline void apply_glm_flux(
    Conserved&       F,
    const Conserved& UL,
    const Conserved& UR,
    Direction        dir,
    double           ch
) {
    if (ch <= 0.0) return;
    const double BnL  = (dir == Direction::X) ? UL.Bx : UL.By;
    const double BnR  = (dir == Direction::X) ? UR.Bx : UR.By;

    const GlmStar glm = glm_resolve(BnL, BnR, UL.psi, UR.psi, ch);

    if (dir == Direction::X) F.Bx = glm.psi;
    else                     F.By = glm.psi;
    F.psi = ch*ch * glm.Bn;
}

HD inline Conserved riemann_flux(
    const Conserved& UL,
    const Conserved& UR,
    Direction        dir,
    RiemannSolver    solver,
    double           ch
) {
    if (solver == RiemannSolver::HLL) {
        return hll_flux(UL, UR, dir, ch);
    }

    Conserved F = (solver == RiemannSolver::HLLC)  ? hllc_flux(UL, UR, dir, ch)
                : (solver == RiemannSolver::HLLD)  ? hlld_flux(UL, UR, dir, ch)
                                                   : force_flux(UL, UR, dir, ch);

    if (!finite_number(F.rho) || !finite_number(F.E) ||
        !finite_number(F.Bx)  || !finite_number(F.psi)) {
        return hll_flux(UL, UR, dir, ch);
    }

    apply_glm_flux(F, UL, UR, dir, ch);
    return F;
}

HD inline Conserved riemann_flux(
    const Conserved& UL,
    const Conserved& UR,
    Direction        dir,
    RiemannSolver    solver
) {
    return riemann_flux(UL, UR, dir, solver, phys::get_ch_glm());
}
