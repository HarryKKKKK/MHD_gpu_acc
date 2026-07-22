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
    // HLLC-L implementation following:
    // F. Zhang et al., "On energy consistency of intermediate states in
    // HLL-type MHD Riemann solvers" (2025): Eq. (18), Eq. (20), Eq. (28),
    // and Appendix B, Eqs. (B.1)-(B.5).
    //
    // The hyperbolic-divergence-cleaning subsystem is kept separate. Its
    // numerical flux is imposed by apply_glm_flux() using Eq. (16).

    const Primitive VL = phys::cons_to_prim(UL);
    const Primitive VR = phys::cons_to_prim(UR);

    if (!primitive_is_physical(VL) || !primitive_is_physical(VR)) {
        return hll_flux(UL, UR, dir, ch);
    }

    const double rhoL = VL.rho;
    const double rhoR = VR.rho;
    const double pL   = VL.p;
    const double pR   = VR.p;

    double unL, unR, utL, utR, uwL, uwR;
    double BnL, BnR, BtL, BtR, BwL, BwR;

    if (dir == Direction::X) {
        unL = VL.u;  unR = VR.u;
        utL = VL.v;  utR = VR.v;
        uwL = VL.w;  uwR = VR.w;
        BnL = VL.Bx; BnR = VR.Bx;
        BtL = VL.By; BtR = VR.By;
        BwL = VL.Bz; BwR = VR.Bz;
    } else {
        unL = VL.v;  unR = VR.v;
        utL = VL.u;  utR = VR.u;
        uwL = VL.w;  uwR = VR.w;
        BnL = VL.By; BnR = VR.By;
        BtL = VL.Bx; BtR = VR.Bx;
        BwL = VL.Bz; BwR = VR.Bz;
    }

    const double cfL = (dir == Direction::X) ? phys::fast_speed_x(VL)
                                             : phys::fast_speed_y(VL);
    const double cfR = (dir == Direction::X) ? phys::fast_speed_x(VR)
                                             : phys::fast_speed_y(VR);

    // Eq. (18): estimates of the fastest left- and right-going MHD waves.
    const double SL = fmin(unL - cfL, unR - cfR);
    const double SR = fmax(unL + cfL, unR + cfR);

    const double denomLR = SR - SL;
    if (fabs(denomLR) < 1.0e-14) {
        return hll_flux(UL, UR, dir, ch);
    }

    const Conserved FL = (dir == Direction::X) ? phys::flux_x(UL, ch)
                                               : phys::flux_y(UL, ch);
    const Conserved FR = (dir == Direction::X) ? phys::flux_x(UR, ch)
                                               : phys::flux_y(UR, ch);

    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    const double BnL2 = BnL * BnL;
    const double BnR2 = BnR * BnR;

    const double BmagL2 = BnL2 + BtL*BtL + BwL*BwL;
    const double BmagR2 = BnR2 + BtR*BtR + BwR*BwR;
    const double PL = pL + 0.5 * BmagL2;
    const double PR = pR + 0.5 * BmagR2;

    // Eq. (28): longitudinal magnetic field used inside the MHD Riemann fan.
    // The GLM/HDC flux itself is imposed separately by apply_glm_flux().
    const double BnM  = 0.5 * (BnL + BnR);
    const double BnM2 = BnM * BnM;

    const double AL = rhoL * (SL - unL);
    const double AR = rhoR * (SR - unR);
    const double denomSM = AR - AL;
    if (fabs(denomSM) < 1.0e-14) {
        return hll_flux(UL, UR, dir, ch);
    }

    // Eq. (B.1): speed of the middle entropy/contact wave.
    const double SM =
        (AR*unR - PR + BnR2 - (AL*unL - PL + BnL2)) / denomSM;

    if (!finite_number(SM) ||
        fabs(SL - SM) < 1.0e-14 ||
        fabs(SR - SM) < 1.0e-14) {
        return hll_flux(UL, UR, dir, ch);
    }

    // Eq. (B.2): one HLL-averaged tangential magnetic field throughout
    // the complete intermediate region.  The induction fluxes use the
    // original left/right longitudinal fields, as in Eq. (13).
    const double FBtL = unL*BtL - BnL*utL;
    const double FBtR = unR*BtR - BnR*utR;
    const double FBwL = unL*BwL - BnL*uwL;
    const double FBwR = unR*BwR - BnR*uwR;

    const double BtM =
        (SR*BtR - SL*BtL - (FBtR - FBtL)) / denomLR;
    const double BwM =
        (SR*BwR - SL*BwL - (FBwR - FBwL)) / denomLR;

    if (!finite_number(BtM) || !finite_number(BwM)) {
        return hll_flux(UL, UR, dir, ch);
    }

    // Eq. (B.4): total pressure in the intermediate region.
    const double PM =
        (AR*(PL - BnL2)
         - AL*(PR - BnR2)
         + AL*AR*(unR - unL)) / denomSM
        + BnM2;

    if (!finite_number(PM)) {
        return hll_flux(UL, UR, dir, ch);
    }

    // Eq. (B.5): left and right intermediate densities.
    const double rhoLs = rhoL * (SL - unL) / (SL - SM);
    const double rhoRs = rhoR * (SR - unR) / (SR - SM);

    if (!(rhoLs > 0.0) || !(rhoRs > 0.0) ||
        !finite_number(rhoLs) || !finite_number(rhoRs)) {
        return hll_flux(UL, UR, dir, ch);
    }

    // Eq. (B.5): tangential momenta.  These must not be replaced by the
    // original tangential velocities when Bt/Bw change in the star region.
    const double mtLs =
        (rhoL*utL*(SL - unL) - (BnM*BtM - BnL*BtL)) / (SL - SM);
    const double mwLs =
        (rhoL*uwL*(SL - unL) - (BnM*BwM - BnL*BwL)) / (SL - SM);
    const double mtRs =
        (rhoR*utR*(SR - unR) - (BnM*BtM - BnR*BtR)) / (SR - SM);
    const double mwRs =
        (rhoR*uwR*(SR - unR) - (BnM*BwM - BnR*BwR)) / (SR - SM);

    const double utLs = mtLs / rhoLs;
    const double uwLs = mwLs / rhoLs;
    const double utRs = mtRs / rhoRs;
    const double uwRs = mwRs / rhoRs;

    if (!finite_number(utLs) || !finite_number(uwLs) ||
        !finite_number(utRs) || !finite_number(uwRs)) {
        return hll_flux(UL, UR, dir, ch);
    }

    const double BdotVL  = BnL*unL + BtL*utL + BwL*uwL;
    const double BdotVR  = BnR*unR + BtR*utR + BwR*uwR;
    const double BdotVLs = BnM*SM  + BtM*utLs + BwM*uwLs;
    const double BdotVRs = BnM*SM  + BtM*utRs + BwM*uwRs;

    // Eq. (B.5): intermediate total energies.  In particular, the magnetic
    // work term is -(BnM * BdotV_star - Bn_side * BdotV_side).
    const double ELs =
        (UL.E*(SL - unL)
         + PM*SM
         - PL*unL
         - (BnM*BdotVLs - BnL*BdotVL)) / (SL - SM);

    const double ERs =
        (UR.E*(SR - unR)
         + PM*SM
         - PR*unR
         - (BnM*BdotVRs - BnR*BdotVR)) / (SR - SM);

    if (!finite_number(ELs) || !finite_number(ERs)) {
        return hll_flux(UL, UR, dir, ch);
    }

    // psi belongs to the separately solved HDC subsystem.  Its star value
    // is used only to build a complete Conserved object for the existing
    // code interface; apply_glm_flux() replaces the Bn and psi fluxes.
    const double psiM = glm_resolve(BnL, BnR, VL.psi, VR.psi, ch).psi;

    auto build_conserved = [&] (
        double rhos, double uns, double uts, double uws,
        double Bns,  double Bts, double Bws, double Es, double psis
    ) -> Conserved {
        if (dir == Direction::X) {
            return Conserved(rhos, rhos*uns, rhos*uts, rhos*uws,
                             Bns, Bts, Bws, Es, psis);
        }
        return Conserved(rhos, rhos*uts, rhos*uns, rhos*uws,
                         Bts, Bns, Bws, Es, psis);
    };

    if (SM >= 0.0) {
        // Materialise only the selected nine-component star state.  Keeping
        // ULs and URs live together raises the register high-water mark when
        // this function is inlined into the fused GPU advance kernel.
        const Conserved ULs = build_conserved(
            rhoLs, SM, utLs, uwLs, BnM, BtM, BwM, ELs, psiM);
        if (!primitive_is_physical(phys::cons_to_prim(ULs))) {
            return hll_flux(UL, UR, dir, ch);
        }
        // Eq. (20), left star branch.
        return FL + SL * (ULs - UL);
    }

    const Conserved URs = build_conserved(
        rhoRs, SM, utRs, uwRs, BnM, BtM, BwM, ERs, psiM);
    if (!primitive_is_physical(phys::cons_to_prim(URs))) {
        return hll_flux(UL, UR, dir, ch);
    }
    // Eq. (20), right star branch.
    return FR + SR * (URs - UR);
}

HD inline bool conserved_is_finite(const Conserved& U) {
    return finite_number(U.rho)  && finite_number(U.rhou) &&
           finite_number(U.rhov) && finite_number(U.rhow) &&
           finite_number(U.Bx)   && finite_number(U.By)   &&
           finite_number(U.Bz)   && finite_number(U.E)    &&
           finite_number(U.psi);
}

HD inline Conserved swap_xy(const Conserved& U) {
    return Conserved(U.rho, U.rhov, U.rhou, U.rhow,
                     U.By, U.Bx, U.Bz, U.E, U.psi);
}

// Peer-style HLL fallback: solve the GLM interface first, install the common
// normal field and psi in both states, then compute primitives, wave bounds
// and physical fluxes from those adjusted states.
HD inline Conserved hll_glm_flux_x(
    const Conserved& UL_in,
    const Conserved& UR_in,
    double           ch
) {
    Conserved UL = UL_in;
    Conserved UR = UR_in;
    const GlmStar glm = glm_resolve(UL.Bx, UR.Bx, UL.psi, UR.psi, ch);
    UL.Bx = glm.Bn; UR.Bx = glm.Bn;
    UL.psi = glm.psi; UR.psi = glm.psi;

    const Primitive WL = phys::cons_to_prim(UL);
    const Primitive WR = phys::cons_to_prim(UR);
    if (!primitive_is_physical(WL) || !primitive_is_physical(WR)) {
        return hll_flux(UL_in, UR_in, Direction::X, ch);
    }

    const double cmax = fmax(phys::fast_speed_x(WL),
                             phys::fast_speed_x(WR));
    const double SL = fmin(WL.u, WR.u) - cmax;
    const double SR = fmax(WL.u, WR.u) + cmax;
    const Conserved FL = phys::flux_x(UL, ch);
    const Conserved FR = phys::flux_x(UR, ch);

    Conserved F;
    if (SL >= 0.0) {
        F = FL;
    } else if (SR <= 0.0) {
        F = FR;
    } else {
        const double denom = SR - SL;
        if (!finite_number(denom) || fabs(denom) < 1.0e-14) {
            return hll_flux(UL_in, UR_in, Direction::X, ch);
        }
        F = (SR*FL - SL*FR + (SR*SL)*(UR - UL)) / denom;
    }
    F.Bx = glm.psi;
    F.psi = ch*ch*glm.Bn;
    return conserved_is_finite(F)
        ? F
        : hll_flux(UL_in, UR_in, Direction::X, ch);
}

// Canonical x-direction HLLD implementation ported expression-for-expression
// from huangyu701/mhd-cuda-solver.  In particular, the GLM interface state is
// installed before primitive conversion, wave-speed estimation and fluxes.
template <Direction Dir>
HD inline Conserved hlld_flux_dir(
    const Conserved& UL_in,
    const Conserved& UR_in,
    double           ch
) {
    auto fallback_hll = [&]() -> Conserved {
        if constexpr (Dir == Direction::X) {
            return hll_glm_flux_x(UL_in, UR_in, ch);
        } else {
            return swap_xy(hll_glm_flux_x(
                swap_xy(UL_in), swap_xy(UR_in), ch));
        }
    };

    Conserved UL = UL_in;
    Conserved UR = UR_in;
    const double BnL = (Dir == Direction::X) ? UL.Bx : UL.By;
    const double BnR = (Dir == Direction::X) ? UR.Bx : UR.By;
    const GlmStar glm = glm_resolve(BnL, BnR, UL.psi, UR.psi, ch);
    const double Bx = glm.Bn;
    const double psi_s = glm.psi;

    if constexpr (Dir == Direction::X) {
        UL.Bx = Bx; UR.Bx = Bx;
    } else {
        UL.By = Bx; UR.By = Bx;
    }
    UL.psi = psi_s; UR.psi = psi_s;

    const Primitive WL = phys::cons_to_prim(UL);
    const Primitive WR = phys::cons_to_prim(UR);
    if (!primitive_is_physical(WL) || !primitive_is_physical(WR)) {
        return fallback_hll();
    }

    const double uL  = (Dir == Direction::X) ? WL.u  : WL.v;
    const double uR  = (Dir == Direction::X) ? WR.u  : WR.v;
    const double vL  = (Dir == Direction::X) ? WL.v  : WL.u;
    const double vR  = (Dir == Direction::X) ? WR.v  : WR.u;
    const double BtL = (Dir == Direction::X) ? WL.By : WL.Bx;
    const double BtR = (Dir == Direction::X) ? WR.By : WR.Bx;

    const double cfL = (Dir == Direction::X) ? phys::fast_speed_x(WL)
                                             : phys::fast_speed_y(WL);
    const double cfR = (Dir == Direction::X) ? phys::fast_speed_x(WR)
                                             : phys::fast_speed_y(WR);
    const double cmax = fmax(cfL, cfR);
    const double SL = fmin(uL, uR) - cmax;
    const double SR = fmax(uL, uR) + cmax;

    const Conserved FL = (Dir == Direction::X) ? phys::flux_x(UL, ch)
                                               : phys::flux_y(UL, ch);
    const Conserved FR = (Dir == Direction::X) ? phys::flux_x(UR, ch)
                                               : phys::flux_y(UR, ch);

    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    const double denomRL = SR - SL;
    if (!finite_number(SL) || !finite_number(SR) ||
        fabs(denomRL) < 1.0e-14) {
        return fallback_hll();
    }

    const double pTL = WL.p + 0.5*(Bx*Bx + BtL*BtL + WL.Bz*WL.Bz);
    const double pTR = WR.p + 0.5*(Bx*Bx + BtR*BtR + WR.Bz*WR.Bz);

    const double denomM = (SR - uR)*WR.rho - (SL - uL)*WL.rho;
    if (!finite_number(denomM) || fabs(denomM) < 1.0e-14) {
        return fallback_hll();
    }

    const double SM = ((SR - uR)*WR.rho*uR
                       - (SL - uL)*WL.rho*uL - pTR + pTL) / denomM;
    const double pTs = ((SR - uR)*WR.rho*pTL
                        - (SL - uL)*WL.rho*pTR
                        + WL.rho*WR.rho*(SR - uR)*(SL - uL)
                          *(uR - uL)) / denomM;
    if (!finite_number(SM) || !finite_number(pTs) ||
        fabs(SL - SM) < 1.0e-14 || fabs(SR - SM) < 1.0e-14) {
        return fallback_hll();
    }

    auto star_state = [&](const Primitive& W, double E, double S,
                          Conserved& Us, double& rhos) -> bool {
        const double u  = (Dir == Direction::X) ? W.u  : W.v;
        const double v  = (Dir == Direction::X) ? W.v  : W.u;
        const double Bt = (Dir == Direction::X) ? W.By : W.Bx;
        const double rhoS = W.rho * (S - u) / (S - SM);
        rhos = rhoS;
        if (!(rhoS > 0.0) || !finite_number(rhoS)) return false;

        const double denom = W.rho*(S - u)*(S - SM) - Bx*Bx;
        double vs, ws, Bys, Bzs;
        if (fabs(denom) < 1.0e-30 *
                          (W.rho*(S - u)*(S - u) + 1.0)) {
            vs = v; ws = W.w;
            Bys = Bt; Bzs = W.Bz;
        } else {
            const double inv = 1.0 / denom;
            vs = v - Bx*Bt*(SM - u)*inv;
            ws = W.w - Bx*W.Bz*(SM - u)*inv;
            Bys = Bt * (W.rho*(S - u)*(S - u) - Bx*Bx)*inv;
            Bzs = W.Bz * (W.rho*(S - u)*(S - u) - Bx*Bx)*inv;
        }

        const double vdotB = u*Bx + v*Bt + W.w*W.Bz;
        const double vsdotB = SM*Bx + vs*Bys + ws*Bzs;
        const double Es = ((S - u)*E
                           - (W.p + 0.5*(Bx*Bx + Bt*Bt + W.Bz*W.Bz))*u
                           + pTs*SM + Bx*(vdotB - vsdotB)) / (S - SM);
        if constexpr (Dir == Direction::X) {
            Us = Conserved(rhoS, rhoS*SM, rhoS*vs, rhoS*ws,
                           Bx, Bys, Bzs, Es, psi_s);
        } else {
            Us = Conserved(rhoS, rhoS*vs, rhoS*SM, rhoS*ws,
                           Bys, Bx, Bzs, Es, psi_s);
        }
        return conserved_is_finite(Us) &&
               primitive_is_physical(phys::cons_to_prim(Us));
    };

    Conserved UsL, UsR;
    double rhosL, rhosR;
    if (!star_state(WL, UL.E, SL, UsL, rhosL) ||
        !star_state(WR, UR.E, SR, UsR, rhosR)) {
        return fallback_hll();
    }

    const double sqrtL = sqrt(rhosL);
    const double sqrtR = sqrt(rhosR);
    const double SsL = SM - fabs(Bx) / sqrtL;
    const double SsR = SM + fabs(Bx) / sqrtR;
    if (!finite_number(SsL) || !finite_number(SsR)) {
        return fallback_hll();
    }

    const bool degenerate =
        fabs(Bx) < 1.0e-12 * (1.0 + fabs(SR - SL));
    const Conserved FsL = FL + SL*(UsL - UL);
    const Conserved FsR = FR + SR*(UsR - UR);
    Conserved F;

    if (!degenerate) {
        if (SL <= 0.0 && 0.0 <= SsL) {
            F = FsL;
        } else if (SsR <= 0.0 && 0.0 <= SR) {
            F = FsR;
        } else {
            const double sgn = (Bx > 0.0) ? 1.0 : -1.0;
            const double denomV = sqrtL + sqrtR;
            if (!finite_number(denomV) || denomV <= 0.0) {
                return fallback_hll();
            }
            const double vss =
                (sqrtL*UsL.rhov/rhosL + sqrtR*UsR.rhov/rhosR
                 + (UsR.By - UsL.By)*sgn) / denomV;
            const double wss =
                (sqrtL*UsL.rhow/rhosL + sqrtR*UsR.rhow/rhosR
                 + (UsR.Bz - UsL.Bz)*sgn) / denomV;
            const double Byss =
                (sqrtL*UsR.By + sqrtR*UsL.By
                 + sqrtL*sqrtR*(UsR.rhov/rhosR - UsL.rhov/rhosL)*sgn)
                / denomV;
            const double Bzss =
                (sqrtL*UsR.Bz + sqrtR*UsL.Bz
                 + sqrtL*sqrtR*(UsR.rhow/rhosR - UsL.rhow/rhosL)*sgn)
                / denomV;
            const double vssdotB = SM*Bx + vss*Byss + wss*Bzss;

            auto inner_state = [&](const Conserved& Us, double rhos,
                                   double sign_side) -> Conserved {
                const double vsdotB =
                    SM*Bx + (Us.rhov/rhos)*Us.By + (Us.rhow/rhos)*Us.Bz;
                return Conserved(
                    rhos, rhos*SM, rhos*vss, rhos*wss,
                    Bx, Byss, Bzss,
                    Us.E + sign_side*sqrt(rhos)*(vsdotB - vssdotB)*sgn,
                    psi_s);
            };

            if (SsL <= 0.0 && 0.0 <= SM) {
                const Conserved UssL = inner_state(UsL, rhosL, -1.0);
                if (!conserved_is_finite(UssL) ||
                    !primitive_is_physical(phys::cons_to_prim(UssL))) {
                    return fallback_hll();
                }
                F = FsL + SsL*(UssL - UsL);
            } else if (SM <= 0.0 && 0.0 <= SsR) {
                const Conserved UssR = inner_state(UsR, rhosR, +1.0);
                if (!conserved_is_finite(UssR) ||
                    !primitive_is_physical(phys::cons_to_prim(UssR))) {
                    return fallback_hll();
                }
                F = FsR + SsR*(UssR - UsR);
            } else {
                return fallback_hll();
            }
        }
    } else {
        if (SL <= 0.0 && 0.0 <= SM) {
            F = FsL;
        } else if (SM <= 0.0 && 0.0 <= SR) {
            F = FsR;
        } else {
            return fallback_hll();
        }
    }

    F.Bx = psi_s;
    F.psi = ch*ch*Bx;
    return conserved_is_finite(F) ? F : fallback_hll();
}

HD inline Conserved hlld_flux(
    const Conserved& UL,
    const Conserved& UR,
    Direction        dir,
    double           ch
) {
    if (dir == Direction::X) return hlld_flux_x(UL, UR, ch);
    return swap_xy(hlld_flux_x(swap_xy(UL), swap_xy(UR), ch));
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

// Compile-time solver selection for GPU kernels (and other callers whose
// solver is known statically).  Keeping Solver in the type removes the
// launch-uniform runtime dispatch from every face solve and lets the compiler
// discard the three unused Riemann implementations from each kernel variant.
template <RiemannSolver Solver>
HD inline Conserved riemann_flux(
    const Conserved& UL,
    const Conserved& UR,
    Direction        dir,
    double           ch
) {
    if constexpr (Solver == RiemannSolver::HLL) {
        return hll_flux(UL, UR, dir, ch);
    } else {
        Conserved F;
        if constexpr (Solver == RiemannSolver::HLLC) {
            F = hllc_flux(UL, UR, dir, ch);
        } else if constexpr (Solver == RiemannSolver::HLLD) {
            F = hlld_flux(UL, UR, dir, ch);
        } else {
            static_assert(Solver == RiemannSolver::FORCE,
                          "unsupported compile-time Riemann solver");
            F = force_flux(UL, UR, dir, ch);
        }

        if (!finite_number(F.rho) || !finite_number(F.E) ||
            !finite_number(F.Bx)  || !finite_number(F.psi)) {
            return hll_flux(UL, UR, dir, ch);
        }

        apply_glm_flux(F, UL, UR, dir, ch);
        return F;
    }
}

template <RiemannSolver Solver>
HD inline Conserved riemann_flux(
    const Conserved& UL,
    const Conserved& UR,
    Direction        dir
) {
    return riemann_flux<Solver>(UL, UR, dir, phys::get_ch_glm());
}

HD inline Conserved riemann_flux(
    const Conserved& UL,
    const Conserved& UR,
    Direction        dir,
    RiemannSolver    solver
) {
    return riemann_flux(UL, UR, dir, solver, phys::get_ch_glm());
}
