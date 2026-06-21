#pragma once

#include <cmath>

#include "physics.hpp"
#include "types.hpp"

// ============================================================
// Riemann solvers for the 2D GLM-MHD system (Dedner et al. 2002).
//
// Supported solvers:
//   HLL   — HLL with MHD fast magnetosonic + GLM wave speeds
//   HLLC  — HLLC (Toro et al. 1994); 3-wave, optimal for pure-hydro cases
//   HLLD  — HLLD (Miyoshi & Kusano 2005); more accurate for MHD
//   FORCE — Local FORCE (average of Lax-Friedrichs and Richtmyer)
//
// All solvers take the global GLM cleaning speed phys::get_ch_glm() which
// must be set (on CPU) before the flux computation step.
// ============================================================

enum class Direction { X, Y };

enum class RiemannSolver { HLL, HLLC, HLLD, FORCE };

// ------------------------------------------------------------
// Physical flux wrapper — uses phys::get_ch_glm() (host: inline double;
// device: __device__ static double set each step via set_gpu_physics_ch)
// ------------------------------------------------------------
HD inline Conserved physical_flux(const Conserved& U, Direction dir) {
    const double ch = phys::get_ch_glm();
    return (dir == Direction::X) ? phys::flux_x(U, ch)
                                 : phys::flux_y(U, ch);
}

// Normal velocity component
HD inline double normal_velocity(const Primitive& V, Direction dir) {
    return (dir == Direction::X) ? V.u : V.v;
}

// Normal magnetic field component
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

// Physical validity check for MHD primitive state
HD inline bool primitive_is_physical(const Primitive& V) {
    return finite_number(V.rho) && finite_number(V.p) &&
           V.rho > 0.0 && V.p > 0.0;
}

// ============================================================
// HLL Riemann solver for the GLM-MHD system.
//
// Wave speed estimates include the GLM cleaning waves at ±ch
// (eigenvalues λ1 = -ch and λ9 = +ch from eq. 33).
//
// S_L = min(u_nL - cf_L, u_nR - cf_R, -ch)
// S_R = max(u_nL + cf_L, u_nR + cf_R,  ch)
//
// F_HLL = (S_R*F_L - S_L*F_R + S_L*S_R*(U_R - U_L)) / (S_R - S_L)
// ============================================================
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

    // Include GLM wave speeds ±ch in the estimate (eq. 33)
    const double SL = fmin(fmin(unL - cfL, unR - cfR), -ch);
    const double SR = fmax(fmax(unL + cfL, unR + cfR),  ch);

    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    const double denom = SR - SL;
    if (fabs(denom) < 1.0e-14) return 0.5 * (FL + FR);

    return (SR * FL - SL * FR + (SL * SR) * (UR - UL)) / denom;
}

// ============================================================
// HLLC Riemann solver (Toro et al. 1994).
//
// 3-wave solver with a contact wave at SM.  Reduces to the
// standard hydro HLLC when B = 0 (e.g. shock_bubble test).
// For the general MHD case the tangential B fields are held
// fixed across the contact — a valid first approximation when
// |B| is small, and exact for pure hydro.
//
// References:
//   Toro, Spruce & Speares (1994), Shock Waves 4, 25-34.
//   Li (2005), J. Comput. Phys. 203, 344-357 (energy formula).
// ============================================================
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

    // Outer wave speeds — physical fast-magnetosonic bounds only.
    // Do NOT include ±ch here: ch_glm is a global maximum and using it as a
    // floor on per-interface bounds adds O(ch/cf - 1) excess dissipation on
    // every cell whose local speed is below the domain maximum (e.g. the
    // helium bubble interior in shock_bubble). The ±ch eigenvalues belong
    // to the psi/GLM sub-system, not to the physical MHD waves, so they
    // must not corrupt the rho/momentum/energy fluxes.
    const double SL = fmin(unL - cfL, unR - cfR);
    const double SR = fmax(unL + cfL, unR + cfR);

    const Conserved FL = (dir == Direction::X) ? phys::flux_x(UL, ch)
                                               : phys::flux_y(UL, ch);
    const Conserved FR = (dir == Direction::X) ? phys::flux_x(UR, ch)
                                               : phys::flux_y(UR, ch);

    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    const double BmagL2 = BnL*BnL + BtL*BtL + BwL*BwL;
    const double BmagR2 = BnR*BnR + BtR*BtR + BwR*BwR;
    const double ptL = pL + 0.5*BmagL2;
    const double ptR = pR + 0.5*BmagR2;

    // Contact wave speed (Miyoshi & Kusano eq. 38)
    const double numerSM = (SR - unR)*rhoR*unR - (SL - unL)*rhoL*unL + ptL - ptR;
    const double denomSM = (SR - unR)*rhoR     - (SL - unL)*rhoL;
    if (fabs(denomSM) < 1.0e-14) return hll_flux(UL, UR, dir, ch);
    const double SM = numerSM / denomSM;

    // Star densities (mass conservation across each outer wave)
    const double rhoLs = rhoL * (SL - unL) / (SL - SM);
    const double rhoRs = rhoR * (SR - unR) / (SR - SM);
    if (rhoLs <= 0.0 || rhoRs <= 0.0 ||
        !finite_number(rhoLs) || !finite_number(rhoRs)) {
        return hll_flux(UL, UR, dir, ch);
    }

    // Total pressure in each star region (momentum conservation)
    const double ptLs = ptL + rhoL*(SL - unL)*(SM - unL);
    const double ptRs = ptR + rhoR*(SR - unR)*(SM - unR);

    // GLM psi: linear interpolation (same treatment as HLLD)
    const double psiM = 0.5*(VL.psi + VR.psi) - 0.5*ch*(BnR - BnL);

    // Tangential B unchanged across contact (exact for B=0, HLLC approximation for MHD)
    // Energy in star states via Li (2005) eq. 17 (generalises Toro to MHD)
    const double BdotUL  = BnL*unL + BtL*utL + BwL*uwL;
    const double BdotULs = BnL*SM  + BtL*utL + BwL*uwL;  // un -> SM, tangential unchanged
    const double BdotUR  = BnR*unR + BtR*utR + BwR*uwR;
    const double BdotURs = BnR*SM  + BtR*utR + BwR*uwR;

    const double ELs = ((SL - unL)*UL.E - ptL*unL + ptLs*SM
                        + BnL*(BdotULs - BdotUL)) / (SL - SM);
    const double ERs = ((SR - unR)*UR.E - ptR*unR + ptRs*SM
                        + BnR*(BdotURs - BdotUR)) / (SR - SM);

    // Build conserved star state with direction rotation (same lambda pattern as HLLD)
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
            rhoLs, SM, utL, uwL, BnL, BtL, BwL, ELs, psiM);
        return FL + SL * (ULs - UL);
    } else {
        const Conserved URs = build_conserved(
            rhoRs, SM, utR, uwR, BnR, BtR, BwR, ERs, psiM);
        return FR + SR * (URs - UR);
    }
}

// ============================================================
// HLLD Riemann solver for ideal MHD (Miyoshi & Kusano 2005).
//
// This is a more accurate 5-wave solver for MHD. It resolves
// fast/Alfvén/slow waves and gives better results across
// discontinuities compared to HLL.
//
// Implementation follows the algorithm in Miyoshi & Kusano (2005),
// J. Comput. Phys. 208, 315-344.
// ============================================================
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

    // Rotate so the "normal" direction is always x
    // For Y direction: swap (x↔y) and (Bx↔By) in inputs/outputs
    const double rhoL = VL.rho, rhoR = VR.rho;
    const double pL   = VL.p,   pR   = VR.p;

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

    // Outer fast wave speeds — physical bounds only (see hllc_flux comment).
    const double SL = fmin(unL - cfL, unR - cfR);
    const double SR = fmax(unL + cfL, unR + cfR);

    // Fall back to HLL if outer waves degenerate
    const double denom_RL = SR - SL;
    if (fabs(denom_RL) < 1.0e-14) {
        return hll_flux(UL, UR, dir, ch);
    }

    // Total pressure
    const double BmagL2 = BnL*BnL + BtL*BtL + BwL*BwL;
    const double BmagR2 = BnR*BnR + BtR*BtR + BwR*BwR;
    const double ptL = pL + 0.5*BmagL2;
    const double ptR = pR + 0.5*BmagR2;

    // Middle (contact) wave speed SM (eq. 38 in Miyoshi & Kusano)
    const double numerSM = (SR - unR)*rhoR*unR - (SL - unL)*rhoL*unL
                         + ptL - ptR;
    const double denomSM = (SR - unR)*rhoR     - (SL - unL)*rhoL;

    if (fabs(denomSM) < 1.0e-14) {
        return hll_flux(UL, UR, dir, ch);
    }
    const double SM = numerSM / denomSM;

    // Density in L* and R* states (eq. 43)
    const double rhoLs = rhoL * (SL - unL) / (SL - SM);
    const double rhoRs = rhoR * (SR - unR) / (SR - SM);

    if (rhoLs <= 0.0 || rhoRs <= 0.0 ||
        !std::isfinite(rhoLs) || !std::isfinite(rhoRs)) {
        return hll_flux(UL, UR, dir, ch);
    }

    // Total pressure in star region (eq. 41)
    const double ptstar = ptL + rhoL*(SL - unL)*(SM - unL);

    // Alfvén speeds in L* and R* states (eq. 51)
    const double sqrtRhoLs = sqrt(rhoLs);
    const double sqrtRhoRs = sqrt(rhoRs);
    const double SLs = SM - fabs(BnL) / sqrtRhoLs;
    const double SRs = SM + fabs(BnR) / sqrtRhoRs;

    // L* state
    double utLs, uwLs, BtLs, BwLs, ELs;
    {
        const double BdotUL = BnL*unL + BtL*utL + BwL*uwL;
        const double d      = rhoL*(SL - unL)*(SL - SM) - BnL*BnL;
        if (fabs(d) < 1.0e-14) {
            utLs = utL; uwLs = uwL; BtLs = BtL; BwLs = BwL;
        } else {
            utLs = utL - BnL*BtL*(SM - unL) / d;
            uwLs = uwL - BnL*BwL*(SM - unL) / d;
            BtLs = BtL * (rhoL*(SL-unL)*(SL-unL) - BnL*BnL) / d;
            BwLs = BwL * (rhoL*(SL-unL)*(SL-unL) - BnL*BnL) / d;
        }
        const double BdotULs = BnL*SM + BtLs*utLs + BwLs*uwLs;
        ELs = (UL.E*(SL - unL) - ptL*unL + ptstar*SM
               + BnL*(BdotULs - BdotUL)) / (SL - SM);
    }

    // R* state
    double utRs, uwRs, BtRs, BwRs, ERs;
    {
        const double BdotUR = BnR*unR + BtR*utR + BwR*uwR;
        const double d      = rhoR*(SR - unR)*(SR - SM) - BnR*BnR;
        if (fabs(d) < 1.0e-14) {
            utRs = utR; uwRs = uwR; BtRs = BtR; BwRs = BwR;
        } else {
            utRs = utR - BnR*BtR*(SM - unR) / d;
            uwRs = uwR - BnR*BwR*(SM - unR) / d;
            BtRs = BtR * (rhoR*(SR-unR)*(SR-unR) - BnR*BnR) / d;
            BwRs = BwR * (rhoR*(SR-unR)*(SR-unR) - BnR*BnR) / d;
        }
        const double BdotURs = BnR*SM + BtRs*utRs + BwRs*uwRs;
        ERs = (UR.E*(SR - unR) - ptR*unR + ptstar*SM
               + BnR*(BdotURs - BdotUR)) / (SR - SM);
    }

    // Sign of Bn determines rotation in double-star states
    const double signBn = (BnL >= 0.0) ? 1.0 : -1.0;

    // Double-star states (eqs. 59-62)
    const double sqRhoLs = sqrtRhoLs;
    const double sqRhoRs = sqrtRhoRs;
    const double denom2  = sqRhoLs + sqRhoRs;

    double utLss, uwLss, BtLss, BwLss;
    double utRss, uwRss, BtRss, BwRss;
    double ELss, ERss;

    if (fabs(denom2) < 1.0e-14) {
        // Fall back if densities are equal and fields cancel
        utLss = utLs; uwLss = uwLs; BtLss = BtLs; BwLss = BwLs; ELss = ELs;
        utRss = utRs; uwRss = uwRs; BtRss = BtRs; BwRss = BwRs; ERss = ERs;
    } else {
        const double inv_d2 = 1.0 / denom2;
        utLss = utRss = (sqRhoLs*utLs + sqRhoRs*utRs + signBn*(BtRs - BtLs)) * inv_d2;
        uwLss = uwRss = (sqRhoLs*uwLs + sqRhoRs*uwRs + signBn*(BwRs - BwLs)) * inv_d2;
        BtLss = BtRss = (sqRhoLs*BtRs + sqRhoRs*BtLs + signBn*sqRhoLs*sqRhoRs*(utRs - utLs)) * inv_d2;
        BwLss = BwRss = (sqRhoLs*BwRs + sqRhoRs*BwLs + signBn*sqRhoLs*sqRhoRs*(uwRs - uwLs)) * inv_d2;

        ELss = ELs - sqRhoLs * signBn *
               (SM*BtLss + uwLss*BwLss - SM*BtLs - uwLs*BwLs  // approximate
                + utLss*BtLss - utLs*BtLs);
        ERss = ERs + sqRhoRs * signBn *
               (SM*BtRss + uwRss*BwRss - SM*BtRs - uwRs*BwRs
                + utRss*BtRss - utRs*BtRs);

        // More careful: eqs. 63-64
        const double BdotULss = BnL*SM + BtLss*utLss + BwLss*uwLss;
        const double BdotULs  = BnL*SM + BtLs *utLs  + BwLs *uwLs;
        ELss = ELs - sqRhoLs*signBn*(BdotULss - BdotULs);

        const double BdotURss = BnR*SM + BtRss*utRss + BwRss*uwRss;
        const double BdotURs  = BnR*SM + BtRs *utRs  + BwRs *uwRs;
        ERss = ERs + sqRhoRs*signBn*(BdotURss - BdotURs);
    }

    // Select which star region we are in and build the flux
    // via F* = F_side + S * (U* - U_side)
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

    // psi is handled by the separate GLM system; pass through unchanged
    const double psiL = VL.psi, psiR = VR.psi;
    const double psiM = 0.5*(psiL + psiR) - 0.5*ch*(BnR - BnL); // linear GLM

    const Conserved FL = (dir == Direction::X) ? phys::flux_x(UL, ch)
                                               : phys::flux_y(UL, ch);
    const Conserved FR = (dir == Direction::X) ? phys::flux_x(UR, ch)
                                               : phys::flux_y(UR, ch);

    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    if (SM >= 0.0) {
        if (SLs >= 0.0) {
            // In L* region
            const Conserved ULs = build_conserved(
                rhoLs, SM, utLs, uwLs, BnL, BtLs, BwLs, ELs, psiM);
            return FL + SL * (ULs - UL);
        } else {
            // In L** region
            const Conserved ULs  = build_conserved(
                rhoLs, SM, utLs,  uwLs,  BnL, BtLs,  BwLs,  ELs,  psiM);
            const Conserved ULss = build_conserved(
                rhoLs, SM, utLss, uwLss, BnL, BtLss, BwLss, ELss, psiM);
            return FL + SL*(ULs - UL) + SLs*(ULss - ULs);
        }
    } else {
        if (SRs <= 0.0) {
            // In R* region
            const Conserved URs = build_conserved(
                rhoRs, SM, utRs, uwRs, BnR, BtRs, BwRs, ERs, psiM);
            return FR + SR * (URs - UR);
        } else {
            // In R** region
            const Conserved URs  = build_conserved(
                rhoRs, SM, utRs,  uwRs,  BnR, BtRs,  BwRs,  ERs,  psiM);
            const Conserved URss = build_conserved(
                rhoRs, SM, utRss, uwRss, BnR, BtRss, BwRss, ERss, psiM);
            return FR + SR*(URs - UR) + SRs*(URss - URs);
        }
    }
}

// ============================================================
// FORCE Riemann solver — average of Lax-Friedrichs and Richtmyer.
// Uses local max signal speed estimate (no dt/dx needed).
// ============================================================
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

// ============================================================
// Unified interface — explicit ch form
// ============================================================
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

    const Conserved F = (solver == RiemannSolver::HLLC)  ? hllc_flux(UL, UR, dir, ch)
                      : (solver == RiemannSolver::HLLD)  ? hlld_flux(UL, UR, dir, ch)
                                                         : force_flux(UL, UR, dir, ch);

    if (!finite_number(F.rho) || !finite_number(F.E) ||
        !finite_number(F.Bx)  || !finite_number(F.psi)) {
        return hll_flux(UL, UR, dir, ch);
    }
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
