#pragma once

#include <cmath>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ============================================================
// 9-component MHD conserved state:
//   U = (rho, rho*ux, rho*uy, rho*uz, Bx, By, Bz, E, psi)
// psi is the GLM divergence-cleaning scalar (Dedner et al. 2002)
// ============================================================
struct Conserved {
    double rho;   // mass density
    double rhou;  // x-momentum
    double rhov;  // y-momentum
    double rhow;  // z-momentum (kept for 2.5D MHD)
    double Bx;    // x-component of magnetic field
    double By;    // y-component of magnetic field
    double Bz;    // z-component of magnetic field
    double E;     // total energy density
    double psi;   // GLM cleaning scalar

    HD Conserved()
        : rho(0), rhou(0), rhov(0), rhow(0),
          Bx(0), By(0), Bz(0), E(0), psi(0) {}

    HD Conserved(double rho_, double rhou_, double rhov_, double rhow_,
                 double Bx_,  double By_,  double Bz_,
                 double E_,   double psi_)
        : rho(rho_), rhou(rhou_), rhov(rhov_), rhow(rhow_),
          Bx(Bx_), By(By_), Bz(Bz_), E(E_), psi(psi_) {}

    // Pure-HD convenience constructor (B=0, rhow=0, psi=0)
    HD Conserved(double rho_, double rhou_, double rhov_, double E_)
        : rho(rho_), rhou(rhou_), rhov(rhov_), rhow(0),
          Bx(0), By(0), Bz(0), E(E_), psi(0) {}
};

// ============================================================
// 9-component MHD primitive state:
//   V = (rho, ux, uy, uz, Bx, By, Bz, p, psi)
// ============================================================
struct Primitive {
    double rho;  // mass density
    double u;    // x-velocity
    double v;    // y-velocity
    double w;    // z-velocity
    double Bx;   // x-component of magnetic field
    double By;   // y-component of magnetic field
    double Bz;   // z-component of magnetic field
    double p;    // thermal (gas) pressure
    double psi;  // GLM cleaning scalar

    HD Primitive()
        : rho(0), u(0), v(0), w(0),
          Bx(0), By(0), Bz(0), p(0), psi(0) {}

    HD Primitive(double rho_, double u_, double v_, double w_,
                 double Bx_,  double By_,  double Bz_,
                 double p_,   double psi_)
        : rho(rho_), u(u_), v(v_), w(w_),
          Bx(Bx_), By(By_), Bz(Bz_), p(p_), psi(psi_) {}

    // Pure-HD convenience constructor (B=0, w=0, psi=0)
    HD Primitive(double rho_, double u_, double v_, double p_)
        : rho(rho_), u(u_), v(v_), w(0),
          Bx(0), By(0), Bz(0), p(p_), psi(0) {}
};

// ------------------------------------------------------------
// Arithmetic operators for Conserved
// ------------------------------------------------------------

HD inline Conserved operator+(const Conserved& a, const Conserved& b) {
    return Conserved(
        a.rho  + b.rho,  a.rhou + b.rhou, a.rhov + b.rhov, a.rhow + b.rhow,
        a.Bx   + b.Bx,   a.By   + b.By,   a.Bz   + b.Bz,
        a.E    + b.E,     a.psi  + b.psi
    );
}

HD inline Conserved operator-(const Conserved& a, const Conserved& b) {
    return Conserved(
        a.rho  - b.rho,  a.rhou - b.rhou, a.rhov - b.rhov, a.rhow - b.rhow,
        a.Bx   - b.Bx,   a.By   - b.By,   a.Bz   - b.Bz,
        a.E    - b.E,     a.psi  - b.psi
    );
}

HD inline Conserved operator*(double s, const Conserved& a) {
    return Conserved(
        s*a.rho,  s*a.rhou, s*a.rhov, s*a.rhow,
        s*a.Bx,   s*a.By,   s*a.Bz,
        s*a.E,    s*a.psi
    );
}

HD inline Conserved operator*(const Conserved& a, double s) { return s * a; }

HD inline Conserved operator/(const Conserved& a, double s) {
    return (1.0 / s) * a;
}

HD inline Conserved& operator+=(Conserved& a, const Conserved& b) {
    a.rho  += b.rho;  a.rhou += b.rhou; a.rhov += b.rhov; a.rhow += b.rhow;
    a.Bx   += b.Bx;   a.By   += b.By;   a.Bz   += b.Bz;
    a.E    += b.E;     a.psi  += b.psi;
    return a;
}

HD inline Conserved& operator-=(Conserved& a, const Conserved& b) {
    a.rho  -= b.rho;  a.rhou -= b.rhou; a.rhov -= b.rhov; a.rhow -= b.rhow;
    a.Bx   -= b.Bx;   a.By   -= b.By;   a.Bz   -= b.Bz;
    a.E    -= b.E;     a.psi  -= b.psi;
    return a;
}

HD inline Conserved& operator*=(Conserved& a, double s) {
    a.rho  *= s; a.rhou *= s; a.rhov *= s; a.rhow *= s;
    a.Bx   *= s; a.By   *= s; a.Bz   *= s;
    a.E    *= s; a.psi  *= s;
    return a;
}

// ------------------------------------------------------------
// Arithmetic operators for Primitive
// ------------------------------------------------------------

HD inline Primitive operator+(const Primitive& a, const Primitive& b) {
    return Primitive(
        a.rho + b.rho, a.u + b.u, a.v + b.v, a.w + b.w,
        a.Bx  + b.Bx,  a.By + b.By, a.Bz + b.Bz,
        a.p   + b.p,   a.psi + b.psi
    );
}

HD inline Primitive operator-(const Primitive& a, const Primitive& b) {
    return Primitive(
        a.rho - b.rho, a.u - b.u, a.v - b.v, a.w - b.w,
        a.Bx  - b.Bx,  a.By - b.By, a.Bz - b.Bz,
        a.p   - b.p,   a.psi - b.psi
    );
}

HD inline Primitive operator*(double s, const Primitive& a) {
    return Primitive(
        s*a.rho, s*a.u, s*a.v, s*a.w,
        s*a.Bx,  s*a.By, s*a.Bz,
        s*a.p,   s*a.psi
    );
}

HD inline Primitive operator*(const Primitive& a, double s) { return s * a; }

HD inline Primitive operator/(const Primitive& a, double s) {
    return (1.0 / s) * a;
}

HD inline Primitive& operator+=(Primitive& a, const Primitive& b) {
    a.rho += b.rho; a.u += b.u; a.v += b.v; a.w += b.w;
    a.Bx  += b.Bx;  a.By += b.By; a.Bz += b.Bz;
    a.p   += b.p;   a.psi += b.psi;
    return a;
}

HD inline Primitive& operator-=(Primitive& a, const Primitive& b) {
    a.rho -= b.rho; a.u -= b.u; a.v -= b.v; a.w -= b.w;
    a.Bx  -= b.Bx;  a.By -= b.By; a.Bz -= b.Bz;
    a.p   -= b.p;   a.psi -= b.psi;
    return a;
}

HD inline Primitive& operator*=(Primitive& a, double s) {
    a.rho *= s; a.u *= s; a.v *= s; a.w *= s;
    a.Bx  *= s; a.By *= s; a.Bz *= s;
    a.p   *= s; a.psi *= s;
    return a;
}
