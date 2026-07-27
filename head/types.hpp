#pragma once

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// Five-component compressible Euler state:
// U = (rho, rho*u, rho*v, rho*w, E).
struct Conserved {
    double rho;
    double rhou;
    double rhov;
    double rhow;
    double E;

    HD Conserved() : rho(0), rhou(0), rhov(0), rhow(0), E(0) {}
    HD Conserved(double rho_, double rhou_, double rhov_, double rhow_,
                 double E_)
        : rho(rho_), rhou(rhou_), rhov(rhov_), rhow(rhow_), E(E_) {}
    HD Conserved(double rho_, double rhou_, double rhov_, double E_)
        : rho(rho_), rhou(rhou_), rhov(rhov_), rhow(0), E(E_) {}
};

// Primitive Euler state: V = (rho, u, v, w, p).
struct Primitive {
    double rho;
    double u;
    double v;
    double w;
    double p;

    HD Primitive() : rho(0), u(0), v(0), w(0), p(0) {}
    HD Primitive(double rho_, double u_, double v_, double w_, double p_)
        : rho(rho_), u(u_), v(v_), w(w_), p(p_) {}
    HD Primitive(double rho_, double u_, double v_, double p_)
        : rho(rho_), u(u_), v(v_), w(0), p(p_) {}
};

static_assert(sizeof(Conserved)==5*sizeof(double),
              "Euler state must contain exactly five doubles");

HD inline Conserved operator+(const Conserved& a, const Conserved& b) {
    return {a.rho+b.rho, a.rhou+b.rhou, a.rhov+b.rhov, a.rhow+b.rhow,
            a.E+b.E};
}
HD inline Conserved operator-(const Conserved& a, const Conserved& b) {
    return {a.rho-b.rho, a.rhou-b.rhou, a.rhov-b.rhov, a.rhow-b.rhow,
            a.E-b.E};
}
HD inline Conserved operator*(double s, const Conserved& a) {
    return {s*a.rho, s*a.rhou, s*a.rhov, s*a.rhow, s*a.E};
}
HD inline Conserved operator*(const Conserved& a, double s) { return s*a; }
HD inline Conserved operator/(const Conserved& a, double s) {
    return (1.0/s)*a;
}
HD inline Conserved& operator+=(Conserved& a, const Conserved& b) {
    a.rho+=b.rho; a.rhou+=b.rhou; a.rhov+=b.rhov; a.rhow+=b.rhow; a.E+=b.E;
    return a;
}
HD inline Conserved& operator-=(Conserved& a, const Conserved& b) {
    a.rho-=b.rho; a.rhou-=b.rhou; a.rhov-=b.rhov; a.rhow-=b.rhow; a.E-=b.E;
    return a;
}
HD inline Conserved& operator*=(Conserved& a, double s) {
    a.rho*=s; a.rhou*=s; a.rhov*=s; a.rhow*=s; a.E*=s;
    return a;
}

HD inline Primitive operator+(const Primitive& a, const Primitive& b) {
    return {a.rho+b.rho, a.u+b.u, a.v+b.v, a.w+b.w, a.p+b.p};
}
HD inline Primitive operator-(const Primitive& a, const Primitive& b) {
    return {a.rho-b.rho, a.u-b.u, a.v-b.v, a.w-b.w, a.p-b.p};
}
HD inline Primitive operator*(double s, const Primitive& a) {
    return {s*a.rho, s*a.u, s*a.v, s*a.w, s*a.p};
}
HD inline Primitive operator*(const Primitive& a, double s) { return s*a; }
HD inline Primitive operator/(const Primitive& a, double s) {
    return (1.0/s)*a;
}
HD inline Primitive& operator+=(Primitive& a, const Primitive& b) {
    a.rho+=b.rho; a.u+=b.u; a.v+=b.v; a.w+=b.w; a.p+=b.p;
    return a;
}
HD inline Primitive& operator-=(Primitive& a, const Primitive& b) {
    a.rho-=b.rho; a.u-=b.u; a.v-=b.v; a.w-=b.w; a.p-=b.p;
    return a;
}
HD inline Primitive& operator*=(Primitive& a, double s) {
    a.rho*=s; a.u*=s; a.v*=s; a.w*=s; a.p*=s;
    return a;
}
