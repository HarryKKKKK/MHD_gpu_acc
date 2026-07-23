#include "cpu/solver3d_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "physics.hpp"

namespace {

enum class Axis3D { X, Y, Z };

inline double minmod(double a, double b) {
    if (std::fabs(b) < 1.0e-12 || a*b <= 0.0) return 0.0;
    return std::copysign(std::min(std::fabs(a), std::fabs(b)), a);
}

inline Conserved limited_slope(const Conserved& l, const Conserved& c,
                               const Conserved& r) {
    return Conserved(
        minmod(c.rho-l.rho,r.rho-c.rho),
        minmod(c.rhou-l.rhou,r.rhou-c.rhou),
        minmod(c.rhov-l.rhov,r.rhov-c.rhov),
        minmod(c.rhow-l.rhow,r.rhow-c.rhow),
        minmod(c.Bx-l.Bx,r.Bx-c.Bx),
        minmod(c.By-l.By,r.By-c.By),
        minmod(c.Bz-l.Bz,r.Bz-c.Bz),
        minmod(c.E-l.E,r.E-c.E),
        minmod(c.psi-l.psi,r.psi-c.psi));
}

inline bool positive(const Conserved& u) {
    const double m2 = u.rhou*u.rhou + u.rhov*u.rhov + u.rhow*u.rhow;
    const double magnetic = 0.5*(u.Bx*u.Bx + u.By*u.By + u.Bz*u.Bz);
    return u.rho > 0.0 && u.rho*(u.E-magnetic) - 0.5*m2 > 0.0;
}

inline Conserved physical_flux_3d(const Conserved& u, Axis3D axis) {
    if (axis == Axis3D::X) return phys::flux_x(u, phys::ch_glm);
    if (axis == Axis3D::Y) return phys::flux_y(u, phys::ch_glm);
    return phys::flux_z(u, phys::ch_glm);
}

inline Conserved swap_xz(const Conserved& u) {
    return Conserved(u.rho, u.rhow, u.rhov, u.rhou,
                     u.Bz, u.By, u.Bx, u.E, u.psi);
}

inline Conserved riemann_flux_3d(const Conserved& l, const Conserved& r,
                                 Axis3D axis, RiemannSolver solver) {
    if (axis == Axis3D::X)
        return riemann_flux(l, r, Direction::X, solver, phys::ch_glm);
    if (axis == Axis3D::Y)
        return riemann_flux(l, r, Direction::Y, solver, phys::ch_glm);
    // Reuse the thoroughly tested x-normal solvers after rotating z onto x.
    return swap_xz(riemann_flux(swap_xz(l), swap_xz(r), Direction::X,
                                solver, phys::ch_glm));
}

inline void reconstruct(const Conserved& um, const Conserved& uc,
                        const Conserved& up, double dt_over_d, Axis3D axis,
                        Conserved& left, Conserved& right) {
    const Conserved half = 0.5*limited_slope(um,uc,up);
    const Conserved ul = uc-half;
    const Conserved ur = uc+half;
    const Conserved base =
        uc + 0.5*dt_over_d*(physical_flux_3d(ul,axis)-physical_flux_3d(ur,axis));
    left=base-half;
    right=base+half;
    if (!positive(left) || !positive(right)) left=right=uc;
}

inline std::size_t cell_index(int i,int j,int k,int tx,int ty) {
    return (static_cast<std::size_t>(k)*ty+j)*tx+i;
}

void cache_state(const Grid3D& q, CpuWorkspace3D& ws) {
    ws.state = q.data();
    ws.recon_L.resize(ws.state.size());
    ws.recon_R.resize(ws.state.size());
}

void reconstruct_x(const Grid3D& q, double dtdx, CpuWorkspace3D& ws) {
    const int tx=q.total_nx(), ty=q.total_ny();
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for(int k=q.k_begin();k<q.k_end();++k)
        for(int j=q.j_begin();j<q.j_end();++j)
            for(int i=q.i_begin()-1;i<=q.i_end();++i) {
                const auto c=cell_index(i,j,k,tx,ty);
                reconstruct(ws.state[c-1],ws.state[c],ws.state[c+1],dtdx,
                            Axis3D::X,ws.recon_L[c],ws.recon_R[c]);
            }
}

void reconstruct_y(const Grid3D& q, double dtdy, CpuWorkspace3D& ws) {
    const int tx=q.total_nx(), ty=q.total_ny();
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for(int k=q.k_begin();k<q.k_end();++k)
        for(int j=q.j_begin()-1;j<=q.j_end();++j)
            for(int i=q.i_begin();i<q.i_end();++i) {
                const auto c=cell_index(i,j,k,tx,ty);
                reconstruct(ws.state[c-tx],ws.state[c],ws.state[c+tx],dtdy,
                            Axis3D::Y,ws.recon_L[c],ws.recon_R[c]);
            }
}

void reconstruct_z(const Grid3D& q, double dtdz, CpuWorkspace3D& ws) {
    const int tx=q.total_nx(), ty=q.total_ny();
    const std::size_t plane=static_cast<std::size_t>(tx)*ty;
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for(int k=q.k_begin()-1;k<=q.k_end();++k)
        for(int j=q.j_begin();j<q.j_end();++j)
            for(int i=q.i_begin();i<q.i_end();++i) {
                const auto c=cell_index(i,j,k,tx,ty);
                reconstruct(ws.state[c-plane],ws.state[c],ws.state[c+plane],dtdz,
                            Axis3D::Z,ws.recon_L[c],ws.recon_R[c]);
            }
}

void sweep_x(const Grid3D& in, Grid3D& out, double dtdx,
             CpuWorkspace3D& ws, RiemannSolver solver) {
    cache_state(in,ws); reconstruct_x(in,dtdx,ws);
    const int tx=in.total_nx(),ty=in.total_ny(), nxf=in.nx()+1;
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for(int k=in.k_begin();k<in.k_end();++k)
        for(int j=in.j_begin();j<in.j_end();++j)
            for(int i=in.i_begin()-1;i<in.i_end();++i) {
                const int lk=k-in.k_begin(),lj=j-in.j_begin(),f=i-(in.i_begin()-1);
                const auto a=cell_index(i,j,k,tx,ty), b=a+1;
                ws.face_flux[(static_cast<std::size_t>(lk)*in.ny()+lj)*nxf+f]=
                    riemann_flux_3d(ws.recon_R[a],ws.recon_L[b],Axis3D::X,solver);
            }
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for(int k=in.k_begin();k<in.k_end();++k)
        for(int j=in.j_begin();j<in.j_end();++j)
            for(int i=in.i_begin();i<in.i_end();++i) {
                const int lk=k-in.k_begin(),lj=j-in.j_begin(),f=i-in.i_begin();
                const auto base=(static_cast<std::size_t>(lk)*in.ny()+lj)*nxf;
                out(i,j,k)=in(i,j,k)-dtdx*(ws.face_flux[base+f+1]-ws.face_flux[base+f]);
            }
}

void sweep_y(const Grid3D& in, Grid3D& out, double dtdy,
             CpuWorkspace3D& ws, RiemannSolver solver) {
    cache_state(in,ws); reconstruct_y(in,dtdy,ws);
    const int tx=in.total_nx(),ty=in.total_ny(), nyf=in.ny()+1;
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for(int k=in.k_begin();k<in.k_end();++k)
        for(int j=in.j_begin()-1;j<in.j_end();++j)
            for(int i=in.i_begin();i<in.i_end();++i) {
                const int lk=k-in.k_begin(),f=j-(in.j_begin()-1),li=i-in.i_begin();
                const auto a=cell_index(i,j,k,tx,ty), b=a+tx;
                ws.face_flux[(static_cast<std::size_t>(lk)*nyf+f)*in.nx()+li]=
                    riemann_flux_3d(ws.recon_R[a],ws.recon_L[b],Axis3D::Y,solver);
            }
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for(int k=in.k_begin();k<in.k_end();++k)
        for(int j=in.j_begin();j<in.j_end();++j)
            for(int i=in.i_begin();i<in.i_end();++i) {
                const int lk=k-in.k_begin(),f=j-in.j_begin(),li=i-in.i_begin();
                const auto a=(static_cast<std::size_t>(lk)*nyf+f)*in.nx()+li;
                const auto b=(static_cast<std::size_t>(lk)*nyf+f+1)*in.nx()+li;
                out(i,j,k)=in(i,j,k)-dtdy*(ws.face_flux[b]-ws.face_flux[a]);
            }
}

void sweep_z(const Grid3D& in, Grid3D& out, double dtdz,
             CpuWorkspace3D& ws, RiemannSolver solver) {
    cache_state(in,ws); reconstruct_z(in,dtdz,ws);
    const int tx=in.total_nx(),ty=in.total_ny();
    const std::size_t plane=static_cast<std::size_t>(tx)*ty;
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for(int k=in.k_begin()-1;k<in.k_end();++k)
        for(int j=in.j_begin();j<in.j_end();++j)
            for(int i=in.i_begin();i<in.i_end();++i) {
                const int f=k-(in.k_begin()-1),lj=j-in.j_begin(),li=i-in.i_begin();
                const auto a=cell_index(i,j,k,tx,ty), b=a+plane;
                ws.face_flux[(static_cast<std::size_t>(f)*in.ny()+lj)*in.nx()+li]=
                    riemann_flux_3d(ws.recon_R[a],ws.recon_L[b],Axis3D::Z,solver);
            }
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for(int k=in.k_begin();k<in.k_end();++k)
        for(int j=in.j_begin();j<in.j_end();++j)
            for(int i=in.i_begin();i<in.i_end();++i) {
                const int f=k-in.k_begin(),lj=j-in.j_begin(),li=i-in.i_begin();
                const auto a=(static_cast<std::size_t>(f)*in.ny()+lj)*in.nx()+li;
                const auto b=(static_cast<std::size_t>(f+1)*in.ny()+lj)*in.nx()+li;
                out(i,j,k)=in(i,j,k)-dtdz*(ws.face_flux[b]-ws.face_flux[a]);
            }
}

} // namespace

double compute_dt_cpu(const Grid3D& q, double cfl) {
    double max_speed=0.0;
#ifdef _OPENMP
#pragma omp parallel for collapse(3) reduction(max:max_speed) schedule(static)
#endif
    for(int k=q.k_begin();k<q.k_end();++k)
        for(int j=q.j_begin();j<q.j_end();++j)
            for(int i=q.i_begin();i<q.i_end();++i) {
                const Primitive v=phys::cons_to_prim(q(i,j,k));
                if (!std::isfinite(v.rho)||!std::isfinite(v.p)||v.rho<=0||v.p<=0) continue;
                const double s=std::max({phys::max_signal_speed_x(v,0.0),
                                         phys::max_signal_speed_y(v,0.0),
                                         phys::max_signal_speed_z(v,0.0)});
                if(std::isfinite(s)) max_speed=std::max(max_speed,s);
            }
    if(max_speed<=0) throw std::runtime_error("compute_dt_cpu(3D): invalid wave speed");
    phys::ch_glm=max_speed;
    return cfl*std::min({q.dx(),q.dy(),q.dz()})/max_speed;
}

void advance_cpu(const Grid3D& old, Grid3D& ux, Grid3D& uy, Grid3D& out,
                 double dt, CpuWorkspace3D& ws, RiemannSolver solver,
                 const BoundaryConfig3D& bc) {
    if(!ws.is_initialized()) throw std::runtime_error("CpuWorkspace3D not initialized");
    sweep_x(old,ux,dt/old.dx(),ws,solver);
    apply_boundary_y(ux,bc);
    sweep_y(ux,uy,dt/old.dy(),ws,solver);
    apply_boundary_z(uy,bc);
    sweep_z(uy,out,dt/old.dz(),ws,solver);

    const double damping=phys::cr_glm>0
        ? std::exp(-dt*phys::ch_glm/phys::cr_glm) : 1.0;
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for(int k=out.k_begin();k<out.k_end();++k)
        for(int j=out.j_begin();j<out.j_end();++j)
            for(int i=out.i_begin();i<out.i_end();++i) out(i,j,k).psi*=damping;
    apply_boundary(out,bc);
}
