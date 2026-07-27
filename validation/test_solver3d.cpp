#include <algorithm>
#include <cmath>
#include <iostream>

#include "cpu/boundary3d_cpu.hpp"
#include "cpu/solver3d_cpu.hpp"
#include "physics.hpp"

namespace {
double max_component_error(const Conserved& a,const Conserved& b) {
    return std::max({std::fabs(a.rho-b.rho),std::fabs(a.rhou-b.rhou),
        std::fabs(a.rhov-b.rhov),std::fabs(a.rhow-b.rhow),
        std::fabs(a.E-b.E)});
}
}

int main() {
    phys::gamma=5.0/3.0;
    const Conserved uniform=phys::prim_to_cons(
        Primitive(1.2,0.2,-0.1,0.3,1.1));
    Grid3D old(8,7,6,2,-1,1,-1,1,-1,1);
    old.fill(uniform);
    const BoundaryConfig3D periodic{
        BoundaryType::Periodic,BoundaryType::Periodic,
        BoundaryType::Periodic,BoundaryType::Periodic,
        BoundaryType::Periodic,BoundaryType::Periodic};
    apply_boundary(old,periodic);
    Grid3D ux=old,uy=old,next=old;
    CpuWorkspace3D ws; ws.init(old.nx(),old.ny(),old.nz());
    const double dt=compute_dt_cpu(old,0.25);
    advance_cpu(old,ux,uy,next,dt,ws,RiemannSolver::HLLC,periodic);

    double error=0;
    for(int k=next.k_begin();k<next.k_end();++k)
        for(int j=next.j_begin();j<next.j_end();++j)
            for(int i=next.i_begin();i<next.i_end();++i)
                error=std::max(error,max_component_error(next(i,j,k),uniform));
    if(error>2e-12) {
        std::cerr<<"uniform-state preservation failed: max error="<<error<<"\n";
        return 1;
    }

    const Conserved hydro=phys::prim_to_cons(
        Primitive(2.0,0.2,-0.1,0.3,1.0));
    const Conserved fz=phys::flux_z(hydro);
    if(std::fabs(fz.rho-0.6)>1e-14 || std::fabs(fz.rhou-0.12)>1e-14 ||
       std::fabs(fz.rhov+0.06)>1e-14 || std::fabs(fz.rhow-1.18)>1e-14) {
        std::cerr<<"z-flux component test failed\n";
        return 1;
    }

    const Conserved contact_left=phys::prim_to_cons(Primitive(1.0,0,0,0,1.0));
    const Conserved contact_right=phys::prim_to_cons(Primitive(2.0,0,0,0,1.0));
    const Conserved contact_flux=riemann_flux(
        contact_left,contact_right,Direction::X,RiemannSolver::HLLC);
    if(std::fabs(contact_flux.rho)>1e-14 ||
       std::fabs(contact_flux.rhou-1.0)>1e-14 ||
       std::fabs(contact_flux.rhov)>1e-14 ||
       std::fabs(contact_flux.rhow)>1e-14 ||
       std::fabs(contact_flux.E)>1e-14) {
        std::cerr<<"HLLC stationary-contact test failed\n";
        return 1;
    }
    std::cout<<"3D solver checks passed; uniform max error="<<error<<"\n";
    return 0;
}
