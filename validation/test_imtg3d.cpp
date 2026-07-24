#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "imtg3d_case.hpp"

int main() {
    phys::gamma=imtg3d::gamma;
    constexpr int n=32;
    const double h=(imtg3d::x_max-imtg3d::x_min)/n;
    double ev=0.0,em=0.0,max_div_v=0.0,max_div_b=0.0;

    auto primitive=[](double x,double y,double z) {
        return phys::cons_to_prim(imtg3d::initial_state(x,y,z));
    };

    for(int k=0;k<n;++k) for(int j=0;j<n;++j) for(int i=0;i<n;++i) {
        const double x=(i+0.5)*h,y=(j+0.5)*h,z=(k+0.5)*h;
        const Primitive q=primitive(x,y,z);
        ev+=0.5*q.rho*(q.u*q.u+q.v*q.v+q.w*q.w);
        em+=0.5*(q.Bx*q.Bx+q.By*q.By+q.Bz*q.Bz);

        const Primitive xp=primitive(x+h,y,z),xm=primitive(x-h,y,z);
        const Primitive yp=primitive(x,y+h,z),ym=primitive(x,y-h,z);
        const Primitive zp=primitive(x,y,z+h),zm=primitive(x,y,z-h);
        const double div_v=(xp.u-xm.u+yp.v-ym.v+zp.w-zm.w)/(2.0*h);
        const double div_b=(xp.Bx-xm.Bx+yp.By-ym.By+zp.Bz-zm.Bz)/(2.0*h);
        max_div_v=std::max(max_div_v,std::abs(div_v));
        max_div_b=std::max(max_div_b,std::abs(div_b));
    }

    const double cells=static_cast<double>(n)*n*n;
    ev/=cells; em/=cells;
    if(std::abs(ev-0.125)>1e-12 || std::abs(em-0.125)>1e-12)
        throw std::runtime_error("IMTG initial energy normalization failed");
    if(max_div_v>1e-12 || max_div_b>1e-12)
        throw std::runtime_error("IMTG discrete divergence check failed");

    const BoundaryConfig3D bc=imtg3d::boundary_conditions();
    if(bc.x_min!=BoundaryType::Periodic || bc.x_max!=BoundaryType::Periodic ||
       bc.y_min!=BoundaryType::Periodic || bc.y_max!=BoundaryType::Periodic ||
       bc.z_min!=BoundaryType::Periodic || bc.z_max!=BoundaryType::Periodic)
        throw std::runtime_error("IMTG boundaries are not all periodic");

    std::cout<<"IMTG checks passed: EV="<<ev<<", EM="<<em
             <<", max div(v)="<<max_div_v
             <<", max div(B)="<<max_div_b<<"\n";
    return 0;
}
