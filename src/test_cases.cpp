#include "test_cases.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {
constexpr double pi=3.14159265358979323846;
Conserved make_euler(double rho,double u,double v,double w,double p,double gamma) {
    return {rho,rho*u,rho*v,rho*w,
            p/(gamma-1.0)+0.5*rho*(u*u+v*v+w*w)};
}
Conserved kh_state(double x,double y) {
    const double u=5.0*(std::tanh(20.0*(y+0.5))-
                        (std::tanh(20.0*(y-0.5))+1.0));
    const double v=0.25*std::sin(2.0*pi*x)*
        (std::exp(-100.0*(y+0.5)*(y+0.5))-
         std::exp(-100.0*(y-0.5)*(y-0.5)));
    return make_euler(1.0,u,v,0.0,50.0,1.4);
}
Conserved shock_bubble_state(double x,double y) {
    constexpr double g=1.4,rho_air=1.29,rho_he=0.214,p0=1.01325e5,ms=1.22;
    constexpr double xc=0.035,yc=0.0445,radius=0.025,xs=0.005;
    const double vs=ms*std::sqrt(g*p0/rho_air);
    const double rho2=rho_air*(g+1)*ms*ms/((g-1)*ms*ms+2);
    const double p2=p0*(2*g*ms*ms-(g-1))/(g+1);
    if(x<xs)return make_euler(rho2,vs*(1-rho_air/rho2),0,0,p2,g);
    const double r2=(x-xc)*(x-xc)+(y-yc)*(y-yc);
    return make_euler(r2<radius*radius?rho_he:rho_air,0,0,0,p0,g);
}
Conserved blast_wave_state(double x,double y) {
    const double dx=x-0.5,dy=y-0.5;
    return make_euler(1,0,0,0,dx*dx+dy*dy<=0.01?100.0:1.0,1.4);
}
} // namespace

CaseId parse_case_id(const std::string& name) {
    if(name=="kelvin_helmholtz")return CaseId::KelvinHelmholtz;
    if(name=="shock_bubble")return CaseId::ShockBubble;
    if(name=="blast_wave")return CaseId::BlastWave;
    throw std::runtime_error("Unknown Euler test case: '"+name+
        "'. Valid names: kelvin_helmholtz, shock_bubble, blast_wave.");
}
std::string case_id_to_string(CaseId id) {
    switch(id) {
        case CaseId::KelvinHelmholtz:return "kelvin_helmholtz";
        case CaseId::ShockBubble:return "shock_bubble";
        case CaseId::BlastWave:return "blast_wave";
    }
    throw std::runtime_error("Unhandled Euler CaseId");
}
CaseConfig get_case_config(CaseId id) {
    if(id==CaseId::KelvinHelmholtz)
        return {200,400,2,0,1,-1,1,0.3,0.5,1.4,
                {BoundaryType::Periodic,BoundaryType::Periodic,
                 BoundaryType::Periodic,BoundaryType::Periodic}};
    if(id==CaseId::BlastWave)
        return {500,500,2,0,1,0,1,0.4,0.2,1.4,{}};
    CaseConfig cfg{500,197,2,0,0.225,0,0.089,0.4,0,1.4,{}};
    constexpr double g=1.4,p0=1.01325e5,rho=1.29,ms=1.22;
    constexpr double xc=0.035,radius=0.025,xs=0.005;
    const double vs=ms*std::sqrt(g*p0/rho);
    const double t0=(xc-radius-xs)/vs,tau=radius/vs;
    for(double t:{0.6,1.2,1.8,3.0,4.6,6.2,7.8,12.6,19.0}) {
        cfg.snapshot_times.push_back(t0+t*tau);
        std::ostringstream tag;
        tag<<"t"<<std::setw(3)<<std::setfill('0')
           <<static_cast<int>(std::round(10*t));
        cfg.snapshot_tags.push_back(tag.str());
    }
    cfg.t_end=cfg.snapshot_times.back();
    return cfg;
}
CaseConfig get_case_config(const std::string& n){return get_case_config(parse_case_id(n));}
CaseConfig get_n_case_config(CaseId id,int n) {
    if(n<1)throw std::runtime_error("resolution scale must be >= 1");
    auto c=get_case_config(id);c.nx*=n;c.ny*=n;return c;
}
CaseConfig get_n_case_config(const std::string& s,int n) {
    return get_n_case_config(parse_case_id(s),n);
}
Conserved initial_state_at(CaseId id,double x,double y) {
    switch(id) {
        case CaseId::KelvinHelmholtz:return kh_state(x,y);
        case CaseId::ShockBubble:return shock_bubble_state(x,y);
        case CaseId::BlastWave:return blast_wave_state(x,y);
    }
    throw std::runtime_error("Unhandled Euler CaseId");
}
Conserved initial_state_at(const std::string& s,double x,double y) {
    return initial_state_at(parse_case_id(s),x,y);
}
