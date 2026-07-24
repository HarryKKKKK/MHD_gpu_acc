// 3D GLM-MHD demonstration: a spherical over-pressure expands through a
// uniform x-directed magnetic field.
//
// Usage:
//   ./bin/main_cpu_3d [--resolution 48] [--t-end 0.01]
//                     [--snapshots 5] [--solver hll|hllc|hlld|force]
//                     [--out output/blast3d] [--no-out]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "blast3d_case.hpp"
#include "cpu/grid3d_cpu.hpp"
#include "cpu/solver3d_cpu.hpp"

namespace {

struct Options {
    int n=48;
    int snapshots=5;
    double t_end=blast3d::t_end;
    double cfl=blast3d::recommended_cfl;
    std::string out="output/blast3d";
    RiemannSolver solver=RiemannSolver::HLLD;
    bool write=true;
};

Options parse_args(int argc,char** argv) {
    Options o;
    for(int a=1;a<argc;++a) {
        const std::string s=argv[a];
        auto value=[&]() -> std::string {
            if(a+1>=argc) throw std::runtime_error("missing value after "+s);
            return argv[++a];
        };
        if(s=="--resolution") o.n=std::stoi(value());
        else if(s=="--t-end") o.t_end=std::stod(value());
        else if(s=="--snapshots") o.snapshots=std::stoi(value());
        else if(s=="--cfl") o.cfl=std::stod(value());
        else if(s=="--out") o.out=value();
        else if(s=="--no-out") o.write=false;
        else if(s=="--solver") {
            const auto v=value();
            if(v=="hll") o.solver=RiemannSolver::HLL;
            else if(v=="hllc") o.solver=RiemannSolver::HLLC;
            else if(v=="hlld") o.solver=RiemannSolver::HLLD;
            else if(v=="force") o.solver=RiemannSolver::FORCE;
            else throw std::runtime_error("unknown solver: "+v);
        } else throw std::runtime_error("unknown argument: "+s);
    }
    if(o.n<8 || o.snapshots<1 || o.t_end<=0 || o.cfl<=0)
        throw std::runtime_error("require resolution>=8, snapshots>=1, t_end>0, cfl>0");
    return o;
}

template<class T> void write_value(std::ofstream& f,const T& v) {
    f.write(reinterpret_cast<const char*>(&v),sizeof(T));
}

void write_snapshot(const Grid3D& q,const std::string& dir,int index,double time) {
    std::filesystem::create_directories(dir);
    std::ostringstream name;
    name<<dir<<"/blast3d_"<<std::setw(3)<<std::setfill('0')<<index<<".mhd3d";
    const std::string tmp=name.str()+".tmp";
    std::ofstream f(tmp,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open "+tmp);
    const char magic[8]={'M','H','D','3','D','0','1','\0'};
    f.write(magic,8);
    const std::uint32_t nx=q.nx(),ny=q.ny(),nz=q.nz();
    write_value(f,nx); write_value(f,ny); write_value(f,nz);
    for(double v:{q.x_min(),q.x_max(),q.y_min(),q.y_max(),
                  q.z_min(),q.z_max(),time,phys::gamma}) write_value(f,v);
    for(int k=q.k_begin();k<q.k_end();++k)
        for(int j=q.j_begin();j<q.j_end();++j)
            for(int i=q.i_begin();i<q.i_end();++i) {
                const Conserved& u=q(i,j,k);
                const Primitive v=phys::cons_to_prim(u);
                for(float field:{static_cast<float>(v.rho),static_cast<float>(v.p),
                                 static_cast<float>(v.u),static_cast<float>(v.v),
                                 static_cast<float>(v.w),static_cast<float>(v.Bx),
                                 static_cast<float>(v.By),static_cast<float>(v.Bz)})
                    write_value(f,field);
            }
    if(!f) throw std::runtime_error("write failed: "+tmp);
    f.close();
    if(std::filesystem::exists(name.str())) std::filesystem::remove(name.str());
    std::filesystem::rename(tmp,name.str());
    std::cout<<"  snapshot "<<index<<" at t="<<time<<" -> "<<name.str()<<"\n";
}

} // namespace

int main(int argc,char** argv) {
    try {
        const Options o=parse_args(argc,argv);
        phys::gamma=blast3d::gamma;
        Grid3D old(o.n,o.n,o.n,2,
                   blast3d::x_min,blast3d::x_max,
                   blast3d::x_min,blast3d::x_max,
                   blast3d::x_min,blast3d::x_max);
        for(int k=0;k<old.total_nz();++k)
            for(int j=0;j<old.total_ny();++j)
                for(int i=0;i<old.total_nx();++i)
                    old(i,j,k)=blast3d::initial_state(
                        old.x_center(i),old.y_center(j),old.z_center(k));
        const BoundaryConfig3D bc=blast3d::boundary_conditions();
        apply_boundary(old,bc);
        Grid3D ux=old,uy=old,next=old;
        CpuWorkspace3D ws; ws.init(o.n,o.n,o.n);

        std::vector<double> targets;
        for(int s=0;s<=o.snapshots;++s)
            targets.push_back(o.t_end*static_cast<double>(s)/o.snapshots);
        if(o.write) write_snapshot(old,o.out,0,0.0);

        std::cout<<"=== 3D magnetized blast wave ===\n"
                 <<"  grid   : "<<o.n<<" x "<<o.n<<" x "<<o.n<<"\n"
                 <<"  domain : [-0.5,0.5]^3\n"
                 <<"  reference: Derigs et al., JCP 317 (2016), Sec. 5.6\n"
                 <<"  B0     : ("<<blast3d::magnetic_field_x()
                 <<",0,0),  p_inner/p_outer=10000\n"
                 <<"  radii  : r_inner=0.09, r_outer=0.10\n"
                 <<"  gamma  : "<<phys::gamma<<", periodic boundaries\n"
                 <<"  t_end  : "<<o.t_end<<"\n";
        const auto start=std::chrono::steady_clock::now();
        int step=0,snapshot=1;
        double t=0;
        while(t<o.t_end-1e-14) {
            const double raw=compute_dt_cpu(old,o.cfl);
            const double target=targets[static_cast<std::size_t>(snapshot)];
            const double dt=std::min(raw,target-t);
            if(!std::isfinite(dt)||dt<=0) throw std::runtime_error("invalid timestep");
            advance_cpu(old,ux,uy,next,dt,ws,o.solver,bc);
            std::swap(old,next); t+=dt; ++step;
            if(t>=target-1e-12) {
                if(o.write) write_snapshot(old,o.out,snapshot,t);
                ++snapshot;
            }
            if(step%10==0 || t>=o.t_end-1e-14)
                std::cout<<"  step "<<std::setw(5)<<step<<"  t="<<std::fixed
                         <<std::setprecision(5)<<t<<"  dt="<<std::scientific<<dt<<"\n";
        }
        const double seconds=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-start).count();
        std::cout<<"Done: "<<step<<" steps, "<<seconds<<" s, "
                 <<(static_cast<double>(step)*o.n*o.n*o.n/seconds/1e6)
                 <<" Mcell-updates/s\n";
        if(o.write)
            std::cout<<"Plot with: python visualization/plot_blast3d_volume.py --input "
                     <<o.out<<"\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"Error: "<<e.what()<<"\n";
        return 1;
    }
}
