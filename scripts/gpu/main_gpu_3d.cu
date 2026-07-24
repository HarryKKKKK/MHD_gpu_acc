#include <cuda_runtime.h>

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
#include "gpu/boundary3d_gpu.cuh"
#include "gpu/grid3d_gpu.cuh"
#include "gpu/solver3d_gpu.cuh"
#include "imtg3d_case.hpp"

namespace {

enum class Case3D { Blast, IMTG };

struct Options {
    int n=128,snapshots=-1;
    double t_end=-1.0,cfl=-1.0;
    std::string out;
    RiemannSolver solver=RiemannSolver::HLLD;
    Case3D test_case=Case3D::Blast;
    bool write=true;
};

Options parse_args(int argc,char** argv) {
    Options o;
    for(int a=1;a<argc;++a) {
        const std::string s=argv[a];
        auto value=[&] {
            if(a+1>=argc) throw std::runtime_error("missing value after "+s);
            return std::string(argv[++a]);
        };
        if(s=="--case") {
            const auto v=value();
            if(v=="blast")o.test_case=Case3D::Blast;
            else if(v=="imtg")o.test_case=Case3D::IMTG;
            else throw std::runtime_error("unknown case: "+v);
        }
        else if(s=="--resolution")o.n=std::stoi(value());
        else if(s=="--snapshots")o.snapshots=std::stoi(value());
        else if(s=="--t-end")o.t_end=std::stod(value());
        else if(s=="--cfl")o.cfl=std::stod(value());
        else if(s=="--out")o.out=value();
        else if(s=="--no-out")o.write=false;
        else if(s=="--solver") {
            const auto v=value();
            if(v=="hll")o.solver=RiemannSolver::HLL;
            else if(v=="hllc")o.solver=RiemannSolver::HLLC;
            else if(v=="hlld")o.solver=RiemannSolver::HLLD;
            else if(v=="force")o.solver=RiemannSolver::FORCE;
            else throw std::runtime_error("unknown solver: "+v);
        } else throw std::runtime_error("unknown argument: "+s);
    }
    const bool imtg=o.test_case==Case3D::IMTG;
    if(o.snapshots<0)o.snapshots=imtg?8:5;
    if(o.t_end<0)o.t_end=imtg?imtg3d::t_end:blast3d::t_end;
    if(o.cfl<0)o.cfl=imtg?imtg3d::recommended_cfl:blast3d::recommended_cfl;
    if(o.out.empty())o.out=imtg?"output/imtg3d_gpu":"output/blast3d_gpu";
    if(o.n<8||o.snapshots<1||o.t_end<=0||o.cfl<=0)
        throw std::runtime_error("require resolution>=8, snapshots>=1, t_end>0, cfl>0");
    return o;
}

std::size_t flat(int i,int j,int k,int tx,int ty) {
    return (static_cast<std::size_t>(k)*ty+j)*tx+i;
}

template<class T> void write_value(std::ofstream& f,const T& v) {
    f.write(reinterpret_cast<const char*>(&v),sizeof(v));
}

bool valid_primitive(const Primitive& v) {
    return v.rho>0.0 && v.p>0.0 &&
        std::isfinite(v.rho) && std::isfinite(v.p) &&
        std::isfinite(v.u) && std::isfinite(v.v) && std::isfinite(v.w) &&
        std::isfinite(v.Bx) && std::isfinite(v.By) && std::isfinite(v.Bz);
}

void write_snapshot(const Grid3DGPU& q,const std::string& dir,
                    const std::string& stem,int frame,double time) {
    std::vector<Conserved> host;
    q.download_to_aos(host);
    std::filesystem::create_directories(dir);
    std::ostringstream name;
    name<<dir<<"/"<<stem<<"_"<<std::setw(3)<<std::setfill('0')
        <<frame<<".mhd3d";
    const std::string tmp=name.str()+".tmp";
    std::ofstream f(tmp,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open "+tmp);
    const char magic[8]={'M','H','D','3','D','0','1','\0'};
    f.write(magic,8);
    const std::uint32_t nx=q.nx(),ny=q.ny(),nz=q.nz();
    write_value(f,nx);write_value(f,ny);write_value(f,nz);
    for(double v:{q.x_min(),q.x_max(),q.y_min(),q.y_max(),q.z_min(),q.z_max(),
                  time,phys::gamma}) write_value(f,v);
    const int tx=q.total_nx(),ty=q.total_ny();
    for(int k=q.k_begin();k<q.k_end();++k)
        for(int j=q.j_begin();j<q.j_end();++j)
            for(int i=q.i_begin();i<q.i_end();++i) {
                const Primitive v=phys::cons_to_prim(host[flat(i,j,k,tx,ty)]);
                if(!valid_primitive(v))
                    throw std::runtime_error(
                        "non-physical state while writing snapshot "
                        +std::to_string(frame));
                for(float x:{static_cast<float>(v.rho),static_cast<float>(v.p),
                             static_cast<float>(v.u),static_cast<float>(v.v),
                             static_cast<float>(v.w),static_cast<float>(v.Bx),
                             static_cast<float>(v.By),static_cast<float>(v.Bz)})
                    write_value(f,x);
            }
    if(!f) throw std::runtime_error("snapshot write failed");
    f.close();
    if(std::filesystem::exists(name.str()))std::filesystem::remove(name.str());
    std::filesystem::rename(tmp,name.str());
    std::cout<<"  snapshot "<<frame<<" at t="<<time<<" -> "<<name.str()<<"\n";
}

const char* solver_name(RiemannSolver s) {
    if(s==RiemannSolver::HLL)return "HLL";
    if(s==RiemannSolver::HLLC)return "HLLC";
    if(s==RiemannSolver::HLLD)return "HLLD";
    return "FORCE";
}

} // namespace

int main(int argc,char** argv) {
    GpuWorkspace3D ws;
    try {
        const Options o=parse_args(argc,argv);
        const bool imtg=o.test_case==Case3D::IMTG;
        const double lo=imtg?imtg3d::x_min:blast3d::x_min;
        const double hi=imtg?imtg3d::x_max:blast3d::x_max;
        const std::string stem=imtg?"imtg3d":"blast3d";
        int device=0;
        cuda3d_check(cudaGetDevice(&device),"cudaGetDevice");
        cudaDeviceProp prop{};
        cuda3d_check(cudaGetDeviceProperties(&prop,device),"cudaGetDeviceProperties");
        std::cout<<"=== CUDA 3D GLM-MHD case: "
                 <<(imtg?"IMTG":"magnetized blast")<<" ===\n"
                 <<"  GPU        : "<<prop.name<<"\n"
                 <<"  grid       : "<<o.n<<" x "<<o.n<<" x "<<o.n<<"\n"
                 <<"  solver     : "<<solver_name(o.solver)<<"\n"
                 <<"  boundaries : periodic\n";
        if(imtg) {
            std::cout
                 <<"  reference  : Pouquet et al., arXiv:0906.1384, IMTG\n"
                 <<"  model      : compressible ideal-MHD counterpart\n"
                 <<"  domain     : [0,2*pi]^3\n"
                 <<"  v0 / b0    : 1 / 1/sqrt(3), EV=EM=0.125\n"
                 <<"  rho / p    : 1 / "<<imtg3d::thermal_pressure<<"\n";
        } else {
            std::cout
                 <<"  reference  : Derigs et al., JCP 317 (2016), Sec. 5.6\n"
                 <<"  B0         : ("<<blast3d::magnetic_field_x()<<",0,0)\n"
                 <<"  pressure   : p_inner=1000, p_outer=0.1\n"
                 <<"  radii      : r_inner=0.09, r_outer=0.10\n";
        }
        std::cout<<"  t_end      : "<<o.t_end<<"\n";

        phys::gamma=imtg?imtg3d::gamma:blast3d::gamma;
        const int ng=2,tx=o.n+2*ng,ty=o.n+2*ng,tz=o.n+2*ng;
        std::vector<Conserved> initial(static_cast<std::size_t>(tx)*ty*tz);
        const double d=(hi-lo)/o.n;
        for(int k=0;k<tz;++k)for(int j=0;j<ty;++j)for(int i=0;i<tx;++i)
            initial[flat(i,j,k,tx,ty)]=imtg
                ? imtg3d::initial_state(
                    lo+(i-ng+0.5)*d,lo+(j-ng+0.5)*d,lo+(k-ng+0.5)*d)
                : blast3d::initial_state(
                    lo+(i-ng+0.5)*d,lo+(j-ng+0.5)*d,lo+(k-ng+0.5)*d);

        Grid3DGPU old(o.n,o.n,o.n,ng,lo,hi,lo,hi,lo,hi);
        Grid3DGPU ux(o.n,o.n,o.n,ng,lo,hi,lo,hi,lo,hi);
        Grid3DGPU uy(o.n,o.n,o.n,ng,lo,hi,lo,hi,lo,hi);
        Grid3DGPU next(o.n,o.n,o.n,ng,lo,hi,lo,hi,lo,hi);
        old.upload_from_aos(initial);
        const BoundaryConfig3D bc=imtg
            ? imtg3d::boundary_conditions()
            : blast3d::boundary_conditions();
        apply_boundary_gpu(old,bc);
        set_gpu3d_physics_gamma(phys::gamma);
        set_gpu3d_physics_ch(0.0);
        init_gpu_workspace(ws,old);

        std::vector<double> targets;
        for(int s=0;s<=o.snapshots;++s)
            targets.push_back(o.t_end*static_cast<double>(s)/o.snapshots);
        if(o.write)write_snapshot(old,o.out,stem,0,0);

        double t=0;int step=0,frame=1;
        const auto start=std::chrono::steady_clock::now();
        while(t<o.t_end-1e-14) {
            const double raw=compute_dt_gpu(old,ws,o.cfl);
            const double dt=std::min(raw,targets[static_cast<std::size_t>(frame)]-t);
            if(!std::isfinite(dt)||dt<=0)throw std::runtime_error("invalid timestep");
            advance_gpu(old,ux,uy,next,ws,dt,o.solver,bc);
            old.swap(next);t+=dt;++step;
            if(t>=targets[static_cast<std::size_t>(frame)]-1e-12) {
                if(o.write)write_snapshot(old,o.out,stem,frame,t);
                ++frame;
            }
            if(step%20==0||t>=o.t_end-1e-14)
                std::cout<<"  step "<<std::setw(5)<<step<<" t="<<std::fixed
                         <<std::setprecision(5)<<t<<" dt="<<std::scientific<<dt<<"\n";
        }
        cuda3d_check(cudaDeviceSynchronize(),"final synchronize");
        const double elapsed=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-start).count();
        const double mcells=static_cast<double>(step)*o.n*o.n*o.n/elapsed/1e6;
        std::cout<<"[GPU3D] nx="<<o.n<<" ny="<<o.n<<" nz="<<o.n
                 <<" steps="<<step<<" elapsed_s="<<elapsed
                 <<" Mcell_updates_s="<<mcells<<"\n";
        if(o.write)std::cout<<"Plot with: python3 visualization/plot_blast3d_volume.py"
                              " --input "<<o.out<<"\n";
        free_gpu_workspace(ws);
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"Error: "<<e.what()<<"\n";
        try { free_gpu_workspace(ws); } catch(...) {}
        return 1;
    }
}
