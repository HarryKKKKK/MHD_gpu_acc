#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "blast3d_case.hpp"
#include "gpu/boundary3d_gpu.cuh"
#include "gpu/grid3d_gpu.cuh"
#include "gpu/solver3d_gpu.cuh"

namespace {
struct Options {
    int n=128;double t_end=blast3d::t_end,cfl=blast3d::recommended_cfl;
    RiemannSolver solver=RiemannSolver::HLLC;
};
Options parse_args(int argc,char** argv) {
    Options o;
    for(int a=1;a<argc;++a) {
        const std::string s=argv[a];
        auto value=[&]{if(++a>=argc)throw std::runtime_error("missing value after "+s);
                       return std::string(argv[a]);};
        if(s=="--case") {
            if(value()!="blast")throw std::runtime_error("only Euler case 'blast' is available");
        } else if(s=="--resolution")o.n=std::stoi(value());
        else if(s=="--t-end")o.t_end=std::stod(value());
        else if(s=="--cfl")o.cfl=std::stod(value());
        else if(s=="--snapshots"||s=="--out")value();
        else if(s=="--no-out"){}
        else if(s=="--solver") {
            const auto v=value();
            if(v=="hll")o.solver=RiemannSolver::HLL;
            else if(v=="hllc")o.solver=RiemannSolver::HLLC;
            else if(v=="force")o.solver=RiemannSolver::FORCE;
            else throw std::runtime_error("solver must be hll, hllc, or force");
        } else throw std::runtime_error("unknown argument: "+s);
    }
    if(o.n<8||o.t_end<=0||o.cfl<=0)throw std::runtime_error("invalid run options");
    return o;
}
std::size_t flat(int i,int j,int k,int tx,int ty) {
    return (static_cast<std::size_t>(k)*ty+j)*tx+i;
}
} // namespace

int main(int argc,char** argv) {
    GpuWorkspace3D ws;
    try {
        const Options o=parse_args(argc,argv);
        int device=0;cuda3d_check(cudaGetDevice(&device),"cudaGetDevice");
        cudaDeviceProp prop{};cuda3d_check(cudaGetDeviceProperties(&prop,device),
                                          "cudaGetDeviceProperties");
        const std::size_t extent=static_cast<std::size_t>(o.n)+4;
        const std::size_t bytes=4*extent*extent*extent*sizeof(Conserved);
        if(bytes>prop.totalGlobalMem*85/100)
            throw std::runtime_error("four Euler state grids exceed safe GPU memory");
        phys::gamma=blast3d::gamma;set_gpu3d_physics_gamma(phys::gamma);
        const int ng=2,tx=o.n+4,ty=o.n+4,tz=o.n+4;
        const double d=(blast3d::x_max-blast3d::x_min)/o.n;
        std::vector<Conserved> initial(static_cast<std::size_t>(tx)*ty*tz);
        for(int k=0;k<tz;++k)for(int j=0;j<ty;++j)for(int i=0;i<tx;++i)
            initial[flat(i,j,k,tx,ty)]=blast3d::initial_state(
                blast3d::x_min+(i-ng+0.5)*d,
                blast3d::x_min+(j-ng+0.5)*d,
                blast3d::x_min+(k-ng+0.5)*d);
        Grid3DGPU old(o.n,o.n,o.n,ng,blast3d::x_min,blast3d::x_max,
                     blast3d::x_min,blast3d::x_max,blast3d::x_min,blast3d::x_max);
        Grid3DGPU ux(o.n,o.n,o.n,ng,blast3d::x_min,blast3d::x_max,
                    blast3d::x_min,blast3d::x_max,blast3d::x_min,blast3d::x_max);
        Grid3DGPU uy(o.n,o.n,o.n,ng,blast3d::x_min,blast3d::x_max,
                    blast3d::x_min,blast3d::x_max,blast3d::x_min,blast3d::x_max);
        Grid3DGPU next(o.n,o.n,o.n,ng,blast3d::x_min,blast3d::x_max,
                      blast3d::x_min,blast3d::x_max,blast3d::x_min,blast3d::x_max);
        old.upload_from_aos(initial);const auto bc=blast3d::boundary_conditions();
        apply_boundary_gpu(old,bc);init_gpu_workspace(ws,old);
        double t=0;int step=0;const auto start=std::chrono::steady_clock::now();
        while(t<o.t_end-1e-14) {
            const double dt=std::min(compute_dt_gpu(old,ws,o.cfl),o.t_end-t);
            advance_gpu(old,ux,uy,next,ws,dt,o.solver,bc);
            old.swap(next);t+=dt;++step;
        }
        cuda3d_check(cudaDeviceSynchronize(),"final synchronize");
        const double seconds=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-start).count();
        std::cout<<"[GPU3D] nx="<<o.n<<" ny="<<o.n<<" nz="<<o.n
                 <<" steps="<<step<<" elapsed_s="<<seconds
                 <<" Mcell_updates_s="<<static_cast<double>(step)*o.n*o.n*o.n/
                    seconds/1e6<<"\n";
        free_gpu_workspace(ws);return 0;
    } catch(const std::exception& e) {
        std::cerr<<"Error: "<<e.what()<<"\n";
        try{free_gpu_workspace(ws);}catch(...){}return 1;
    }
}
