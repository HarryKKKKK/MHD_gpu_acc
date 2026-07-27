#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cpu/boundary3d_cpu.hpp"
#include "cpu/grid3d_cpu.hpp"
#include "cpu/solver3d_cpu.hpp"
#include "gpu/boundary3d_gpu.cuh"
#include "gpu/grid3d_gpu.cuh"
#include "gpu/solver3d_gpu.cuh"
#include "physics.hpp"

namespace {

double component_error(const Conserved& a,const Conserved& b) {
    return std::max({std::fabs(a.rho-b.rho),std::fabs(a.rhou-b.rhou),
        std::fabs(a.rhov-b.rhov),std::fabs(a.rhow-b.rhow),
        std::fabs(a.E-b.E)});
}

Primitive initial(double x,double y,double z) {
    constexpr double pi=3.14159265358979323846;
    return Primitive(
        1.0+0.08*std::sin(2*pi*x)*std::cos(2*pi*y)*std::sin(2*pi*z),
        0.05*std::sin(2*pi*y),0.04*std::cos(2*pi*z),0.03*std::sin(2*pi*x),
        1.0+0.05*std::cos(2*pi*x)*std::cos(2*pi*y));
}

} // namespace

int main() {
    GpuWorkspace3D gpu_ws;
    try {
        constexpr int n=8,ng=2;
        phys::gamma=5.0/3.0;
        Grid3D cpu_old(n,n,n,ng,0,1,0,1,0,1);
        for(int k=0;k<cpu_old.total_nz();++k)
            for(int j=0;j<cpu_old.total_ny();++j)
                for(int i=0;i<cpu_old.total_nx();++i)
                    cpu_old(i,j,k)=phys::prim_to_cons(initial(
                        cpu_old.x_center(i),cpu_old.y_center(j),cpu_old.z_center(k)));
        const BoundaryConfig3D periodic{
            BoundaryType::Periodic,BoundaryType::Periodic,
            BoundaryType::Periodic,BoundaryType::Periodic,
            BoundaryType::Periodic,BoundaryType::Periodic};
        apply_boundary(cpu_old,periodic);
        Grid3D cpu_x=cpu_old,cpu_y=cpu_old,cpu_out=cpu_old;
        CpuWorkspace3D cpu_ws;cpu_ws.init(n,n,n);
        const double cpu_dt=compute_dt_cpu(cpu_old,0.2);

        Grid3DGPU gpu_old(n,n,n,ng,0,1,0,1,0,1);
        Grid3DGPU gpu_x(n,n,n,ng,0,1,0,1,0,1);
        Grid3DGPU gpu_y(n,n,n,ng,0,1,0,1,0,1);
        Grid3DGPU gpu_out(n,n,n,ng,0,1,0,1,0,1);
        gpu_old.upload_from_aos(cpu_old.data());
        set_gpu3d_physics_gamma(phys::gamma);
        init_gpu_workspace(gpu_ws,gpu_old);
        const double gpu_dt=compute_dt_gpu(gpu_old,gpu_ws,0.2);
        const double dt_error=std::fabs(cpu_dt-gpu_dt);
        if(dt_error>2e-14*std::max(1.0,std::fabs(cpu_dt))) {
            std::cerr<<"CPU/GPU CFL mismatch: cpu="<<cpu_dt<<" gpu="<<gpu_dt<<"\n";
            return 1;
        }

        const double dt=std::min(cpu_dt,gpu_dt);
        advance_cpu(cpu_old,cpu_x,cpu_y,cpu_out,dt,cpu_ws,
                    RiemannSolver::HLLC,periodic);
        advance_gpu(gpu_old,gpu_x,gpu_y,gpu_out,gpu_ws,dt,
                    RiemannSolver::HLLC,periodic);
        cuda3d_check(cudaDeviceSynchronize(),"GPU parity synchronization");
        std::vector<Conserved> downloaded;
        gpu_out.download_to_aos(downloaded);

        double max_error=0,max_reference=0;
        const int tx=cpu_out.total_nx(),ty=cpu_out.total_ny();
        for(int k=cpu_out.k_begin();k<cpu_out.k_end();++k)
            for(int j=cpu_out.j_begin();j<cpu_out.j_end();++j)
                for(int i=cpu_out.i_begin();i<cpu_out.i_end();++i) {
                    const std::size_t p=(static_cast<std::size_t>(k)*ty+j)*tx+i;
                    max_error=std::max(max_error,
                        component_error(cpu_out(i,j,k),downloaded[p]));
                    max_reference=std::max(max_reference,std::fabs(cpu_out(i,j,k).E));
                }
        const double tolerance=2e-10*std::max(1.0,max_reference);
        if(max_error>tolerance) {
            std::cerr<<"CPU/GPU 3D one-step parity failed: max error="
                     <<max_error<<" tolerance="<<tolerance<<"\n";
            return 1;
        }
        std::cout<<"CUDA 3D parity passed: dt error="<<dt_error
                 <<", state max error="<<max_error<<"\n";
        free_gpu_workspace(gpu_ws);
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"CUDA 3D test error: "<<e.what()<<"\n";
        try{free_gpu_workspace(gpu_ws);}catch(...){}
        return 1;
    }
}
