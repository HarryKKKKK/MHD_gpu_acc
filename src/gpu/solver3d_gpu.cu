#include "gpu/solver3d_gpu.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "gpu/boundary3d_gpu.cuh"
#include "physics.hpp"

namespace {

constexpr int DT_BLOCK=256;
constexpr int AXIS_X=0,AXIS_Y=1,AXIS_Z=2;

__device__ double minmod_scalar_3d(double a,double b) {
    if(fabs(b)<1e-12 || a*b<=0.0) return 0.0;
    const double limited=fmin(fabs(a),fabs(b));
    return a>0.0?limited:-limited;
}

__device__ Conserved limited_slope_3d(const Conserved& l,const Conserved& c,
                                      const Conserved& r) {
    return Conserved(
        minmod_scalar_3d(c.rho-l.rho,r.rho-c.rho),
        minmod_scalar_3d(c.rhou-l.rhou,r.rhou-c.rhou),
        minmod_scalar_3d(c.rhov-l.rhov,r.rhov-c.rhov),
        minmod_scalar_3d(c.rhow-l.rhow,r.rhow-c.rhow),
        minmod_scalar_3d(c.E-l.E,r.E-c.E));
}

__device__ bool positive_3d(const Conserved& u) {
    const double m2=u.rhou*u.rhou+u.rhov*u.rhov+u.rhow*u.rhow;
    return u.rho>0.0 && u.rho*u.E-0.5*m2>0.0;
}

template<int Axis>
__device__ Conserved flux_axis(const Conserved& u) {
    if constexpr(Axis==AXIS_X) {
        return phys::flux_x(u);
    } else if constexpr(Axis==AXIS_Y) {
        return phys::flux_y(u);
    } else {
        return phys::flux_z(u);
    }
    // CUDA 11.4 can fail to prove that the if-constexpr chain is exhaustive.
    return Conserved{};
}

__device__ Conserved swap_xz_3d(const Conserved& u) {
    return Conserved(u.rho,u.rhow,u.rhov,u.rhou,u.E);
}

template<RiemannSolver Solver,int Axis>
__device__ Conserved riemann_axis(const Conserved& l,const Conserved& r) {
    if constexpr(Axis==AXIS_X) {
        return riemann_flux<Solver>(l,r,Direction::X);
    } else if constexpr(Axis==AXIS_Y) {
        return riemann_flux<Solver>(l,r,Direction::Y);
    } else {
        return swap_xz_3d(riemann_flux<Solver>(
            swap_xz_3d(l),swap_xz_3d(r),Direction::X));
    }
    // Unreachable for the three explicitly instantiated axes.
    return Conserved{};
}

template<int Axis>
__device__ void shift_index(int& i,int& j,int& k,int offset) {
    // Suppress CUDA 11.4 warnings for the two coordinates discarded by each
    // compile-time specialization.
    (void)i; (void)j; (void)k;
    if constexpr(Axis==AXIS_X) i+=offset;
    if constexpr(Axis==AXIS_Y) j+=offset;
    if constexpr(Axis==AXIS_Z) k+=offset;
}

template<int Axis>
__device__ void reconstruct_at(ConstGrid3DGPUView q,int i,int j,int k,
                               double dt_over_d,Conserved& left,Conserved& right) {
    int im=i,jm=j,km=k,ip=i,jp=j,kp=k;
    shift_index<Axis>(im,jm,km,-1);
    shift_index<Axis>(ip,jp,kp,+1);
    const Conserved um=q.cells[q.flat_index(im,jm,km)];
    const Conserved uc=q.cells[q.flat_index(i,j,k)];
    const Conserved up=q.cells[q.flat_index(ip,jp,kp)];
    const Conserved half=0.5*limited_slope_3d(um,uc,up);
    const Conserved ul=uc-half, ur=uc+half;
    const Conserved base=uc+0.5*dt_over_d*
        (flux_axis<Axis>(ul)-flux_axis<Axis>(ur));
    left=base-half; right=base+half;
    if(!positive_3d(left)||!positive_3d(right)) left=right=uc;
}

template<RiemannSolver Solver,int Axis>
__global__ void advance_axis_kernel(ConstGrid3DGPUView in,Grid3DGPUView out,
                                    double dt_over_d) {
    const int li=blockIdx.x*blockDim.x+threadIdx.x;
    const int lj=blockIdx.y*blockDim.y+threadIdx.y;
    const int lk=blockIdx.z*blockDim.z+threadIdx.z;
    if(li>=in.nx||lj>=in.ny||lk>=in.nz) return;
    const int i=in.i_begin()+li,j=in.j_begin()+lj,k=in.k_begin()+lk;

    int mi=i,mj=j,mk=k,pi=i,pj=j,pk=k;
    shift_index<Axis>(mi,mj,mk,-1);
    shift_index<Axis>(pi,pj,pk,+1);

    Conserved ml,mr,cl,cr,pl,pr;
    reconstruct_at<Axis>(in,mi,mj,mk,dt_over_d,ml,mr);
    reconstruct_at<Axis>(in,i,j,k,dt_over_d,cl,cr);
    reconstruct_at<Axis>(in,pi,pj,pk,dt_over_d,pl,pr);
    (void)ml;
    (void)pr;
    const Conserved fm=riemann_axis<Solver,Axis>(mr,cl);
    const Conserved fp=riemann_axis<Solver,Axis>(cr,pl);
    Conserved updated=in.cells[in.flat_index(i,j,k)]-dt_over_d*(fp-fm);
    out.cells[out.flat_index(i,j,k)]=updated;
}

template<int BLOCK>
__global__ void block_max_speed_kernel(ConstGrid3DGPUView q,double* maxima) {
    constexpr int WARPS=BLOCK/32;
    __shared__ double warp_max[WARPS];
    const int n=q.nx*q.ny*q.nz;
    const int p=blockIdx.x*BLOCK+threadIdx.x;
    double speed=0.0;
    if(p<n) {
        const int li=p%q.nx;
        const int t=p/q.nx;
        const int lj=t%q.ny,lk=t/q.ny;
        const Conserved u=q.cells[q.flat_index(q.i_begin()+li,q.j_begin()+lj,
                                               q.k_begin()+lk)];
        const Primitive v=phys::cons_to_prim(u);
        if(isfinite(v.rho)&&isfinite(v.p)&&v.rho>0.0&&v.p>0.0) {
            const double sx=phys::max_signal_speed_x(v);
            const double sy=phys::max_signal_speed_y(v);
            const double sz=phys::max_signal_speed_z(v);
            if(isfinite(sx)&&isfinite(sy)&&isfinite(sz))
                speed=fmax(sx,fmax(sy,sz));
        }
    }
    const unsigned mask=0xffffffffu;
    for(int offset=16;offset>0;offset/=2)
        speed=fmax(speed,__shfl_down_sync(mask,speed,offset));
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    if(lane==0) warp_max[warp]=speed;
    __syncthreads();
    if(warp==0) {
        double value=lane<WARPS?warp_max[lane]:0.0;
        for(int offset=16;offset>0;offset/=2)
            value=fmax(value,__shfl_down_sync(mask,value,offset));
        if(lane==0) maxima[blockIdx.x]=value;
    }
}

template<RiemannSolver Solver>
void advance_specialized(const Grid3DGPU& old,Grid3DGPU& ux,Grid3DGPU& uy,
                         Grid3DGPU& out,double dt,const BoundaryConfig3D& bc) {
    // 128 threads retain contiguous x-lane accesses.
    const dim3 threads(8,4,4);
    const dim3 blocks((old.nx()+threads.x-1)/threads.x,
                      (old.ny()+threads.y-1)/threads.y,
                      (old.nz()+threads.z-1)/threads.z);
    advance_axis_kernel<Solver,AXIS_X><<<blocks,threads>>>(
        make_view(old),make_view(ux),dt/old.dx());
    cuda3d_check(cudaGetLastError(),"advance x kernel");
    apply_boundary_y_gpu(ux,bc);
    advance_axis_kernel<Solver,AXIS_Y><<<blocks,threads>>>(
        make_view(static_cast<const Grid3DGPU&>(ux)),make_view(uy),
        dt/old.dy());
    cuda3d_check(cudaGetLastError(),"advance y kernel");
    apply_boundary_z_gpu(uy,bc);
    advance_axis_kernel<Solver,AXIS_Z><<<blocks,threads>>>(
        make_view(static_cast<const Grid3DGPU&>(uy)),make_view(out),
        dt/old.dz());
    cuda3d_check(cudaGetLastError(),"advance z kernel");
    apply_boundary_gpu(out,bc);
}

} // namespace

void set_gpu3d_physics_gamma(double g) {
    cuda3d_check(cudaMemcpyToSymbol(phys::d_gamma,&g,sizeof(g)),"set gamma");
    phys::gamma=g;
}
void init_gpu_workspace(GpuWorkspace3D& ws,const Grid3DGPU& q) {
    free_gpu_workspace(ws);
    ws.nx=q.nx();ws.ny=q.ny();ws.nz=q.nz();
    const int cells=q.nx()*q.ny()*q.nz();
    const int blocks=(cells+DT_BLOCK-1)/DT_BLOCK;
    cuda3d_check(cudaMalloc(&ws.block_max,static_cast<std::size_t>(blocks)*sizeof(double)),
                 "allocate block maxima");
    cuda3d_check(cudaMalloc(&ws.max_speed,sizeof(double)),"allocate maximum speed");
    cuda3d_check(cub::DeviceReduce::Max(nullptr,ws.reduce_tmp_bytes,
                                        ws.block_max,ws.max_speed,blocks),
                 "query CUB workspace");
    cuda3d_check(cudaMalloc(&ws.reduce_tmp,ws.reduce_tmp_bytes),
                 "allocate CUB workspace");
}
void free_gpu_workspace(GpuWorkspace3D& ws) {
    if(ws.block_max) cuda3d_check(cudaFree(ws.block_max),"free block maxima");
    if(ws.max_speed) cuda3d_check(cudaFree(ws.max_speed),"free maximum speed");
    if(ws.reduce_tmp) cuda3d_check(cudaFree(ws.reduce_tmp),"free CUB workspace");
    ws=GpuWorkspace3D{};
}

double compute_dt_gpu(const Grid3DGPU& q,GpuWorkspace3D& ws,double cfl) {
    if(ws.nx!=q.nx()||ws.ny!=q.ny()||ws.nz!=q.nz()||!ws.block_max)
        throw std::runtime_error("3D GPU workspace is not initialized");
    const int cells=q.nx()*q.ny()*q.nz();
    const int blocks=(cells+DT_BLOCK-1)/DT_BLOCK;
    block_max_speed_kernel<DT_BLOCK><<<blocks,DT_BLOCK>>>(
        make_view(q),ws.block_max);
    cuda3d_check(cudaGetLastError(),"CFL kernel");
    cuda3d_check(cub::DeviceReduce::Max(ws.reduce_tmp,ws.reduce_tmp_bytes,
                                        ws.block_max,ws.max_speed,blocks),
                 "CUB maximum reduction");
    double maximum=0;
    cuda3d_check(cudaMemcpy(&maximum,ws.max_speed,sizeof(double),
                           cudaMemcpyDeviceToHost),"download maximum speed");
    if(!std::isfinite(maximum)||maximum<=0)
        throw std::runtime_error("3D GPU CFL maximum speed is invalid");
    return cfl*std::min({q.dx(),q.dy(),q.dz()})/maximum;
}

void advance_gpu(const Grid3DGPU& old,Grid3DGPU& ux,Grid3DGPU& uy,
                 Grid3DGPU& out,GpuWorkspace3D& ws,double dt,
                 RiemannSolver solver,const BoundaryConfig3D& bc) {
    if(ws.nx!=old.nx()||ws.ny!=old.ny()||ws.nz!=old.nz())
        throw std::runtime_error("3D GPU workspace/grid mismatch");
    switch(solver) {
        case RiemannSolver::HLL:
            advance_specialized<RiemannSolver::HLL>(old,ux,uy,out,dt,bc);break;
        case RiemannSolver::HLLC:
            advance_specialized<RiemannSolver::HLLC>(old,ux,uy,out,dt,bc);break;
        case RiemannSolver::FORCE:
            advance_specialized<RiemannSolver::FORCE>(old,ux,uy,out,dt,bc);break;
    }
}
